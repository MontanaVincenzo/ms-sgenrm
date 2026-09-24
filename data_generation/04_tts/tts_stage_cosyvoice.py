"""TTS synthesis stage for the ASR -> LLM -> TTS pipeline, using CosyVoice3
(zero-shot voice cloning) as the TTS backend instead of Qwen3-TTS VoiceDesign.

CosyVoice3 clones a reference speaker from an audio clip + its transcript
rather than accepting a text style instruction, so this stage does not run
any LLM-based perturbation step (see ``main``): it warns if the pipeline
config has ``perturb_input: true`` and otherwise just:
1. Passes the (unperturbed) LLM-output dataset through unchanged.
2. For every pipeline's target text, samples a random reference clip from
   ``--ref-audio-dir``, transcribes it with the ASR model (needed for
   CosyVoice3's cross-attention conditioning), and renders the target text
   in that cloned voice, writing the wav path back into the dataset.
"""

import gc
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from nemo.collections.asr.models import ASRModel
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

logger = logging.getLogger(__name__)

MIN_REF_SAMPLE_RATE = 16000


class TTS_Stage:
    def __init__(self, args):
        self.limit = args.limit
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir
        self.dataset_path = args.input
        self.output_dataset_path = args.output
        self.tts_model_name = args.tts_model
        self.tokenizer_path = args.tokenizer
        self.asr_model_name = "nvidia/canary-1b-v2"
        self.ref_audio_dir = args.ref_audio_dir
        self._tts = None
        self._asr_model = None
        self._ref_transcript_cache: dict[str, str] = {}
        self._rng = random.Random(args.seed)

        # vllm-omni entrypoint kwargs: forward the CLI namespace like the upstream
        # CosyVoice3 end2end example, minus this script's own options, and pin the model.
        self.omni_kwargs = vars(args).copy()
        for key in ("input", "output", "limit", "pipeline_config", "tts_model",
                    "asr_model", "ref_audio_dir", "seed"):
            self.omni_kwargs.pop(key, None)
        self.omni_kwargs["model"] = self.tts_model_name

        self._cv3_config = CosyVoice3Config()
        self._tokenizer = get_qwen_tokenizer(
            token_path=self.tokenizer_path,
            skip_special_tokens=self._cv3_config.skip_special_tokens,
            version=self._cv3_config.version,
        )

        self._ref_audio_paths = sorted(glob.glob(os.path.join(self.ref_audio_dir, "*.wav")))
        if not self._ref_audio_paths:
            raise ValueError(f"No .wav files found in --ref-audio-dir {self.ref_audio_dir!r}")

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_dataset(self, path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        if self.limit is not None:
            dataset = dataset[: self.limit]
        if not dataset:
            raise ValueError(f"Empty dataset at {path}")
        return dataset

    @property
    def asr_model(self):
        if self._asr_model is None:
            self._asr_model = ASRModel.from_pretrained(model_name=self.asr_model_name).eval()
        return self._asr_model

    def _free_asr(self):
        self._asr_model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def tts(self):
        if self._tts is None:
            self._tts = Omni(**self.omni_kwargs)
        return self._tts

    def _free_tts(self):
        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._tts = None
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _save_wav(output_dir: str, request_id: str, mm: dict) -> str:
        """Concatenate audio chunks and write to a wav file; return its path."""
        audio_data = mm["audio"]
        sr_raw = mm.get("sr", 24000)
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

        if isinstance(audio_data, list):
            audio_data = (
                torch.cat(audio_data, dim=-1)
                if torch.is_tensor(audio_data[0])
                else np.concatenate(audio_data, axis=-1)
            )
        audio_np = audio_data.float().cpu().numpy() if torch.is_tensor(audio_data) else np.asarray(audio_data)
        audio_np = audio_np.flatten()

        out_wav = os.path.join(output_dir, f"{request_id}_tts.wav")
        sf.write(out_wav, audio_np, samplerate=sr, format="WAV")
        logger.info(f"Request ID: {request_id}, Saved audio to {out_wav}")
        return out_wav

    def _transcribe_refs(self, paths: list[str]) -> None:
        """Batch-transcribe every distinct sampled reference clip with the ASR model.

        CosyVoice3 needs each reference clip's own transcript (``prompt_text``)
        for cross-attention conditioning, so a fixed/generic caption won't do.
        """
        uncached = sorted({p for p in paths if p not in self._ref_transcript_cache})
        if not uncached:
            return
        transcripts = (
            self.asr_model.transcribe(uncached, source_lang="en", target_lang="en")
            if self.asr_model_name == "nvidia/canary-1b-v2"
            else self.asr_model.transcribe(uncached)
        )
        for path, transcript in zip(uncached, transcripts):
            self._ref_transcript_cache[path] = transcript.text

    def get_query(self, data):
        """Build Omni inputs.

        Args:
            data: list of ``(uid, target_text, ref_audio_path)`` tuples.

        Returns:
            list of CosyVoice3 prompt dicts (``prompt`` + reference-audio
            ``multi_modal_data`` + ``mm_processor_kwargs``), parallel to ``data``.
        """
        inputs = []
        for _uid, text, ref_audio_path in data:
            audio_signal, sr = load_audio(ref_audio_path, sr=None)
            if sr < MIN_REF_SAMPLE_RATE:
                raise ValueError(
                    f"Reference audio {ref_audio_path!r} sample rate {sr} Hz is below "
                    f"the minimum required {MIN_REF_SAMPLE_RATE} Hz."
                )
            inputs.append(
                {
                    "prompt": text,
                    "multi_modal_data": {"audio": (audio_signal.astype(np.float32), sr)},
                    "mm_processor_kwargs": {
                        "prompt_text": self._ref_transcript_cache[ref_audio_path],
                        "sample_rate": sr,
                    },
                }
            )
        return inputs

    def build_inputs(self):
        """Build the ordered list of Omni requests from the LLM-output dataset.

        Returns one record per pipeline that has an LLM answer to synthesize:
        ``{"annotation_id", "pipeline_key", "omni_input"}``.
        """
        if self.batch_size < 1:
            raise ValueError(f"--batch-size must be >= 1 (got {self.batch_size})")

        dataset = self._load_dataset(self.output_dataset_path)

        meta = []      # (annotation_id, pipeline_key)
        triples = []   # (uid, target_text, ref_audio_path)
        for sample in dataset:
            anno_id = sample["annotation_id"]
            for pipeline_key in ["pipeline1", "pipeline2"]:
                target_text = sample[pipeline_key]["llm"]["output"]
                if not target_text:
                    continue  # LLM stage not run for this pipeline -> nothing to say

                ref_audio_path = self._rng.choice(self._ref_audio_paths)
                meta.append((anno_id, pipeline_key))
                triples.append((f"{anno_id}_{pipeline_key}", target_text, ref_audio_path))

        self._transcribe_refs([ref_audio_path for _, _, ref_audio_path in triples])

        return [
            {"annotation_id": anno_id, "pipeline_key": pipeline_key, "omni_input": omni_input}
            for (anno_id, pipeline_key), omni_input in zip(meta, self.get_query(triples))
        ]

    def passthrough_dataset(self):
        """Copy the LLM-output dataset through unchanged.

        CosyVoice3 has no text-style-instruction input, so unlike the
        Qwen3-TTS/OmniVoice variants of this stage there is no perturbation
        step to run here; this just seeds ``output_dataset_path`` the same
        way ``generate_perturbations()`` does for those variants.
        """
        dataset = self._load_dataset(self.dataset_path)
        with open(self.output_dataset_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _batch_sampling_params(self, texts: list[str]) -> list[SamplingParams]:
        """Build the (GPT, S2Mel) SamplingParams pair for one Omni batch.

        Omni takes one SamplingParams per pipeline stage for the whole batch
        (not per request), so ``min_tokens``/``max_tokens`` are derived from
        the shortest/longest target text in this batch -- the safe bounds
        that neither truncate the longest text nor force the shortest one
        to overrun.
        """
        token_lens = [extract_text_token(t, self._tokenizer, self._cv3_config.allowed_special)[1] for t in texts]
        min_len = int(min(token_lens) * self._cv3_config.min_token_text_ratio)
        max_len = int(max(token_lens) * self._cv3_config.max_token_text_ratio)

        gpt_sampling = SamplingParams(
            temperature=1.0,
            top_p=self._cv3_config.llm["sampling"]["top_p"],
            top_k=self._cv3_config.llm["sampling"]["top_k"],
            repetition_penalty=2.0,
            min_tokens=min_len,
            max_tokens=max_len,
            stop_token_ids=[self._cv3_config.llm["eos_token_id"]],
            detokenize=False,
        )
        # Not used by CosyVoice3's Code2Wav/S2Mel path but still required as a
        # stage placeholder, matching the upstream end2end example.
        s2mel_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=2.0,
            max_tokens=256,
            detokenize=False,
        )
        return [gpt_sampling, s2mel_sampling]

    def tts_synthesis(self, inputs):
        """Render each request to a wav file and record its path in the dataset."""
        dataset = self._load_dataset(self.output_dataset_path)
        by_id = {sample["annotation_id"]: sample for sample in dataset}

        batch_size = self.batch_size
        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]
            omni_batch = [record["omni_input"] for record in batch]
            sampling_params_list = self._batch_sampling_params(
                [record["omni_input"]["prompt"] for record in batch]
            )

            for output in self.tts.generate(omni_batch, sampling_params_list=sampling_params_list):
                # Omni yields outputs as they finish, not in submission order;
                # request_id is "<batch index>_<uuid>".
                local_idx = int(str(output.request_id).split("_", 1)[0])
                record = batch[local_idx]
                anno_id = record["annotation_id"]
                pipeline_key = record["pipeline_key"]
                mm = output.outputs[0].multimodal_output
                wav_path = self._save_wav(self.output_dir, f"{anno_id}_{pipeline_key}", mm)
                by_id[anno_id][pipeline_key]["tts"]["output"] = wav_path

        with open(self.output_dataset_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(args):
    """Run the CosyVoice3 TTS synthesis stage."""
    tts_stage = TTS_Stage(args)

    tts_stage.passthrough_dataset()      # no perturbation step for CosyVoice3

    inputs = tts_stage.build_inputs()    # transcribes sampled ref clips, loads ASR lazily
    tts_stage._free_asr()                # release the GPU before Omni loads

    tts_stage.tts_synthesis(inputs)      # loads Omni lazily
    tts_stage._free_tts()


def parse_args():
    parser = TrackingArgumentParser(description="vLLM-Omni CosyVoice3 TTS stage for offline inference")

    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output_audio",
        help="Output directory for generated wav files (default: data/output_audio).",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path. If unset, auto-loads the CosyVoice3 "
        "deploy config based on the HF model_type.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (e.g., <model_path>/CosyVoice-BlankEN).",
    )
    parser.add_argument(
        "--ref-audio-dir",
        type=str,
        required=True,
        help="Folder of .wav reference clips; one is sampled at random per request for voice cloning.",
    )
    # NOTE: named --pipeline-config (not --config[-path]) on purpose -- vLLM's
    # FlexibleArgumentParser reserves the "--config" prefix for its own CLI-args
    # config-file loading, which shadows any custom flag that starts with it.
    parser.add_argument(
        "--pipeline-config",
        required=True,
        help="Path to the pipeline YAML config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reference-clip sampling (default: random).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prompts per Omni batch (default: 64).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        with open(args.pipeline_config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.pipeline_config}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    if config.get("perturb_input"):
        logger.warning(
            "Pipeline config has perturb_input=true, but CosyVoice3 has no text-style-"
            "instruction input, so this stage never applies TTS perturbations; ignoring."
        )

    data_dir = Path(config["data_dir"])
    args.input = data_dir / config["llm_output_file"]
    args.output = data_dir / config["tts_output_file"]
    args.output_dir = str(data_dir / "output_audio")
    args.tts_model = config["tts_model"]
    args.asr_model = config["asr_model"]

    main(args)
