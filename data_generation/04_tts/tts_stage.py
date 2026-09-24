"""TTS perturbation + synthesis stage for the ASR -> LLM -> TTS pipeline.

``main`` runs two steps:
1. ``generate_perturbations()`` -- for pipelines whose planned TTS perturbation
   is ``wrong_emotion`` / ``added_words`` / ``missing_words``, ask the LLM for a
   steered voice-style instruction or a perturbed target text. Each perturbation
   family has its own JSON schema and therefore its own ``SamplingParams``; the
   two are mixed in one ``LLM.generate`` call via a per-prompt list.
2. ``tts_synthesis()`` -- render every pipeline's (possibly perturbed) text with
   Qwen3-TTS VoiceDesign and write the wav path back into the dataset.
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml

#os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from vllm_omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

logger = logging.getLogger(__name__)


TASK_TYPE = "VoiceDesign"

ADD_WORDS_INSTR = "Insert extra words into the target text (such as filler words like \"um\", \"you know\", extra adjectives, stutters or out of context words) to test how the TTS handles text changes and disfluencies."
OMITT_WORDS_INSTR ="Remove words from the target text (such as dropping articles, prepositions, or key words) to test how the TTS handles incomplete or telegraphic text."
with open("/home/vmontana/synthetic_data_generation/src/tts/prompts/style_perturbation/system.prompt", 'r') as f: 
    STYLE_PERTURBATION_SYSTEM_PROMPT = f.read()
with open("/home/vmontana/synthetic_data_generation/src/tts/prompts/style_perturbation/user.prompt", 'r') as f: 
    STYLE_PERTURBATION_USER_PROMPT = f.read()

with open("/home/vmontana/synthetic_data_generation/src/tts/prompts/text_perturbation/system.prompt", 'r') as f: 
    TEXT_PERTURBATION_SYSTEM_PROMPT = f.read()
with open("/home/vmontana/synthetic_data_generation/src/tts/prompts/text_perturbation/user.prompt", 'r') as f: 
    TEXT_PERTURBATION_USER_PROMPT = f.read()

REQUEST_SCHEMA_STYLE = {
    "type": "object",
    "properties": {
        "steered_style_instruction": {"type": "string"},
        "steering_impact": {"type": "string"},
    },
    "required": ["steered_style_instruction", "steering_impact"],
    "additionalProperties": False,
}
REQUIRED_KEYS_STYLE = tuple(REQUEST_SCHEMA_STYLE["required"])

REQUEST_SCHEMA_TEXT = {
    "type": "object",
    "properties": {
        "perturbed_text": {"type": "string"},
        "steering_impact": {"type": "string"},
    },
    "required": ["perturbed_text", "steering_impact"],
    "additionalProperties": False,
}
REQUIRED_KEYS_TEXT = tuple(REQUEST_SCHEMA_TEXT["required"])


class TTS_Stage:
    def __init__(self, args):
        self.limit = args.limit
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir
        self.dataset_path = args.input
        self.output_dataset_path = args.output
        self.llm_model_name = args.llm_model
        self.tts_model_name = args.tts_model
        self._tts = None
        self._llm = None

        # vllm-omni entrypoint kwargs: forward the CLI namespace like the upstream
        # end2end example, minus this script's own options, and pin the model.
        self.omni_kwargs = vars(args).copy()
        for key in ("input", "output", "limit", "config_path", "llm_model", "tts_model"):
            self.omni_kwargs.pop(key, None)
        self.omni_kwargs["model"] = self.tts_model_name

        # One JSON schema per perturbation family -> one SamplingParams each.
        # LLM.generate() accepts a per-prompt list that mixes the two.
        self.style_sampling_params = self._make_sampling_params(REQUEST_SCHEMA_STYLE)
        self.text_sampling_params = self._make_sampling_params(REQUEST_SCHEMA_TEXT)

        self.processor = AutoProcessor.from_pretrained(self.llm_model_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _make_sampling_params(schema: dict) -> SamplingParams:
        return SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.80,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            # Constrain every token to a valid instance of `schema`; no <think>
            # block is emitted and no regex salvage is needed.
            structured_outputs=StructuredOutputsParams(json=schema),
        )

    def _load_dataset(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        if self.limit is not None:
            dataset = dataset[: self.limit]
        if not dataset:
            raise ValueError(f"Empty dataset at {path}")
        return dataset

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLM(model=self.llm_model_name, tensor_parallel_size=1,
                            max_model_len=262144, language_model_only=True)
        return self._llm

    def _free_llm(self):
        self._llm = None
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
    def _estimate_prompt_len(
        additional_information: dict[str, Any],
        model_name: str,
        _cache: dict[str, Any] = {},
    ) -> int:
        """Estimate prompt_token_ids placeholder length for the Talker stage.

        The AR Talker replaces all input embeddings via ``preprocess``, so the
        placeholder values are irrelevant but the **length** must match the
        embeddings that ``preprocess`` will produce.
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )

            if model_name not in _cache:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
                cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)

                # Load speech tokenizer (codec encoder) for exact ref_code_len.
                speech_tok = None
                try:
                    import os

                    from transformers.utils import cached_file

                    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

                    st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
                    if st_cfg_path:
                        speech_tok = Qwen3TTSTokenizer.from_pretrained(
                            os.path.dirname(st_cfg_path), torch_dtype=torch.bfloat16
                        )
                        logger.info("Loaded speech tokenizer for exact ref_code_len estimation")
                except Exception as e:
                    logger.debug("Could not load speech tokenizer: %s", e)

                _cache[model_name] = (tok, getattr(cfg, "talker_config", None), speech_tok)

            tok, tcfg, speech_tok = _cache[model_name]
            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

            def _estimate_ref_code_len(ref_audio: object) -> int | None:
                """Encode ref_audio with the actual codec to get exact frame count."""
                if not isinstance(ref_audio, (str, list)):
                    return None
                audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
                if not isinstance(audio_path, str) or not audio_path.strip():
                    return None
                try:
                    from urllib.parse import urlparse

                    import numpy as np

                    def _is_url(path: str) -> bool:
                        try:
                            parsed = urlparse(path)
                            if parsed.scheme in ("http", "https"):
                                return bool(parsed.netloc)
                            return parsed.scheme in ("file", "data")
                        except Exception:
                            return False

                    if _is_url(audio_path):
                        from vllm.multimodal.media import MediaConnector

                        connector = MediaConnector(allowed_local_media_path="/")
                        audio, sr = connector.fetch_audio(audio_path)
                    else:
                        from vllm.multimodal.media.audio import load_audio

                        audio, sr = load_audio(audio_path, sr=None, mono=True)

                    wav_np = np.asarray(audio, dtype=np.float32)

                    if speech_tok is not None:
                        enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                        ref_code = getattr(enc, "audio_codes", None)
                        if isinstance(ref_code, list):
                            ref_code = ref_code[0] if ref_code else None
                        if ref_code is not None and hasattr(ref_code, "shape"):
                            shape = ref_code.shape
                            return int(shape[0]) if len(shape) == 2 else int(shape[1]) if len(shape) == 3 else None

                    # Fallback: estimate from duration
                    codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
                    return int(len(audio) / sr * codec_hz)
                except Exception:
                    return None

            return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(tcfg, "codec_language_id", None),
                spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
                estimate_ref_code_len=_estimate_ref_code_len,
            )
        except Exception as exc:
            logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
            return 2048
        
    @staticmethod
    def _extract_json(text: str, required: tuple[str, ...]) -> dict | None:
        """Extract the last schema-valid JSON object from raw LLM output.

        Generation is constrained by REQUEST_SCHEMA so `text` is normally already a
        bare JSON object, but this stays robust to a leading <think> block, prose
        around the object, and the JSON example echoed from the prompt.
        """
        if "<think>" in text:
            if "</think>" not in text:
                return None  # generation truncated mid-reasoning
            text = text.rsplit("</think>", 1)[-1]

        decoder = json.JSONDecoder()
        best: dict | None = None
        i = 0
        while (start := text.find("{", i)) != -1:
            try:
                obj, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                i = start + 1
                continue
            if isinstance(obj, dict) and all(k in obj for k in required):
                best = obj  # keep the last well-formed match (the final answer)
            i = end
        return best

    @staticmethod
    def _save_wav(output_dir: str, request_id: str, mm: dict) -> str:
        """Concatenate audio chunks and write to a wav file; return its path."""
        audio_data = mm["audio"]
        sr_raw = mm["sr"]
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
        audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        out_wav = os.path.join(output_dir, f"{request_id}_tts.wav")
        sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
        logger.info(f"Request ID: {request_id}, Saved audio to {out_wav}")
        return out_wav

    def get_query(self, data):
        """Build Omni inputs.

        Args:
            data: list of ``(uid, target_text, style_instruct)`` tuples.

        Returns:
            list of Omni request dicts (``prompt_token_ids`` + ``additional_information``),
            parallel to ``data``.
        """
        inputs = []
        for _uid, text, instruct in data:
            additional_information = {
                "task_type": [TASK_TYPE],
                "text": [text],
                "language": ["English"],
                "instruct": [instruct],
                "max_new_tokens": [2048],
                "non_streaming_mode": [True],
            }
            inputs.append(
                {
                    "prompt_token_ids": [0]
                    * self._estimate_prompt_len(additional_information, self.tts_model_name),
                    "additional_information": additional_information,
                }
            )
        return inputs

    def build_inputs(self):
        """Build the ordered list of Omni requests from the perturbed dataset.

        Returns one record per pipeline that has an LLM answer to synthesize:
        ``{"annotation_id", "pipeline_key", "omni_input"}``.
        """
        # Code2Wav only captures CUDA graphs for power-of-two batch sizes.
        if self.batch_size < 1 or (self.batch_size & (self.batch_size - 1)) != 0:
            raise ValueError(
                f"--batch-size must be a power of two (got {self.batch_size}); "
                "non-power-of-two values do not align with CUDA graph capture sizes "
                "of Code2Wav."
            )

        dataset = self._load_dataset(self.output_dataset_path)

        meta = []      # (annotation_id, pipeline_key)
        triples = []   # (uid, target_text, style_instruct)
        for sample in dataset:
            anno_id = sample["annotation_id"]
            ideal_voice = sample["additional_info"]["ideal_voice_response"]
            for pipeline_key in ["pipeline1", "pipeline2"]:
                target_text = sample[pipeline_key]["llm"]["output"]
                if not target_text:
                    continue  # LLM stage not run for this pipeline -> nothing to say

                perturbation = sample[pipeline_key]["tts"]["perturbation"]
                info = perturbation.get("additional_info") or {}
                ptype = perturbation["type"]

                instruct = ideal_voice
                if isinstance(info, dict):
                    if ptype == "wrong_emotion":
                        instruct = info.get("steered_style_instruction") or ideal_voice
                    elif ptype in ("added_words", "missing_words"):
                        target_text = info.get("perturbed_text") or target_text

                meta.append((anno_id, pipeline_key))
                triples.append((f"{anno_id}_{pipeline_key}", target_text, instruct))

        return [
            {"annotation_id": anno_id, "pipeline_key": pipeline_key, "omni_input": omni_input}
            for (anno_id, pipeline_key), omni_input in zip(meta, self.get_query(triples))
        ]

    def generate_perturbations(self):
        """
        Perform perturbations before TTS synthesis. Two kind of perturbations are applied:
        1.  Style Perturbation: it performs adversarial prompt to cause a voice style/emotion
            that steers from the expectations.
        2.  Text Perturbation: it performs target text perturbation to force hallucinations/missing words in
            the target speech.
        """
        dataset = self._load_dataset(self.dataset_path)

        prompts = []
        sampling_params = []          # one entry per prompt (mixes both schemas)
        pending = []                  # (tts_stage_dict, required_keys)

        for sample in dataset:
            ideal_voice = sample["additional_info"]["ideal_voice_response"]
            for pipeline_key in ["pipeline1", "pipeline2"]:
                stage = sample[pipeline_key]
                target_text = stage["llm"]["output"]
                if not target_text:
                    continue  # LLM stage not run for this pipeline yet
                ptype = stage["tts"]["perturbation"]["type"]

                if ptype == "wrong_emotion":
                    system_prompt = STYLE_PERTURBATION_SYSTEM_PROMPT
                    user_prompt = STYLE_PERTURBATION_USER_PROMPT.format(
                        target_text=target_text,
                        #expected_voice_style=ideal_voice,
                    )
                    sampling_params.append(self.style_sampling_params)
                    pending.append((stage["tts"], REQUIRED_KEYS_STYLE))
                elif ptype in ("added_words", "missing_words"):
                    pert_specifics = ADD_WORDS_INSTR if ptype == "added_words" else OMITT_WORDS_INSTR
                    system_prompt = TEXT_PERTURBATION_SYSTEM_PROMPT
                    user_prompt = TEXT_PERTURBATION_USER_PROMPT.format(
                        target_text=target_text,
                        perturbation_instruction=pert_specifics,
                    )
                    sampling_params.append(self.text_sampling_params)
                    pending.append((stage["tts"], REQUIRED_KEYS_TEXT))
                else:
                    continue

                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompts.append(
                    self.processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                )

        if prompts:
            # vLLM accepts a per-prompt list of SamplingParams, so both schemas
            # run in a single batch.
            outputs = self.llm.generate(prompts, sampling_params)
            for (tts_stage, required_keys), o in zip(pending, outputs):
                json_data = self._extract_json(o.outputs[0].text, required_keys)
                if json_data:
                    tts_stage["evaluation"] = json_data.get("steering_impact")
                    tts_stage["perturbation"]["additional_info"] = json_data
        else:
            logger.info("No TTS perturbations to generate; passing dataset through.")

        with open(self.output_dataset_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def tts_synthesis(self, inputs):
        """Render each request to a wav file and record its path in the dataset."""
        dataset = self._load_dataset(self.output_dataset_path)
        by_id = {sample["annotation_id"]: sample for sample in dataset}

        batch_size = self.batch_size
        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]
            omni_batch = [record["omni_input"] for record in batch]

            for output in self.tts.generate(omni_batch):
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
    """Run the TTS perturbation + synthesis stage."""
    tts_stage = TTS_Stage(args)
    if args.perturb_input:
        tts_stage.generate_perturbations()  # loads the LLM lazily
        tts_stage._free_llm()               # release the GPU before Omni loads

    inputs = tts_stage.build_inputs()
    tts_stage.tts_synthesis(inputs)     # loads Omni lazily
    tts_stage._free_tts()


def parse_args():
    parser = TrackingArgumentParser(description="vLLM for offline inference with audio language models")
    
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
        "--config-path",
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
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--mode-tag",
        type=str,
        default="icl",
        choices=["icl", "xvec_only"],
        help="Mode tag for Base query x_vector_only_mode (default: icl).",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prompts per Omni batch; must be a power of two (default: 16).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config_path}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(config["data_dir"])
    args.input = data_dir / config["llm_output_file"]
    args.output = data_dir / config["tts_output_file"]
    args.output_dir = str(data_dir / "output_audio")
    args.llm_model = config["llm_model"]
    args.tts_model = config["tts_model"]
    args.perturb_input = config["perturb_input"]

    main(args)
