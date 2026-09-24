"""TTS perturbation + synthesis stage for the ASR -> LLM -> TTS pipeline, using
OmniVoice (k2-fsa/OmniVoice) as the TTS backend instead of Qwen3-TTS VoiceDesign.

``main`` runs two steps:
1. ``generate_perturbations()`` -- for pipelines whose planned TTS perturbation
   is ``wrong_emotion`` / ``added_words`` / ``missing_words``, ask the LLM for a
   steered voice-style instruction or a perturbed target text. Each perturbation
   family has its own JSON schema and therefore its own ``SamplingParams``; the
   two are mixed in one ``LLM.generate`` call via a per-prompt list.
2. ``tts_synthesis()`` -- render every pipeline's (possibly perturbed) text with
   OmniVoice (single float32 diffusion stage: text -> iterative unmasking ->
   DAC decode -> 24kHz audio) and write the wav path back into the dataset.
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

logger = logging.getLogger(__name__)


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
        self.seed = args.seed
        self._tts = None
        self._llm = None

        # vllm-omni entrypoint kwargs: forward the CLI namespace like the upstream
        # OmniVoice end2end example, minus this script's own options, and pin the model.
        self.omni_kwargs = vars(args).copy()
        for key in ("input", "output", "limit", "pipeline_config", "llm_model", "tts_model", "seed"):
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

    def get_query(self, data):
        """Build Omni inputs.

        Args:
            data: list of ``(uid, target_text, style_instruct)`` tuples.

        Returns:
            list of OmniVoice prompt dicts (``prompt`` + ``mm_processor_kwargs``),
            parallel to ``data``.
        """
        inputs = []
        for _uid, text, instruct in data:
            inputs.append(
                {
                    "prompt": text,
                    "mm_processor_kwargs": {
                        "lang": "en",
                        "instruct": instruct,
                    },
                }
            )
        return inputs

    def build_inputs(self):
        """Build the ordered list of Omni requests from the perturbed dataset.

        Returns one record per pipeline that has an LLM answer to synthesize:
        ``{"annotation_id", "pipeline_key", "omni_input"}``.
        """
        if self.batch_size < 1:
            raise ValueError(f"--batch-size must be >= 1 (got {self.batch_size})")

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

        # OmniVoice runs as a single diffusion stage, so sampling_params_list
        # has exactly one entry (applied to every request in the batch).
        sampling_params_list = [OmniDiffusionSamplingParams(extra_args={"seed": self.seed})]

        batch_size = self.batch_size
        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]
            omni_batch = [record["omni_input"] for record in batch]

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
    """Run the TTS perturbation + synthesis stage."""
    tts_stage = TTS_Stage(args)

    tts_stage.generate_perturbations()  # loads the LLM lazily
    tts_stage._free_llm()               # release the GPU before Omni loads

    inputs = tts_stage.build_inputs()
    tts_stage.tts_synthesis(inputs)     # loads Omni lazily
    tts_stage._free_tts()


def parse_args():
    parser = TrackingArgumentParser(description="vLLM-Omni OmniVoice TTS stage for offline inference")

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
        default="/home/vmontana/synthetic_data_generation/vllm-omni/vllm_omni/deploy/omnivoice.yaml",
        help="Path to the OmniVoice deploy config YAML (single diffusion stage).",
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
        help="Seed for OmniVoice generation (default: random).",
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

    data_dir = Path(config["data_dir"])
    args.input = data_dir / config["llm_output_file"]
    args.output = data_dir / config["tts_output_file"]
    args.output_dir = str(data_dir / "output_audio")
    args.llm_model = config["llm_model"]
    args.tts_model = config["tts_model"]

    main(args)
