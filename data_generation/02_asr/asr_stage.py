from nemo.collections.asr.models import ASRModel
import json
import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import random
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from argparse import ArgumentParser
import gc, torch
import yaml
import sys
with open(Path(__file__).resolve().parent / "prompts" / "asr_perturbation.prompt", 'r') as f:
    ASR_PERTURBATION_PROMPT = f.read()

MODIF_LEVEL = [
    ("slightly", "Make a subtle shift in nuance or detail without reversing the core intent. Preference for sound-alikes: When choosing replacement words, strongly prefer homophones, near-homophones, or phonetically similar words (e.g., mishearings like \"flour\" vs. \"flower\", \"affect\" vs. \"effect\", \"later\" vs. \"latter\", \"accept\" vs. \"except\")."), 
    ("heavily", "Fundamentally invert or redirect the core intent, objective, or topic (e.g., changing an affirmation to a negation, switching the target action)."
)]

REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "perturbed_transcription": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["perturbed_transcription", "explanation"],
    "additionalProperties": False,
}
_REQUIRED_KEYS = tuple(REQUEST_SCHEMA["required"])



class ASR_Stage:
    def __init__(self, input_file, output_file, asr_model_name="nvidia/canary-1b-v2", llm_model_name="Qwen/Qwen3.6-35B-A3B-FP8", input_perturbed=False):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.asr_model_name = asr_model_name
        self.llm_model_name = llm_model_name
        self._asr_model = None
        self._llm = None
        self.input_perturbed = input_perturbed

        self.sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.7, 
                top_p=0.80, 
                top_k=20, 
                min_p=0.0, 
                presence_penalty=1.5, 
                repetition_penalty=1.0,
                # Force the output to be a valid instance of REQUEST_SCHEMA -- no regex
                # salvage needed. (Constrains all tokens, so no <think> block is emitted.)
                structured_outputs=StructuredOutputsParams(json=REQUEST_SCHEMA),
            )
        
        self.processor = AutoProcessor.from_pretrained(self.llm_model_name)


    @property
    def asr_model(self):
        if self._asr_model is None:
            self._asr_model = ASRModel.from_pretrained(model_name=self.asr_model_name).eval()
        return self._asr_model

    def _free_asr(self):
        self._asr_model = None
        gc.collect(); torch.cuda.empty_cache()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLM(
                    model=self.llm_model_name,
                    tensor_parallel_size=1,
                    max_model_len=262144,  # native 262144 leaves no room for the FP8 weights on one 48 GB A6000
                    language_model_only=True
                )
        return self._llm

    @staticmethod
    def _mask_random_silence(input_path, output_path, min_duration_sec=0.5, max_duration_sec=2.0):
        sample_rate, data = wav.read(input_path)
        
        # Total audio duration in seconds
        total_duration = len(data) / sample_rate
        
        if total_duration == 0:
            raise ValueError("Audio file is empty.")

        # 1. Ensure mask duration doesn't exceed the total audio length
        max_dur = min(max_duration_sec, total_duration)
        min_dur = min(min_duration_sec, max_dur)
        
        # 2. Pick a random duration for the mask
        mask_duration = random.uniform(min_dur, max_dur)
        
        # 3. Pick a random start time safely within [0, total_duration - mask_duration]
        max_start_sec = max(0.0, total_duration - mask_duration)
        start_sec = random.uniform(0.0, max_start_sec)
        end_sec = start_sec + mask_duration

        # 4. Convert timestamps to sample indices and clamp strictly within data bounds
        start_sample = max(0, min(int(start_sec * sample_rate), len(data)))
        end_sample = max(0, min(int(end_sec * sample_rate), len(data)))

        # Zero out the selected range
        masked_data = data.copy()
        masked_data[start_sample:end_sample] = 0

        wav.write(output_path, sample_rate, masked_data)

        # Return start and end times (rounded to 3 decimal places for clean logs)
        return round(start_sec, 3), round(end_sec, 3)

    @staticmethod
    def _extract_json(text: str, required: tuple[str, ...] = _REQUIRED_KEYS) -> dict | None:
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


    def silence_masking(self):
        with open(self.input_file, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]
    
        data_dir = self.input_file.resolve().parent
        pertubed_audios_dir = data_dir / "input_audios_perturbed"
        pertubed_audios_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset:
            for pipeline_key in ["pipeline1", "pipeline2"]:
                if sample[pipeline_key]["asr"]["perturbation"]["type"] == "silence_masking":
    
                    original_audio_name = Path(sample["input_request"]["audio_path"]).name
                    pertubed_audio_path = pertubed_audios_dir / original_audio_name
    
                    start_time, end_time = self._mask_random_silence(
                        input_path=sample["input_request"]["audio_path"],
                        output_path=pertubed_audio_path,
                        min_duration_sec=0.8,  # Minimum second mask
                        max_duration_sec=1.5   # Maximum second mask
                    )
                    sample[pipeline_key]["asr"]["perturbation"]["additional_info"] = {
                        "start": start_time, 
                        "end": end_time
                    }
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def transcribe(self):

        in_file = self.output_file if self.input_perturbed else self.input_file

        with open(in_file, encoding="utf-8") as f: # Use output_file on purpose
            dataset = [json.loads(line) for line in f]

        clean_dir = self.input_file.resolve().parent / "input_audios_original"
        perturbed_dir = self.input_file.resolve().parent / "input_audios_perturbed"
        batch_size = 32
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            samples = []
            for sample in batch:
                audio_name = os.path.basename(sample["input_request"]["audio_path"])
                for pipeline_key in ["pipeline1", "pipeline2"]:
                    if sample[pipeline_key]["asr"]["perturbation"]["type"] == "silence_masking":
                        samples.append(os.path.join(perturbed_dir, audio_name))
                    else:
                        samples.append(os.path.join(clean_dir, audio_name))

            transcripts = self.asr_model.transcribe(samples, source_lang='en', target_lang='en') if self.asr_model_name == "nvidia/canary-1b-v2" else self.asr_model.transcribe(samples)
            for j in range(len(batch)):
                dataset[i + j]["pipeline1"]["asr"]["output"] = transcripts[2*j].text
                dataset[i + j]["pipeline2"]["asr"]["output"] = transcripts[2*j + 1].text


        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate_hallucinations(self):
        # Prepare a list of prompts in batch
        prompts = []
    
        with open(self.output_file, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
    
        samples_with_target_perturbation = []
        for sample in dataset:
            for pipeline_key in ["pipeline1", "pipeline2"]:
                if sample[pipeline_key]["asr"]["perturbation"]["type"] == "hallucination":
                    transcript = sample[pipeline_key]["asr"]["output"]
    
                    degree, m_rule = random.choice(MODIF_LEVEL)
    
                    perturbation_prompt = ASR_PERTURBATION_PROMPT.format(degree=degree, modification_rule=m_rule, transcription=transcript)
                    conversation = [
                        {"role": "system", "content": "You are an expert in linguistic."},
                        {"role": "user", "content": f"{perturbation_prompt}"},
                    ]
    
                    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) 
                    samples_with_target_perturbation.append((sample, pipeline_key))
                    prompts.append(prompt)
    
    
        # Run offline batch inference
        outputs = self.llm.generate(prompts, self.sampling_params)
    
        
        for i, ((sample, pipeline_key), o) in enumerate(zip(samples_with_target_perturbation, outputs)):
            generated_text = o.outputs[0].text
            json_data = self._extract_json(generated_text)
    
            if json_data:
                sample[pipeline_key]["asr"]["output"] = json_data["perturbed_transcription"]
                sample[pipeline_key]["asr"]["evaluation"] = json_data["explanation"]
            else:
                print(f"[hallucination] parse failed pk={pipeline_key} "
                f"finish={o.outputs[0].finish_reason!r} text={generated_text[:200]!r}")
    
        # Extract JSON objects and append to .jsonl
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--config_path", required=True)

    args = parser.parse_args()


    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config_path}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    # Paths in the config and in the dataset (e.g. "data/input_audios_original/x.wav") are relative to the repo root.
    os.chdir(Path(args.config_path).resolve().parent)
    config = config["data_generation"]

    data_dir = Path(config["data_dir"])
    input_file = data_dir / config["input_request_output_file"]
    output_file = data_dir / config["asr_output_file"]

    asr_stage = ASR_Stage(
        input_file,
        output_file,
        asr_model_name=config["asr_model"],
        llm_model_name=config["llm_model"],
        input_perturbed=config["perturb_input"]
    )
    if config["perturb_input"]:
        asr_stage.silence_masking() # loads Canary lazily
    asr_stage.transcribe()
    asr_stage._free_asr() # release GPU 

    if config["perturb_input"]:
        asr_stage.generate_hallucinations()   # loads vLLM lazily
