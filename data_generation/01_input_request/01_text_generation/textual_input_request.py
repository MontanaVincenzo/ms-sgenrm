import numpy as np
import json
import random

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import copy
from argparse import ArgumentParser
from pathlib import Path
import sys
import yaml

with open('templates/topic.prompt', 'r') as f:
    PROMPT_TOPIC = f.read()
with open('templates/output_format.prompt', 'r') as f:
    PROMPT_OUTPUT_FORMAT = f.read()

REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "request": {"type": "string"},
        "suitability": {"type": "string"},
        "ideal_voice": {"type": "string"},
    },
    "required": ["request", "suitability", "ideal_voice"],
    "additionalProperties": False,
}
_REQUIRED_KEYS = tuple(REQUEST_SCHEMA["required"])

SAMPLE_TEMPLATE = {
    "annotation_id": None,
    "additional_info": {
        "topic": None,
        "description": None,
        "ideal_voice_response": None
    },
    "input_request": {
        "text": None,
        "audio_path": None,
    },
    "pipeline1": {
        "asr": {
            "perturbation" : {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None
        },
        "llm": {
            "perturbation": {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None
        },
        "tts": {
            "perturbation": {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None,
        },
        "overall_evaluation" : None
    },
    "pipeline2": {
        "asr": {
            "perturbation" : {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None
        },
        "llm": {
            "perturbation" : {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None
        },
        "tts": {
            "perturbation" : {
                "type": None,
                "additional_info": None
            },
            "output": None,
            "evaluation": None,
        },
        "overall_evaluation" : None
    },
    "comparison": None
}

PERTURBATIONS_ASR = [None, "silence_masking", "hallucination"]
PERTURBATIONS_LLM = [None, "incoherence", "no answer", "clarification"]
PERTURBATIONS_TTS = [None, "wrong_emotion", "missing_words", "added_words"]



def extract_json(text: str, required: tuple[str, ...] = _REQUIRED_KEYS) -> dict | None:
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




def main(args) -> None:
    # Qwen3.5-MoE is a multimodal checkpoint; language_model_only skips the
    # vision tower (equivalent of `vllm serve --language-model-only`).
    # ~35B params in FP8 (~35 GB) fit on a single 48 GB A6000 once the context
    # length is capped (the checkpoint's native 262144 leaves no room for
    # weights).
    llm = LLM(
        model=args.llm_model,
        tensor_parallel_size=1,
        max_model_len=262144,  # native 262144 leaves no room for the FP8 weights on one 48 GB A6000
        language_model_only=True
    )

    sampling_params = SamplingParams(
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

    processor = AutoProcessor.from_pretrained(args.llm_model)

    # Prepare a list of prompts in batch
    

    with open(args.data_dir / "topics/topics.jsonl", "r", encoding="utf-8") as f:
        topics_data = [json.loads(line) for line in f]

    prompts = []
    chosen_topics = []  # kept in lockstep with prompts / outputs

    for _ in range(args.num_samples):
        # random.choice picks one item; unpack the inner "topic" key
        item = random.choice(topics_data)
        topic = item.get("topic", item)  # Handles both nested and flat JSONL shapes

        prompt_topic = PROMPT_TOPIC.format(topic=topic.get("name"), example_question=topic.get("example_prompt"), suitability_rationale=topic.get("voice_suitability"))
        conversation = [
            {"role": "system", "content": "You are a user who is talking with an AI assistant."},
            {"role": "user", "content": f"Generate a request for the assistant. {prompt_topic}. {PROMPT_OUTPUT_FORMAT}"},
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        prompts.append(prompt)
        chosen_topics.append(topic.get("name"))


    # Run offline batch inference
    outputs = llm.generate(prompts, sampling_params)

    # Extract JSON objects and append to .jsonl
    output_filename = args.data_dir / args.input_request_output_file
    successful_count = 0

    with open(output_filename, "w", encoding="utf-8") as f:
        for i, (topic_name, o) in enumerate(zip(chosen_topics, outputs)):
            sample = copy.deepcopy(SAMPLE_TEMPLATE)
            generated_text = o.outputs[0].text
            json_data = extract_json(generated_text)

            asr_pert1, asr_pert2 = np.random.choice(PERTURBATIONS_ASR, size=2, replace=False) if args.perturb_input else (None, None)
            llm_pert1, llm_pert2 = np.random.choice(PERTURBATIONS_LLM, size=2, replace=False, p=[0.5, 0.3, 0.1, 0.1]) if args.perturb_input else (None, None)
            tts_pert1, tts_pert2 = np.random.choice(PERTURBATIONS_TTS, size=2, replace=False, p=[0.4, 0.2, 0.2, 0.2]) if args.perturb_input else (None, None)

            if json_data:
                # Prepend the source topic, then the model's fields
                sample["additional_info"]["topic"] = topic_name
                sample["additional_info"]["description"] = json_data["suitability"]
                sample["additional_info"]["ideal_voice_response"] = json_data["ideal_voice"]
                sample["input_request"]["text"] = json_data["request"]
                sample["annotation_id"] = i
                sample["pipeline1"]["asr"]["perturbation"]["type"] = asr_pert1
                sample["pipeline2"]["asr"]["perturbation"]["type"] = asr_pert2
                sample["pipeline1"]["llm"]["perturbation"]["type"] = llm_pert1
                sample["pipeline2"]["llm"]["perturbation"]["type"] = llm_pert2
                sample["pipeline1"]["tts"]["perturbation"]["type"] = tts_pert1
                sample["pipeline2"]["tts"]["perturbation"]["type"] = tts_pert2
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                successful_count += 1
            else:
                print(f"⚠️ Failed to parse JSON from generation:\n{generated_text}\n---")

    print(f"\nSaved {successful_count}/{args.num_samples} records to {output_filename}")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--config-path", required=True)
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

    args.data_dir = Path(config["data_dir"])
    args.input_request_output_file = config["input_request_output_file"]
    args.llm_model = config["llm_model"]
    args.perturb_input = config["perturb_input"]

    main(args)
