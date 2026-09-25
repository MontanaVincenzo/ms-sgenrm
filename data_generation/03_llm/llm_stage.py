import json
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

PIPELINE_KEYS = ("pipeline1", "pipeline2")

PROMPTS_DIR = Path(__file__).parent / "prompts"
LLM_PROMPT = (PROMPTS_DIR / "llm_prompt.prompt").read_text()  # has {action_policy} and {transcription}

SYSTEM_PROMPT = (
    "You are the LLM component of a cascaded ASR -> LLM -> TTS pipeline. "
    "You are generating synthetic data to train a pipeline-error detector; "
    "for this sample follow the behavior instruction exactly, even when it asks "
    "you to answer imperfectly."
)

# behavior key -> text injected into the prompt's {action_policy} slot
BEHAVIORS = {
    "helpful": (
        "Be helpful and answer the user's question accurately and directly."
    ),
    "incoherence": (
        "Take the transcription at face value and answer exactly what it says, even "
        "if the request seems odd or unlikely and steer the answer to degrade coherence"
        "with respect to the user need. Do not correct, second-guess, or point "
        "out possible mis-recognitions. Never ask the user to repeat or clarify. "
        
    ),
    "no answer": (
        "Answer partially or do not answer at all: produce a random answer (e.g., '...') or stop before completing your sentence."
    ),
    "clarification": (
        "Ask for clarifications, even though the request seems clear."
    ),
}

EXPLANATION_BY_ACTION = {
    "helpful": "Explain in one sentence how your answer correctly addresses what the user asked for.",
    "incoherence": "Explain in one sentence how your answer diverges from the user's likely intent (e.g. takes a probable misrecognition at face value).",
    "no answer": "Explain in one sentence how your answer is incomplete, truncated, or non-responsive.",
    "clarification": "Explain in one sentence why the clarification you asked for is unnecessary given the request.",
}


REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "llm_answer": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["llm_answer", "explanation"],
    "additionalProperties": False,
}
_REQUIRED_KEYS = tuple(REQUEST_SCHEMA["required"])


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


def decide_action(stage: dict) -> str:
    """Map a pipeline stage's planned perturbation to a BEHAVIORS key.
    """
    llm_pert = stage["llm"]["perturbation"]["type"]
    if llm_pert:
        return llm_pert
    return "helpful"


def build_prompt(processor, transcript: str, action: str) -> str:
    user_msg = LLM_PROMPT.format(action_policy=BEHAVIORS[action], transcription=transcript, explanation_description=EXPLANATION_BY_ACTION[action])
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def main(args) -> None:
    rng = random.Random(args.seed)

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    dataset = [json.loads(line) for line in lines if line.strip()]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    processor = AutoProcessor.from_pretrained(args.llm_model)

    # Build every prompt first so a malformed sample fails before the model loads.
    index: list[tuple[dict, str, str]] = []
    prompts: list[str] = []
    for sample in dataset:
        for pk in PIPELINE_KEYS:
            stage = sample[pk]
            transcript = stage["asr"]["output"]
            if not transcript:
                continue  # ASR stage not run for this pipeline yet
            action = decide_action(stage)
            prompts.append(build_prompt(processor, transcript, action))
            index.append((sample, pk, action))

    llm = LLM(model=args.llm_model,
            tensor_parallel_size=1,
            max_model_len=262144, 
            language_model_only=True, 
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



    outputs = llm.generate(prompts, sampling_params)

    n_ok = 0
    for (sample, pk, action), out in zip(index, outputs):
        data = extract_json(out.outputs[0].text)
        if not data:
            continue
        llm_stage = sample[pk]["llm"]
        llm_stage["output"] = data["llm_answer"]
        llm_stage["perturbation"]["additional_info"] = {
            "behavior": action
        }
        llm_stage["evaluation"] = data["explanation"]
        n_ok += 1
    print(f"parsed {n_ok}/{len(index)} generations")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    parser.add_argument("--seed", type=int, default=0)
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

    # Paths in the config and in the dataset are relative to the repo root.
    os.chdir(Path(args.config_path).resolve().parent)
    config = config["data_generation"]

    data_dir = Path(config["data_dir"])
    args.input = data_dir / config["asr_output_file"]
    args.output = data_dir / config["llm_output_file"]
    args.llm_model = config["llm_model"]

    main(args)
