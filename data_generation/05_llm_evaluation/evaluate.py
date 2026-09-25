"""LLM-as-judge evaluation of each ASR -> LLM -> TTS pipeline in the dataset.

Reads a dataset that already has ASR hypotheses (`asr_stage.py`), LLM answers
(`llm/llm_stage.py`) and TTS output (`tts/tts_stage.py`), asks the judge model to
score each pipeline end to end, and writes the parsed verdict into
`sample[pipeline]["overall_evaluation"]`.

The judge is text-only: the TTS stage is assessed from the text it actually
vocalized (vs. the LLM answer) and the voice style the pipeline delivered (vs.
the expected one) -- no audio is loaded.

Each pipeline is judged ``--votes`` times at ``--temperature`` > 0. The verdict
written back is one sampled at random from the runs whose ``overall_score`` is
the modal score (majority voting). The full set of raw verdicts is dumped to
``--votes-output`` for `agreement.py` to compute judge self-consistency stats.
"""

import json
import os
import random
import statistics
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

PIPELINE_KEYS = ("pipeline1", "pipeline2")

PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt.prompt"
EVAL_PROMPT = PROMPT_PATH.read_text()

SYSTEM_PROMPT = (
    "You are a meticulous evaluator of spoken cascaded AI pipelines. "
    "Follow the rubric exactly and return only the requested JSON."
)

# Fallback map from a stage's planned perturbation type to an intended behavior,
# used only when the LLM stage did not record an explicit `behavior`.
_BEHAVIOR_FROM_PERTURBATION = {
    "hallucination": "hallucinate",
    "coherence": "recover",
}

REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "asr_assessment": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["none", "minor", "major"]},
                "meaning_preserved": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": ["severity", "meaning_preserved", "notes"],
            "additionalProperties": False,
        },
        "llm_assessment": {
            "type": "object",
            "properties": {
                "answer_coherent": {"type": "boolean"},
                "answer_hallucinated": {"type": "boolean"},
                "intent_matches_answer": {"type": "boolean"},
                "answer_serves_true_need": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": [
                "answer_coherent",
                "answer_hallucinated",
                "intent_matches_answer",
                "answer_serves_true_need",
                "notes",
            ],
            "additionalProperties": False,
        },
        "tts_assessment": {
            "type": "object",
            "properties": {
                "spoken_severity": {"type": "string", "enum": ["none", "minor", "major"]},
                "spoken_meaning_preserved": {"type": "boolean"},
                "voice_style_matches_expected": {"type": "boolean"},
                "voice_style_appropriate": {"type": "boolean"},
                "delivery_serves_true_need": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": [
                "spoken_severity",
                "spoken_meaning_preserved",
                "voice_style_matches_expected",
                "voice_style_appropriate",
                "delivery_serves_true_need",
                "notes",
            ],
            "additionalProperties": False,
        },
        "compounding": {
            "type": "object",
            "properties": {
                "primary_failure_stage": {
                    "type": "string",
                    "enum": ["asr", "llm", "tts", "multiple", "none"],
                },
                "notes": {"type": "string"},
            },
            "required": ["primary_failure_stage", "notes"],
            "additionalProperties": False,
        },
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_notes": {"type": "string"},
    },
    "required": [
        "asr_assessment",
        "llm_assessment",
        "tts_assessment",
        "compounding",
        "overall_score",
        "overall_notes",
    ],
    "additionalProperties": False,
}
_REQUIRED_KEYS = tuple(REQUEST_SCHEMA["required"])


def extract_json(text: str, required: tuple[str, ...] = _REQUIRED_KEYS) -> dict | None:
    """Extract the last schema-valid JSON object from raw LLM output.

    Generation is constrained by REQUEST_SCHEMA so `text` is normally already a
    bare JSON object, but this stays robust to a leading <think> block and prose
    around the object.
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


def _behavior_and_intent(stage: dict, asr_perturbed: bool) -> tuple[str, str]:
    """Recover the intended LLM behavior and the LLM's self-reported intent."""
    info = stage["llm"]["perturbation"].get("additional_info")
    if isinstance(info, dict):
        behavior = info.get("behavior")
        intent = info.get("intent") or "(not recorded)"
        if behavior:
            return behavior, intent
    else:
        intent = info if isinstance(info, str) and info else "(not recorded)"

    pert_type = stage["llm"]["perturbation"].get("type")
    behavior = _BEHAVIOR_FROM_PERTURBATION.get(
        pert_type, "propagate" if asr_perturbed else "helpful"
    )
    return behavior, intent


def _as_text(value) -> str:
    """Render an optional external-evaluation field for the prompt."""
    if value in (None, "", {}, []):
        return "None"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _tts_effective(sample: dict, pk: str) -> tuple[str, str]:
    """Text the TTS actually spoke and the voice style it delivered.

    Mirrors ``tts_stage.build_inputs``: a ``wrong_emotion`` perturbation swaps the
    style instruction, an ``added_words`` / ``missing_words`` perturbation swaps
    the target text; otherwise both pass through unchanged.
    """
    stage = sample[pk]
    answer = stage["llm"]["output"]
    expected_style = sample["additional_info"]["ideal_voice_response"]

    pert = stage["tts"]["perturbation"]
    ptype = pert.get("type")
    info = pert.get("additional_info")
    info = info if isinstance(info, dict) else {}

    spoken_text, delivered_style = answer, expected_style
    if ptype == "wrong_emotion":
        delivered_style = info.get("steered_style_instruction") or expected_style
    elif ptype in ("added_words", "missing_words"):
        spoken_text = info.get("perturbed_text") or answer
    return spoken_text, delivered_style


def _pairwise_rate(values: list) -> float:
    """P(two distinct runs picked the same value) -- Fleiss-style percent agreement."""
    n = len(values)
    if n < 2:
        return 1.0
    counts = Counter(values)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def majority_vote(verdicts: list[dict], rng: random.Random, temperature: float) -> dict:
    """Return one verdict whose ``overall_score`` is the modal score across ``verdicts``.

    Ties on the modal score are broken toward the score closest to the mean, then
    toward the lower score; one full verdict with that score is then sampled at
    random so the returned notes stay internally consistent with the score. A
    compact ``voting`` summary is attached under that key.
    """
    scores = [v["overall_score"] for v in verdicts]
    counts = Counter(scores)
    top = max(counts.values())
    mean_score = statistics.fmean(scores)
    modal = sorted(
        (s for s, c in counts.items() if c == top),
        key=lambda s: (abs(s - mean_score), s),
    )[0]

    picked = rng.choice([v for v in verdicts if v["overall_score"] == modal])
    return {
        **picked,
        "voting": {
            "n_votes": len(verdicts),
            "temperature": temperature,
            "overall_score": {
                "chosen": modal,
                "distribution": {str(s): counts[s] for s in sorted(counts)},
                "modal_rate": round(top / len(scores), 4),
                "pairwise_rate": round(_pairwise_rate(scores), 4),
                "mean": round(mean_score, 3),
                "std": round(statistics.pstdev(scores), 3),
            },
        },
    }


def build_prompt(processor, sample: dict, pk: str) -> str | None:
    stage = sample[pk]
    hypothesis = stage["asr"]["output"]
    answer = stage["llm"]["output"]
    tts_output = stage["tts"]["output"]
    if not hypothesis or not answer or not tts_output:
        return None  # ASR, LLM or TTS stage has not run for this pipeline

    asr_perturbed = stage["asr"]["perturbation"]["type"] is not None
    behavior, intent = _behavior_and_intent(stage, asr_perturbed)
    spoken_text, delivered_style = _tts_effective(sample, pk)

    user_msg = EVAL_PROMPT.format(
        reference_transcript=sample["input_request"]["text"],
        asr_hypothesis=hypothesis,
        llm_answer=answer,
        tts_spoken_text=spoken_text,
        #expected_voice_style=sample["additional_info"]["ideal_voice_response"],
        delivered_voice_style=delivered_style,
        asr_external_evaluation=_as_text(stage["asr"].get("evaluation")),
        llm_external_evaluation=_as_text(stage["llm"].get("evaluation")),
        tts_external_evaluation=_as_text(stage["tts"].get("evaluation")),
    )
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

def _is_valid(sample: dict) -> bool:
    for pk in PIPELINE_KEYS:
        asr = sample[pk]["asr"]["output"]
        llm = sample[pk]["llm"]["output"]
        tts = sample[pk]["tts"]["output"]

        if not asr or not llm or not tts:
            return False
    return True

def main(args) -> None:
    if args.votes > 1 and args.temperature == 0:
        raise SystemExit("--temperature must be > 0 when --votes > 1 (majority voting needs diversity)")
    rng = random.Random(args.seed)

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    dataset = [json.loads(line) for line in lines if line.strip()]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    processor = AutoProcessor.from_pretrained(args.llm_model)

    index: list[tuple[dict, str]] = []
    prompts: list[str] = []

    for sample in dataset:
        prompts_sample = []
        indices_sample = []
        if not _is_valid(sample):
            continue
        for pk in PIPELINE_KEYS:
            prompt = build_prompt(processor, sample, pk)
            if prompt is None :
                continue
            prompts_sample.append(prompt)
            indices_sample.append((sample, pk))
        prompts.extend(prompts_sample)
        index.extend(indices_sample)

    if not prompts:
        raise SystemExit(
            "no pipelines with an ASR hypothesis, an LLM answer and a TTS output to evaluate"
        )

    llm = LLM(
        model=args.llm_model,
        tensor_parallel_size=1,
        max_model_len=262144,  # native 262144 leaves no room for the FP8 weights on one 48 GB A6000
        language_model_only=True
    )

    sampling_params = SamplingParams(
        n=args.votes,
        temperature=1.0, 
        top_p=0.95, 
        top_k=20, 
        min_p=0.0, 
        presence_penalty=1.5, 
        repetition_penalty=1.0,
        max_tokens=2048,                # up from 2048: headroom so a wordy verdict isn't truncated → unparseable → lost vote
        structured_outputs=StructuredOutputsParams(json=REQUEST_SCHEMA),
    )


    outputs = llm.generate(prompts, sampling_params)

    n_ok = 0
    vote_records: list[dict] = []
    for (sample, pk), out in zip(index, outputs):
        verdicts = [v for v in (extract_json(c.text) for c in out.outputs) if v is not None]
        vote_records.append(
            {
                "annotation_id": sample["annotation_id"],
                "pipeline": pk,
                "n_requested": args.votes,
                "verdicts": verdicts,
            }
        )
        if not verdicts:
            continue
        sample[pk]["overall_evaluation"] = majority_vote(verdicts, rng, args.temperature)
        n_ok += 1

    n_valid = sum(len(r["verdicts"]) for r in vote_records)
    print(
        f"parsed {n_ok}/{len(index)} pipelines "
        f"({n_valid}/{len(index) * args.votes} valid votes)"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    votes_path = Path(args.votes_output)
    votes_path.parent.mkdir(parents=True, exist_ok=True)
    with votes_path.open("w", encoding="utf-8") as f:
        for rec in vote_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(vote_records)} vote records to {votes_path}  (feed to agreement.py)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--input", default=None, help="override the input (default: <data_dir>/<tts_output_file>)")
    parser.add_argument("--output", default=None, help="override the output (default: <data_dir>/<final_output_file>)")
    parser.add_argument(
        "--votes-output",
        default=None,
        help="raw per-pipeline verdict sets, consumed by agreement.py "
             "(default: <data_dir>/<votes_output_file>)",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    parser.add_argument("--votes", type=int, default=10, help="judge samples per pipeline for majority voting")
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="judge sampling temperature; must be > 0 when --votes > 1",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the tie-break sampling")
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

    # Command-line paths are relative to where you launch from, so resolve them before switching folder.
    overrides = {k: Path(getattr(args, k)).resolve() for k in ("input", "output", "votes_output") if getattr(args, k)}

    # Paths in the config and in the dataset are relative to the repo root.
    os.chdir(Path(args.config_path).resolve().parent)
    config = config["data_generation"]

    data_dir = Path(config["data_dir"])
    args.input = overrides.get("input", data_dir / config["tts_output_file"])
    args.output = overrides.get("output", data_dir / config["final_output_file"])
    args.votes_output = overrides.get("votes_output", data_dir / config["votes_output_file"])
    args.llm_model = config["llm_model"]

    main(args)
