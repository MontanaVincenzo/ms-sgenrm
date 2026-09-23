"""Out-of-distribution eval: trained model vs. speakbench-v1-labelled-508.

https://huggingface.co/datasets/potsawee/speakbench-v1-labelled-508

Unlike `eval.py` (in-distribution ASR->LLM->TTS pipeline eval), speakbench has
no separate ASR transcription / LLM answer text for its candidate responses --
just a spoken instruction (+ its ground-truth text) and two response audios
(`audio_a`, `audio_b`) with a human preference label (`a`, `b`, `tie`). To
reuse the trained model's fixed 4-slot template (request audio, ASR text, LLM
answer text, TTS audio) we map:

    input_audio  = instruction audio
    asr text     = instruction_text (treated as a perfect transcription)
    llm answer   = "" (not available for this dataset -- left blank)
    tts audio    = audio_a / audio_b

Each prompt is sampled `--n_votes` times (default 10) at `--vote_temperature`
(default 0.7, since temperature=0.0 would make every trial identical) and the
final a/b/tie decision is a strict-plurality majority vote across trials (a
genuine tie in the vote itself, e.g. 4/4/2, is recorded as "tie"). That
decision is checked against the dataset's `label`.

Dataset audio is downloaded once via `datasets.load_dataset` (cached under
`~/.cache/huggingface` like any other HF dataset) and then decoded to local
wav files under `--audio_cache_dir` on first use; later runs reuse those wavs.

`--prompt_style` picks how the model is prompted:
  - "training": the exact ASR->LLM->TTS system prompt from `sft.py`, with the
    (unavailable) LLM-answer slot left blank. Isolates the OOD-data variable
    but its "TTS vs. the LLM's text" instruction is meaningless with no text.
    Only supports `--approach pointwise` (the template scores one candidate
    response at a time).
  - "task_specific" (default): a prompt written for this task -- judge a
    spoken assistant response against the instruction it answers, no
    ASR/LLM/TTS staging language. Never seen during SFT, so JSON-format
    adherence is itself part of what's being tested.

`--approach` picks how candidates are judged:
  - "pointwise" (default): each candidate is scored on its own (one prompt
    per candidate). Per vote trial, the candidates' `overall_score`s are
    compared (`--tie_tol`) to get that trial's a/b/tie preference; the
    majority vote across trials is the row's decision.
  - "pairwise": both candidate responses are shown to the model together in
    a single prompt, which directly outputs an a/b/tie `decision` per trial;
    the majority vote across trials is the row's decision.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwen_omni_utils import process_mm_info
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from eval import AUDIO_SAMPLE_RATE, BASE_MODEL, _sanitize, build_llm, coerce_score, extract_json
from utils.create_templates import LEVELS
from utils.create_templates import build_conversation as build_training_conversation

DATASET_NAME = "potsawee/speakbench-v1-labelled-508"
CANDIDATES = ("a", "b")
PROMPT_STYLES = ("training", "task_specific")
APPROACHES = ("pointwise", "pairwise")

TASK_SYSTEM_PROMPT_BODY = {
    "pointwise": (
        "You are an evaluator of audio outputs produced by different audio-capable large language models. You are given the user's spoken "
        "instruction and the assistant's full spoken audio response to that instruction.\n\n"

        "Evaluate based on these criteria:\n"
        "1. Semantics: Does the content fulfill the user’s request accurately?\n"
        "2. Paralinguistics: How well does the speech match requested tone, emotion, style, pacing, and expressiveness?\n"
        "Important: Do not favor verbalized descriptions of tone over actual tonal expression. "
        "For example, a response that says \"I am speaking excitedly\" but sounds flat should be scored low.\n\n"

        "Provide four kind of outputs:\n"
        "Transcription: Trascribe here what the Assistant's spoken response says.\n"
        "Paralinguistics Features: Write here paralinguistic characteristics you extract from audio (tone, emotion, style, pacing, and expressiveness).\n"
        "Reasoning: Based on Transcription and the Paralinguistic Features, write here your evaluation of the response.\n"
        "Overall Score: Conditioned by what you wrote in content, write here a score in the discrete range [1-5], where 1 indicates complete failure, while 5 expresses complete fulfillment of the request. \n\n"
    ),
    "pairwise": """
        You are an evaluator of audio outputs produced by different audio-capable large language models. 
        Your task is to compare two audio responses (Audio A and Audio B) generated according to a user’s instruction.

        Evaluate based on these criteria: 
        1. Semantics: Does the content fulfill the user’s request accurately? 
        2. Paralinguistics: How well does the speech match requested tone, emotion, style, pacing, and expressiveness?

        Important: Do not favor verbalized descriptions of tone over actual tonal expression. 
        A response that says "I am speaking excitedly" but sounds flat should rank lower than one that genuinely sounds excited.

        Follow this process: 
        1. Analyze the key characteristics requested in the user’s instruction 
        2. Evaluate how well Audio A performs on these characteristics 
        3. Evaluate how well Audio B performs on these characteristics 
        4. Compare their strengths and weaknesses 
        5. Decide which is better overall
        
        """
            ,
    "pairwise_old": (
        "You are an evaluator of a spoken virtual assistant. You are given the user's spoken "
        "instruction and two assistants' full spoken audio responses to that instruction (Response A vs Response B).\n\n"
        "Evaluation Rules:\n"
        "Content: Compare the responses and assess whether each correctly and completely fulfills what the instruction asked for.\n"
        "Decision: Express a preference bewteen the two. You must call a tie when you are uncertain about the decision or the responses result in the same quality.\n\n"
    )

}
TASK_USER_INSTRUCTION_POINTWISE = "Evaluate this spoken response to the instruction and return the evaluation JSON."
TASK_USER_INSTRUCTION_PAIRWISE = "Evaluate the two spoken responses to the instruction and return the evaluation JSON."

_TASK_LEVEL_INSTRUCTION = {
    "pointwise": {
        "A": (
            "Return ONLY a JSON object with key: overall_score.\n"
            'Output format example: {"transcription": "...", "overall_score": 4}'
        ),
        "B": (
            "Return ONLY a JSON object with keys: content_assessment, overall_score.\n"
            "Output format example: "
            "{"
            "\"transcription\": \"I'm talking about the weather's of today in a low pitch\","
            "\"paralinguistis_features\": \"Monotone High Pitch, unnatural voice.\","
            "\"content_assessment\": \"Misleading answer without actual content.\","
            "\"overall_score\": 3"
            "}"
        ),
        "C": (
            "Return ONLY a valid JSON object containing the keys: "
            "\"transcription\", \"paralinguistic_features\", \"reasoning\", and \"overall_score\". "
            "Do not include Markdown wrapping (like ```json) or any conversational text.\n\n"
            "Output format example:\n"
            "{\n"
            "\"transcription\": \"I'm talking about the weather's of today in a low pitch\","
            "\"paralinguistis_features\": \"Monotone High Pitch, unnatural voice.\","
            "\"reasoning\": \"Misleading answer without actual content and wrong pitch.\","
            "\"overall_score\": 1"
            "}"
        ),
    },
    "pairwise": {
        "A": (
            "Return ONLY a JSON object with key: decision. The value must be 'A', 'B', or 'tie'.\n"
            'Output format example: {"decision": "A"}'
        ),
        "B": (
            "Return ONLY a JSON object with keys: content, decision. 'content' is a short textual "
            "comparison of the two responses; 'decision' is the preference ('A', 'B', or 'tie').\n"
            'Output format example: {"content": "...", "decision": "A"}'
        ),
        "C_old": (
            "Return ONLY a JSON object with keys: content, decision. 'content' is a short textual "
            "comparison of the two responses; 'decision' is the preference ('A', 'B', or 'tie')."
            "Please use \"tie\" sparingly, and only when you absolutely cannot choose the winner.\n"
            'Output format example: {"content": "...", "decision": "A"}'
        ),
        "C": (
            "Avoid position bias and don’t let response length influence your evaluation. After your analysis, output valid JSON with exactly two keys:"
            "’reasoning’ (your explanation of the comparison) and ’label’ (a string value: ’A’ if the first audio is better, ’B’ if the second audio is better, or"
            "’tie’ if they are equally good/bad or when you cannot choose the winner.)"
        ),
    },
}


def task_system_prompt(approach: str, level: str) -> str:
    return TASK_SYSTEM_PROMPT_BODY[approach] + _TASK_LEVEL_INSTRUCTION[approach][level]


def build_task_conversation_pointwise(instruction_audio_path, response_audio_path, level: str) -> list[dict]:
    user_content = [
        {"type": "text", "text": TASK_USER_INSTRUCTION_POINTWISE},
        {"type": "text", "text": "User's spoken instruction:"},
        {"type": "audio", "audio": str(instruction_audio_path)},
        {"type": "text", "text": "Assistant's spoken response:"},
        {"type": "audio", "audio": str(response_audio_path)},
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": task_system_prompt("pointwise", level)}]},
        {"role": "user", "content": user_content},
    ]


def build_task_conversation_pairwise(instruction_audio_path, response_audio_path1, response_audio_path2, level: str) -> list[dict]:
    user_content = [
        {"type": "text", "text": TASK_USER_INSTRUCTION_PAIRWISE},
        {"type": "text", "text": "User's spoken instruction:"},
        {"type": "audio", "audio": str(instruction_audio_path)},
        {"type": "text", "text": "Assistant's spoken response A:"},
        {"type": "audio", "audio": str(response_audio_path1)},
        {"type": "text", "text": "Assistant's spoken response B:"},
        {"type": "audio", "audio": str(response_audio_path2)},
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": task_system_prompt("pairwise", level)}]},
        {"role": "user", "content": user_content},
    ]


def render_request(processor, conversation: list[dict]) -> dict:
    """Render a chat-message conversation into a vLLM request dict."""
    conversation = _sanitize(conversation)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, _images, _videos = process_mm_info([conversation], use_audio_in_video=False)
    request = {"prompt": prompt}
    if audios:
        request["multi_modal_data"] = {"audio": [(a, AUDIO_SAMPLE_RATE) for a in audios]}
    return request


def build_candidate_request(processor, rec: dict, response_audio_path: str, prompt_style: str, level: str) -> dict:
    if prompt_style == "training":
        conversation = build_training_conversation(
            rec["instruction_audio_path"], rec["instruction_text"], "", response_audio_path,
            target=None, level=level,
        )
    else:
        conversation = build_task_conversation_pointwise(
            rec["instruction_audio_path"], response_audio_path, level=level,
        )
    return render_request(processor, conversation)


def build_pairwise_request(processor, rec: dict, level: str) -> dict:
    conversation = build_task_conversation_pairwise(
        rec["instruction_audio_path"], rec["audio_a_path"], rec["audio_b_path"], level=level,
    )
    return render_request(processor, conversation)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default="train")
    p.add_argument("--audio_cache_dir", default="data/speakbench_audio",
                   help="local wav cache; existing files are reused, missing ones are decoded from the HF dataset")
    p.add_argument("--output", default=None,
                   help="default: results/speakbench/<prompt_style>/<approach>/<level>/trained_model.jsonl")
    p.add_argument("--stats_output", default=None,
                   help="path for the statistics JSON (default: <output dir>/evaluation.json)")
    p.add_argument("--prompt_style", choices=PROMPT_STYLES, default="task_specific",
                   help="'training' reuses sft.py's ASR->LLM->TTS prompt verbatim; 'task_specific' (default) "
                        "is written for this instruction-vs-response task")
    p.add_argument("--approach", choices=APPROACHES, default="pointwise",
                   help="'pointwise' (default) scores each candidate separately and compares scores; "
                        "'pairwise' shows both candidates in one generation and reads off its a/b/tie decision. "
                        "'pairwise' requires --prompt_style task_specific")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir; pass --no_lora to evaluate the base model")
    p.add_argument("--no_lora", action="store_true", help="run the base model without any adapter")
    p.add_argument("--level", choices=LEVELS, default="C",
                   help="output-JSON verbosity for --prompt_style task_specific; must match the grounding "
                        "level the --adapter was trained with for --prompt_style training (see sft.py --level)")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows")
    p.add_argument("--overwrite", action="store_true", help="ignore any existing output and start fresh")
    p.add_argument("--stats_only", action="store_true",
                   help="skip generation, just (re)compute evaluation.json from an existing --output")
    p.add_argument("--tie_tol", type=float, default=0.5,
                   help="--approach pointwise only: |score_a - score_b| at or below this counts as "
                        "that vote trial's preference being 'tie'")
    p.add_argument("--n_votes", type=int, default=10,
                   help="number of times to independently sample each prompt for the majority-voting scheme")
    p.add_argument("--vote_temperature", type=float, default=0.7,
                   help="sampling temperature shared by all --n_votes trials (temperature=0.0 would make "
                        "every trial identical, defeating the point of voting)")
    # vLLM engine knobs
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_num_seqs", type=int, default=16)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset download / audio caching
# --------------------------------------------------------------------------- #

def ensure_audio_file(cache_dir: Path, name: str, decoder) -> Path:
    """Return the local wav path for a dataset audio cell, decoding+saving it if missing."""
    path = cache_dir / f"{name}.wav"
    if path.is_file():
        return path
    import soundfile as sf

    samples = decoder.get_all_samples()
    data = samples.data.numpy()  # (channels, n_samples)
    if data.shape[0] == 1:
        data = data[0]
    else:
        data = data.T
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, samples.sample_rate)
    return path


def load_records(args) -> list[dict]:
    """Download (if needed) speakbench and materialize every row's audio as local wavs."""
    from datasets import load_dataset

    print(f"loading {args.dataset_name} (downloads to the HF cache on first use)...")
    ds = load_dataset(args.dataset_name, split=args.split)

    cache_dir = Path(args.audio_cache_dir)
    records = []
    for row in ds:
        idx = row["i"]
        instruction_path = ensure_audio_file(cache_dir, f"{idx}_instruction", row["instruction"])
        audio_a_path = ensure_audio_file(cache_dir, f"{idx}_audio_a", row["audio_a"])
        audio_b_path = ensure_audio_file(cache_dir, f"{idx}_audio_b", row["audio_b"])
        records.append({
            "i": idx,
            "instruction_ID": row["instruction_ID"],
            "instruction_text": row["instruction_text"],
            "instruction_audio_path": str(instruction_path),
            "audio_a_path": str(audio_a_path),
            "audio_b_path": str(audio_b_path),
            "label": row["label"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
        })
    print(f"{len(records)} rows ready ({cache_dir} holds the decoded wavs)")
    return records


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def majority_vote(votes: list[str]) -> str | None:
    """Strict-plurality winner among `votes` ('a'/'b'/'tie').

    Returns None if `votes` is empty (no trial produced a usable vote -- a
    different failure mode than a genuine tie). Returns "tie" when the top two
    vote counts are equal (e.g. 4/4/2), since no candidate has a real majority.
    """
    if not votes:
        return None
    counts = Counter(votes).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return "tie"


def generate(args, records, out_path: Path) -> None:
    """Run vLLM over any rows not already present in `out_path` and append them."""
    done: set = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
        print(f"resuming: {len(done)} rows already in {out_path}")
    elif args.overwrite and out_path.exists():
        out_path.unlink()

    todo = [r for r in records if r["i"] not in done]
    if not todo:
        print("nothing to generate")
        return

    from transformers import Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)

    llm = build_llm(args, audio_limit=3 if args.approach == "pairwise" else 2)

    if not args.adapter:
        adapter_path = f"./qwen-omni-thinker-sft-vllm/level{args.level}"
    else: 
        adapter_path = args.adapter
    lora_request = (
        None if args.no_lora
        else LoRARequest(
            lora_name="sft_adapter", lora_int_id=1,
            lora_path=str(Path(adapter_path).resolve()),
        )
    )

    sampling_params = SamplingParams(
        temperature=args.vote_temperature, max_tokens=args.max_new_tokens,
        seed=args.seed, n=args.n_votes,
    )

    if args.approach == "pointwise":
        # Build every prompt up front (2 per row: candidate a, candidate b), then
        # run one batched generate with --n_votes samples per prompt.
        work: list[tuple] = []  # (row_i, candidate) aligned with vllm_inputs
        vllm_inputs: list[dict] = []
        for rec in todo:
            for cand in CANDIDATES:
                work.append((rec["i"], cand))
                vllm_inputs.append(build_candidate_request(
                    processor, rec, rec[f"audio_{cand}_path"], args.prompt_style, args.level,
                ))

        outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        trials_by_row: dict = {rec["i"]: {"a": [], "b": []} for rec in todo}
        for (row_i, cand), out in zip(work, outputs):
            for completion in out.outputs:
                raw = completion.text.strip()
                parsed = extract_json(raw)
                score = coerce_score(parsed.get("overall_score")) if isinstance(parsed, dict) else None
                trials_by_row[row_i][cand].append({"raw": raw, "parsed": parsed, "score": score})

        n_invalid = 0
        with out_path.open("a", encoding="utf-8") as fout:
            for rec in todo:  # preserve input order
                trials_a = trials_by_row[rec["i"]]["a"]
                trials_b = trials_by_row[rec["i"]]["b"]
                votes = [
                    pref for trial_a, trial_b in zip(trials_a, trials_b)
                    if (pref := _preference(trial_a["score"], trial_b["score"], args.tie_tol)) is not None
                ]
                decision = majority_vote(votes)
                if decision is None:
                    n_invalid += 1
                fout.write(json.dumps({
                    "i": rec["i"],
                    "instruction_ID": rec["instruction_ID"],
                    "instruction_text": rec["instruction_text"],
                    "model_a": rec["model_a"],
                    "model_b": rec["model_b"],
                    "label": rec["label"],
                    "prompt_style": args.prompt_style,
                    "approach": args.approach,
                    "n_votes": args.n_votes,
                    "votes": votes,
                    "decision": decision,
                    "audio_a_trials": trials_a,
                    "audio_b_trials": trials_b,
                }, ensure_ascii=False) + "\n")

        print(f"wrote {out_path}: {len(todo)} rows, {len(work) * args.n_votes} evaluations "
              f"({n_invalid} rows with no valid majority)")

    else:  # pairwise: one generation per row, comparing both candidates directly
        vllm_inputs = [build_pairwise_request(processor, rec, args.level) for rec in todo]

        outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        n_invalid = 0
        with out_path.open("a", encoding="utf-8") as fout:
            for rec, out in zip(todo, outputs):  # preserve input order
                trials = []
                votes = []
                for completion in out.outputs:
                    raw = completion.text.strip()
                    parsed = extract_json(raw)
                    decision = _normalize_decision(parsed.get("label")) if isinstance(parsed, dict) else None # decision
                    trials.append({"raw": raw, "parsed": parsed, "decision": decision})
                    if decision is not None:
                        votes.append(decision)
                final_decision = majority_vote(votes)
                if final_decision is None:
                    n_invalid += 1
                fout.write(json.dumps({
                    "i": rec["i"],
                    "instruction_ID": rec["instruction_ID"],
                    "instruction_text": rec["instruction_text"],
                    "model_a": rec["model_a"],
                    "model_b": rec["model_b"],
                    "label": rec["label"],
                    "prompt_style": args.prompt_style,
                    "approach": args.approach,
                    "n_votes": args.n_votes,
                    "votes": votes,
                    "decision": final_decision,
                    "trials": trials,
                }, ensure_ascii=False) + "\n")

        print(f"wrote {out_path}: {len(todo)} rows ({n_invalid} rows with no valid majority)")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _preference(s_a: float | None, s_b: float | None, tol: float) -> str | None:
    if s_a is None or s_b is None:
        return None
    if abs(s_a - s_b) <= tol:
        return "tie"
    return "a" if s_a > s_b else "b"


def _normalize_decision(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in ("a", "b", "tie") else None


def _row_prediction(row: dict) -> str | None:
    """Return the row's majority-vote a/b/tie decision, or None if no trial produced a usable vote."""
    decision = row.get("decision")
    return decision if decision in ("a", "b", "tie") else None


def compute_statistics(output_path: Path, stats_path: Path, approach: str) -> dict:
    rows = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    n_scored = n_invalid = 0
    gt_pref: Counter = Counter()
    pred_pref: Counter = Counter()
    confusion: Counter = Counter()
    correct = correct_no_gt_tie = n_no_gt_tie = 0
    for row in rows:
        gt = row["label"]
        gt_pref[gt] += 1
        pred = _row_prediction(row)
        if pred is None:
            n_invalid += 1
            continue
        n_scored += 1
        pred_pref[pred] += 1
        confusion[(gt, pred)] += 1
        hit = gt == pred
        correct += hit
        if gt != "tie":
            n_no_gt_tie += 1
            correct_no_gt_tie += hit

    stats = {
        "output": str(output_path),
        "approach": approach,
        "n_rows": len(rows),
        "n_scored": n_scored,
        "n_invalid_prediction": n_invalid,
        "pairwise_accuracy": (correct / n_scored) if n_scored else None,
        "pairwise_accuracy_excluding_gt_ties": (correct_no_gt_tie / n_no_gt_tie) if n_no_gt_tie else None,
        "gt_preference_distribution": dict(gt_pref),
        "pred_preference_distribution": dict(pred_pref),
        "confusion_matrix": {f"gt_{gg}__pred_{pp}": int(c) for (gg, pp), c in sorted(confusion.items())},
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return stats


def main(args):
    if args.approach == "pairwise" and args.prompt_style == "training":
        raise SystemExit(
            "--approach pairwise requires --prompt_style task_specific "
            "(the training template's ASR/LLM/TTS slots only score one candidate response at a time)"
        )

    out_path = (
        Path(args.output) if args.output
        else Path("results/speakbench") / args.prompt_style / args.approach / args.level / "trained_model.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        records = load_records(args)
        if args.limit is not None:
            records = records[: args.limit]
        generate(args, records, out_path)

    if not out_path.exists():
        print("no output file to score; skipping statistics")
        return

    stats_path = Path(args.stats_output) if args.stats_output else out_path.parent / "evaluation.json"
    compute_statistics(out_path, stats_path, args.approach)


if __name__ == "__main__":
    main(parse_args())
