

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from qwen_omni_utils import process_mm_info
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from eval import AUDIO_SAMPLE_RATE, BASE_MODEL, _sanitize, build_llm, coerce_score, extract_json
from utils.create_templates import LEVELS

DATASET_NAME = "potsawee/speakbench-v1-labelled-508"
CANDIDATES = ("a", "b")



SYSTEM_PROMPT = """
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

Avoid position bias and don’t let response length influence your evaluation. After your analysis, output valid JSON with exactly two keys:
’reasoning’ (your explanation of the comparison) and ’label’ (a string value: ’A’ if the first audio is better, ’B’ if the second audio is better, or
’tie’ if they are equally good/bad. Please use "tie" sparingly, and only when you absolutely cannot choose the winner.)
"""


EXAMPLES_LABELS = ["tie", "tie", "a", "b"]
CONCAT_EXAMPLES_PATH = "/home/vmontana/reward_model_icaasp/code/sft/data/concatenated_audios/all_examples_concatenated.wav"

def build_conversation(test_example_audio_path) -> list[dict]:

    examples_str = ""
    for i, label in enumerate(EXAMPLES_LABELS):
        examples_str += f"Example {i+1}: Label: {label}\n"

    user_message = ("Please analyze which of the two recordings "
    "follows the instruction better, or tie. Respond ONLY in text and "
    "output valid JSON with keys 'reasoning' and 'label' (string, 'A', 'B' or 'tie').\n"
    'Output format example: {"reasoning": "...", "label": "A"}')

    structure_info = ("This single recording contains, in order: the spoken "
    "instruction, a 2-second silence, Response A, a 2-second silence, then "
    "Response B. Use the pauses to locate each segment.")

    user_content1 = [
        {"type": "text", "text": (
            f"Here are {len(EXAMPLES_LABELS)} examples for reference. "
            "For each example, you will here first instruction, a 2-second silence, "
            "Response A, a 2-second silence, then Response B, then again a 2-second silence.")},
        {"type": "audio", "audio": str(CONCAT_EXAMPLES_PATH)},
        {"type": "text", "text": examples_str},
    ]
    user_content2 = [
        {"type": "text", "text": structure_info},
        {"type": "audio", "audio": str(test_example_audio_path)},
        {"type": "text", "text": user_message }
    ]
    """{"role": "user", "content": user_content1},
    {"role": "assistant", "content": [
        {"type": "text", "text": "I understand these examples. I’ll apply this understanding to analyze the new audio clips you provide."},
    ]},"""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content1},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I understand these examples. I’ll apply this understanding to analyze the new audio clips you provide."},
        ]},
        {"role": "user", "content": user_content2},
    ]


def concat_audios(
    instruction_path, audio_a_path, audio_b_path,
    silence_seconds: float, out_dir: Path, sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path:
    """ffmpeg-concatenate instruction + audio_a + audio_b, with `silence_seconds` of
    silence between each clip. Every input is resampled to a common mono `sample_rate`
    first since instruction/a/b are not guaranteed to share one (e.g. TTS outputs at
    24kHz vs. 16kHz ASR audio), which ffmpeg's concat filter requires.

    Cached by content path so repeat eval runs over the same records reuse the file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{instruction_path}|{audio_a_path}|{audio_b_path}|{silence_seconds}|{sample_rate}".encode()
    ).hexdigest()[:16]
    out_path = out_dir / f"{key}.wav"
    if out_path.is_file():
        return out_path

    filter_complex = (
        f"[0:a]aresample={sample_rate},aformat=channel_layouts=mono[a0];"
        f"[1:a]aresample={sample_rate},aformat=channel_layouts=mono[a1];"
        f"[2:a]aresample={sample_rate},aformat=channel_layouts=mono[a2];"
        f"anullsrc=r={sample_rate}:cl=mono:d={silence_seconds}[sil1];"
        f"anullsrc=r={sample_rate}:cl=mono:d={silence_seconds}[sil2];"
        f"[a0][sil1][a1][sil2][a2]concat=n=5:v=0:a=1[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(instruction_path),
        "-i", str(audio_a_path),
        "-i", str(audio_b_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def render_request(processor, conversation: list[dict]) -> dict:
    """Render a chat-message conversation into a vLLM request dict."""
    conversation = _sanitize(conversation)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, _images, _videos = process_mm_info([conversation], use_audio_in_video=False)
    request = {"prompt": prompt}
    if audios:
        request["multi_modal_data"] = {"audio": [(a, AUDIO_SAMPLE_RATE) for a in audios]}
    return request


def build_request(processor, rec: dict, concat_cache_dir: Path) -> dict:
    concatenated_test_audio = concat_audios(
        rec["instruction_audio_path"], rec["audio_a_path"], rec["audio_b_path"],
        silence_seconds=2.0, out_dir=concat_cache_dir,
    )
    conversation = build_conversation(concatenated_test_audio)
    return render_request(processor, conversation)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default="train")
    p.add_argument("--audio_cache_dir", default="data/speakbench_audio",
                   help="local wav cache; existing files are reused, missing ones are decoded from the HF dataset")
    p.add_argument("--output", default=None)
    p.add_argument("--stats_output", default=None,
                   help="path for the statistics JSON (default: <output dir>/evaluation.json)")
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
                   help="|score_a - score_b| at or below this counts as a predicted tie")
    # vLLM engine knobs
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_model_len", type=int, default=32768)
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

    llm = build_llm(args, audio_limit=2)

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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)

    concat_cache_dir = Path(args.audio_cache_dir) / "concatenated_test"
    vllm_inputs = [build_request(processor, rec, concat_cache_dir) for rec in todo]

    outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

    n_bad = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for rec, out in zip(todo, outputs):  # preserve input order
            raw = out.outputs[0].text.strip()
            parsed = extract_json(raw)
            if parsed is None:
                n_bad += 1
                parsed = {"_raw": raw, "_parse_error": True}
            fout.write(json.dumps({
                "i": rec["i"],
                "instruction_ID": rec["instruction_ID"],
                "instruction_text": rec["instruction_text"],
                "model_a": rec["model_a"],
                "model_b": rec["model_b"],
                "label": rec["label"],
                "pairwise_evaluation": parsed,
            }, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}: {len(todo)} rows ({n_bad} unparseable)")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_LABEL_TO_PREFERENCE = {"1": "a", "2": "b", "tie": "tie"}


def _normalize_decision(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in ("a", "b", "tie") else None


def _row_prediction(row: dict) -> tuple[str | None, bool]:
    """Return (predicted a/b/tie preference or None, whether the row's model output is missing)."""

    pairwise_eval = row.get("pairwise_evaluation")
    if pairwise_eval is None:
        return None, True
    pred = _normalize_decision(pairwise_eval.get("label")) if isinstance(pairwise_eval, dict) else None
    return pred, False


def compute_statistics(output_path: Path, stats_path: Path, tie_tol: float) -> dict:
    rows = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    n_scored = n_missing = n_invalid = 0
    gt_pref: Counter = Counter()
    pred_pref: Counter = Counter()
    confusion: Counter = Counter()
    correct = correct_no_gt_tie = n_no_gt_tie = 0
    for row in rows:
        gt = row["label"]
        gt_pref[gt] += 1
        pred, missing = _row_prediction(row)
        if missing:
            n_missing += 1
            continue
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
        "n_rows": len(rows),
        "n_scored": n_scored,
        "n_missing_prediction": n_missing,
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

    out_path = (
        Path(args.output) if args.output
        else Path("results/speakbench/concatenated_audios/trained_model.jsonl")
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
    compute_statistics(out_path, stats_path, args.tie_tol)


if __name__ == "__main__":
    main(parse_args())
