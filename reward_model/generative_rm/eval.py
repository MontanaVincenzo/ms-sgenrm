"""Run the LoRA-tuned Qwen2.5-Omni evaluator over `data/eval_data.jsonl` with vLLM.

For every pipeline of every record we feed the model the same prompt used in
training -- (spoken request audio, ASR transcription, LLM answer) -- and store the
evaluation JSON it generates. All prompts are submitted to vLLM in a single
`generate` call (continuous batching); the LoRA adapter is applied via a
`LoRARequest` without merging it into the base weights.

Output rows (one per input record):

    {
        "annotation_id": ...,
        "input_request": {"text": ..., "audio_path": ...},
        "pipeline1": {
            "asr": <transcription>, "llm": <answer>, "tts": <audio_path>,
            "perturbation": {"asr": <type|None>, "llm": <type|None>, "tts": <type|None>},
            "gt_evaluation": <dict from --input>, "model_evaluation": <dict>,
        },
        "pipeline2": {...},
    }

`model_evaluation` is the parsed JSON object; if the model emits something
unparseable it is stored as {"_raw": <text>, "_parse_error": true}.

After generation the script scores the predicted `overall_score` against the
ground-truth `overall_score` carried in `--input` and derives a pairwise
(pipeline1 vs pipeline2) preference accuracy. All of it is written to
`evaluation.json` next to `--output`.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.create_templates import LEVELS, PIPELINE_KEYS, build_conversation

BASE_MODEL = "Qwen/Qwen2.5-Omni-7B"
AUDIO_SAMPLE_RATE = 16000  # process_mm_info resamples every clip to 16 kHz

# Some (mostly base-model) generations report the score as a word rather than a
# 1-5 integer; map the common ones so they still count toward the metrics.
TEXT_SCORE_MAP = {
    "very low": 1, "lowest": 1, "worst": 1,
    "low": 2, "poor": 2, "bad": 2,
    "medium": 3, "moderate": 3, "mixed": 3, "average": 3, "fair": 3, "ok": 3,
    "high": 4, "good": 4,
    "very high": 5, "highest": 5, "excellent": 5, "perfect": 5, "best": 5,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/eval_data.jsonl")
    p.add_argument("--output", default="results/trained_model.jsonl")
    p.add_argument("--stats_output", default=None,
                   help="path for the statistics JSON (default: <output dir>/evaluation.json)")
    p.add_argument("--adapter", default="./qwen-omni-thinker-sft-vllm",
                   help="LoRA adapter dir; pass --no_lora to evaluate the base model")
    p.add_argument("--no_lora", action="store_true", help="run the base model without any adapter")
    p.add_argument("--level", choices=LEVELS, default="C",
                   help="must match the grounding level the --adapter was trained with (see sft.py --level)")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None, help="only the first N records")
    p.add_argument("--audio_dir", default=None, help="prefix for audio_path basenames if not absolute")
    p.add_argument("--overwrite", action="store_true", help="ignore any existing output and start fresh")
    p.add_argument("--stats_only", action="store_true",
                   help="skip generation, just (re)compute evaluation.json from an existing --output")
    # vLLM engine knobs
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_num_seqs", type=int, default=16)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _sanitize(conversation):
    """Drop None-valued content keys.

    `build_conversation` returns clean dicts, but the Qwen-Omni chat template
    routes on `'audio' in content` (key presence), so a stray `audio: None` on a
    text item would render as an <|AUDIO|> placeholder.
    """
    return [
        {
            **msg,
            "content": [{k: v for k, v in item.items() if v is not None} for item in msg["content"]],
        }
        for msg in conversation
    ]


def extract_json(text: str) -> dict | None:
    """Return the last well-formed JSON object embedded in `text`, or None."""
    decoder = json.JSONDecoder()
    best, i = None, 0
    while (start := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            best = obj
        i = end
    return best


def resolve_audio(raw_path: str, audio_dir: str | None) -> str:
    return str(Path(audio_dir) / Path(raw_path).name) if audio_dir else raw_path


def build_request(processor, input_audio_path, asr_text, llm_text, tts_audio_path, level: str = "C") -> dict:
    """Render one training-shaped prompt into a vLLM request dict."""
    conversation = _sanitize(
        build_conversation(input_audio_path, asr_text, llm_text, tts_audio_path, target=None, level=level)
    )
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, _images, _videos = process_mm_info([conversation], use_audio_in_video=False)
    request = {"prompt": prompt}
    if audios:
        request["multi_modal_data"] = {"audio": [(a, AUDIO_SAMPLE_RATE) for a in audios]}
    return request


def build_llm(args, audio_limit: int = 2) -> LLM:
    gpu_index = int(args.device.split(":")[1]) if ":" in args.device else int(args.device)
    return LLM(
        model=args.base_model,
        device_ids=[gpu_index],
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": audio_limit},
        enable_lora=not args.no_lora,
        max_lora_rank=16,
        seed=args.seed,
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def coerce_score(value) -> float | None:
    """Best-effort map an `overall_score` (GT or predicted) to a float, or None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        try:
            return float(s)
        except ValueError:
            pass
        for key, num in TEXT_SCORE_MAP.items():
            if key in s:
                return float(num)
    return None


def _preference(s1: float | None, s2: float | None, tol: float = 0.5) -> str | None:
    """Which pipeline the pair of scores prefers: 'pipeline1', 'pipeline2', 'tie'."""
    if s1 is None or s2 is None:
        return None
    if abs(s1 - s2) <= tol:
        return "tie"
    return "pipeline1" if s1 > s2 else "pipeline2"


def _load_scores(path: Path, key: str) -> dict:
    """{annotation_id: {pipeline_key: raw overall_score}} from a jsonl file.

    `key` is 'overall_evaluation' for the GT input or 'model_evaluation' for the
    predictions. Pipelines without an overall_score are simply omitted.
    """
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        per_pipeline = {}
        for pk in PIPELINE_KEYS:
            stage = rec.get(pk) or {}
            block = stage.get(key) or {}
            if isinstance(block, dict) and "overall_score" in block:
                per_pipeline[pk] = block["overall_score"]
        out[rec["annotation_id"]] = per_pipeline
    return out


def compute_statistics(input_path: str, output_path: Path, stats_path: Path) -> dict:
    """Score predicted `overall_score` vs GT and derive pairwise accuracy."""
    import numpy as np

    gt = _load_scores(Path(input_path), "overall_evaluation")
    pred = _load_scores(Path(output_path), "model_evaluation")
    common_ids = sorted(set(gt) & set(pred))

    # ---- per-pipeline absolute score prediction --------------------------- #
    g_vals: list[float] = []
    p_vals: list[float] = []
    n_pipelines_with_gt = n_missing = n_invalid = 0
    confusion: Counter = Counter()
    for aid in common_ids:
        for pk in PIPELINE_KEYS:
            if pk not in gt[aid]:
                continue
            n_pipelines_with_gt += 1
            gs = coerce_score(gt[aid][pk])
            if pk not in pred[aid]:
                n_missing += 1
                continue
            ps = coerce_score(pred[aid][pk])
            if gs is None or ps is None:
                n_invalid += 1
                continue
            g_vals.append(gs)
            p_vals.append(ps)
            confusion[(int(round(gs)), int(round(ps)))] += 1

    score_stats: dict = {
        "n_pipelines_with_gt": n_pipelines_with_gt,
        "n_scored": len(g_vals),
        "n_missing_prediction": n_missing,
        "n_invalid_prediction": n_invalid,
    }
    if g_vals:
        g = np.asarray(g_vals, dtype=float)
        p = np.asarray(p_vals, dtype=float)
        gr = np.round(g).astype(int)
        pr = np.round(p).astype(int)
        err = p - g
        score_stats.update({
            "exact_match_accuracy": float(np.mean(gr == pr)),
            "within_1_accuracy": float(np.mean(np.abs(gr - pr) <= 1)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_error_bias": float(np.mean(err)),
            "gt_distribution": {str(k): int(v) for k, v in sorted(Counter(gr.tolist()).items())},
            "pred_distribution": {str(k): int(v) for k, v in sorted(Counter(pr.tolist()).items())},
            "confusion_matrix": {
                f"gt{gg}_pred{pp}": int(c) for (gg, pp), c in sorted(confusion.items())
            },
        })
        if len(g_vals) > 1 and g.std() > 0 and p.std() > 0:
            try:
                from scipy.stats import pearsonr, spearmanr
                score_stats["pearson_r"] = float(pearsonr(g, p).statistic)
                score_stats["spearman_r"] = float(spearmanr(g, p).statistic)
            except Exception:  # scipy missing / degenerate input
                pass

    # ---- pairwise pipeline1 vs pipeline2 preference ---------------------- #
    pair_ids = [
        aid for aid in common_ids
        if "pipeline1" in gt[aid] and "pipeline2" in gt[aid]
    ]
    gt_pref: Counter = Counter()
    pred_pref: Counter = Counter()
    pair_confusion: Counter = Counter()
    n_pairs_scored = correct = 0
    n_pairs_no_gt_tie = correct_no_gt_tie = 0
    for aid in pair_ids:
        gp = _preference(coerce_score(gt[aid]["pipeline1"]), coerce_score(gt[aid]["pipeline2"]))
        pp = _preference(
            coerce_score(pred[aid].get("pipeline1")),
            coerce_score(pred[aid].get("pipeline2")),
        )
        if gp is None:
            continue
        gt_pref[gp] += 1
        if pp is None:
            continue
        n_pairs_scored += 1
        pred_pref[pp] += 1
        pair_confusion[(gp, pp)] += 1
        hit = gp == pp
        correct += hit
        if gp != "tie":
            n_pairs_no_gt_tie += 1
            correct_no_gt_tie += hit

    pairwise_stats = {
        "n_pairs_with_both_gt": len(pair_ids),
        "n_pairs_scored": n_pairs_scored,
        "pairwise_accuracy": (correct / n_pairs_scored) if n_pairs_scored else None,
        "pairwise_accuracy_excluding_gt_ties": (
            (correct_no_gt_tie / n_pairs_no_gt_tie) if n_pairs_no_gt_tie else None
        ),
        "gt_preference_distribution": dict(gt_pref),
        "pred_preference_distribution": dict(pred_pref),
        "confusion_matrix": {
            f"gt_{gg}__pred_{pp}": int(c) for (gg, pp), c in sorted(pair_confusion.items())
        },
    }

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "n_records_in_common": len(common_ids),
        "score_prediction": score_stats,
        "pairwise_preference": pairwise_stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return stats


def generate(args, records, out_path: Path) -> None:
    """Run vLLM over any records not already present in `out_path` and append them."""
    done: set = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["annotation_id"])
        print(f"resuming: {len(done)} records already in {out_path}")
    elif args.overwrite and out_path.exists():
        out_path.unlink()

    todo = [r for r in records if r["annotation_id"] not in done]
    if not todo:
        print("nothing to generate")
        return

    # Load the processor from the base model: the adapter dir from sft.py has no
    # preprocessor_config.json, and SFT added no tokens or template changes.
    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)

    # Build every prompt up front, then run one batched generate.
    rows_by_id: dict = {}
    work: list[tuple] = []          # (annotation_id, pipeline_key) aligned with vllm_inputs
    vllm_inputs: list[dict] = []
    for rec in todo:
        ann_id = rec["annotation_id"]
        rows_by_id[ann_id] = {
            "annotation_id": ann_id,
            "input_request": {
                "text": rec["input_request"]["text"],
                "audio_path": rec["input_request"]["audio_path"],
            },
        }
        audio_path = resolve_audio(rec["input_request"]["audio_path"], args.audio_dir)
        for pk in PIPELINE_KEYS:
            stage = rec.get(pk)
            if not stage:
                continue
            asr_out = stage["asr"]["output"]
            llm_out = stage["llm"]["output"]
            tts_out = stage["tts"]["output"]
            rows_by_id[ann_id][pk] = {
                "asr": asr_out,
                "llm": llm_out,
                "tts": tts_out,
                "perturbation": {
                    stage_key: (stage[stage_key].get("perturbation") or {}).get("type")
                    for stage_key in ("asr", "llm", "tts")
                },
                "gt_evaluation": stage.get("overall_evaluation"),
                "model_evaluation": None,
            }
            if not asr_out or not llm_out or not tts_out:
                continue
            tts_out = args.audio_dir / Path(stage["tts"]["output"]).name if args.audio_dir else Path(stage["tts"]["output"])
            if not tts_out.is_file():
                continue
            work.append((ann_id, pk))
            vllm_inputs.append(build_request(processor, audio_path, asr_out, llm_out, tts_out, level=args.level))

    if not vllm_inputs:
        print("no pipelines with both an ASR hypothesis and an LLM answer")
        return

    llm = build_llm(args)
    lora_request = (
        None if args.no_lora
        else LoRARequest(
            lora_name="sft_adapter", lora_int_id=1,
            lora_path=str(Path(args.adapter).resolve()),
        )
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)

    outputs = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

    n_bad = 0
    for (ann_id, pk), out in zip(work, outputs):
        raw = out.outputs[0].text.strip()
        parsed = extract_json(raw)
        if parsed is None:
            n_bad += 1
            parsed = {"_raw": raw, "_parse_error": True}
        rows_by_id[ann_id][pk]["model_evaluation"] = parsed

    with out_path.open("a", encoding="utf-8") as fout:
        for rec in todo:  # preserve input order
            fout.write(json.dumps(rows_by_id[rec["annotation_id"]], ensure_ascii=False) + "\n")

    print(f"wrote {out_path}: {len(todo)} records, {len(work)} evaluations ({n_bad} unparseable)")


def main(args):
    records = [
        json.loads(l)
        for l in Path(args.input).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        generate(args, records, out_path)

    if not out_path.exists():
        print("no output file to score; skipping statistics")
        return

    stats_path = (
        Path(args.stats_output) if args.stats_output
        else out_path.parent / "evaluation.json"
    )
    compute_statistics(args.input, out_path, stats_path)


if __name__ == "__main__":
    main(parse_args())
