"""Judge self-consistency / agreement statistics.

Reads the raw per-pipeline verdict sets dumped by ``evaluate.py``
(``--votes-output``, default ``dataset_eval_votes.jsonl``) and reports how often
the judge's repeated evaluations of the same pipeline agree -- per field and end
to end -- averaged over the whole dataset.

Two agreement measures per field:
  * modal    -- fraction of runs landing on the single most common value
  * pairwise -- P(two distinct runs picked the same value); Fleiss-style
                percent agreement, which also penalises a split second place.
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

# Categorical / boolean fields to measure (free-text ``notes`` excluded).
FIELD_PATHS = (
    "asr_assessment.severity",
    "asr_assessment.meaning_preserved",
    "llm_assessment.answer_coherent",
    "llm_assessment.answer_hallucinated",
    "llm_assessment.intent_matches_answer",
    "llm_assessment.answer_serves_true_need",
    "tts_assessment.spoken_severity",
    "tts_assessment.spoken_meaning_preserved",
    "tts_assessment.voice_style_matches_expected",
    "tts_assessment.voice_style_appropriate",
    "tts_assessment.delivery_serves_true_need",
    "compounding.primary_failure_stage",
    "overall_score",
)


def _get(obj: dict, path: str):
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def modal_rate(values: list) -> float:
    """Fraction of runs landing on the single most common value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return math.nan
    return Counter(vals).most_common(1)[0][1] / len(vals)


def pairwise_rate(values: list) -> float:
    """P(two distinct runs picked the same value) -- Fleiss-style percent agreement."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return math.nan
    counts = Counter(vals)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _mean(xs) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.fmean(xs) if xs else math.nan


def _round(x, ndigits=4):
    return round(x, ndigits) if isinstance(x, (int, float)) and not math.isnan(x) else None


def compute_stats(records: list[dict]) -> dict:
    items = [r for r in records if len(r.get("verdicts") or []) >= 2]

    per_field = {}
    for path in FIELD_PATHS:
        modal, pair = [], []
        for r in items:
            vals = [_get(v, path) for v in r["verdicts"]]
            modal.append(modal_rate(vals))
            pair.append(pairwise_rate(vals))
        per_field[path] = {
            "mean_modal_agreement": _round(_mean(modal)),
            "mean_pairwise_agreement": _round(_mean(pair)),
        }

    os_modal, os_pair, os_std, os_range = [], [], [], []
    unanimous = 0
    chosen_dist = Counter()
    for r in items:
        scores = [s for s in (_get(v, "overall_score") for v in r["verdicts"]) if s is not None]
        if not scores:
            continue
        os_modal.append(modal_rate(scores))
        os_pair.append(pairwise_rate(scores))
        os_std.append(statistics.pstdev(scores) if len(scores) > 1 else 0.0)
        os_range.append(max(scores) - min(scores))
        unanimous += len(set(scores)) == 1
        chosen_dist[Counter(scores).most_common(1)[0][0]] += 1

    votes_valid = sum(len(r["verdicts"]) for r in items)
    votes_requested = sum(r.get("n_requested") or len(r["verdicts"]) for r in items)

    return {
        "model": MODEL,
        "n_pipelines": len(items),
        "n_pipelines_dropped_lt2_valid": len(records) - len(items),
        "votes_valid": votes_valid,
        "votes_requested": votes_requested,
        "vote_parse_rate": _round(votes_valid / votes_requested) if votes_requested else None,
        "overall_score": {
            "mean_modal_agreement": _round(_mean(os_modal)),
            "mean_pairwise_agreement": _round(_mean(os_pair)),
            "unanimous_rate": _round(unanimous / len(items)) if items else None,
            "mean_std": _round(_mean(os_std)),
            "mean_range": _round(_mean(os_range)),
            "chosen_score_distribution": {str(k): chosen_dist[k] for k in sorted(chosen_dist)},
        },
        "fields": per_field,
        "mean_field_modal_agreement": _round(
            _mean([per_field[p]["mean_modal_agreement"] for p in FIELD_PATHS])
        ),
        "mean_field_pairwise_agreement": _round(
            _mean([per_field[p]["mean_pairwise_agreement"] for p in FIELD_PATHS])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--votes",
        default="/home/vmontana/synthetic_data_generation/src/data/dataset_eval_votes.jsonl",
        help="raw verdict sets from evaluate.py --votes-output",
    )
    parser.add_argument(
        "--output",
        default="/home/vmontana/synthetic_data_generation/src/data/agreement_stats.json",
    )
    args = parser.parse_args()

    lines = Path(args.votes).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if not records:
        raise SystemExit(f"no vote records in {args.votes}")

    stats = compute_stats(records)
    Path(args.output).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    os_ = stats["overall_score"]
    print(f"pipelines: {stats['n_pipelines']}   valid votes: {stats['votes_valid']}/{stats['votes_requested']}")
    print(
        f"overall_score  modal={os_['mean_modal_agreement']:.1%}  "
        f"pairwise={os_['mean_pairwise_agreement']:.1%}  "
        f"unanimous={os_['unanimous_rate']:.1%}  "
        f"mean_std={os_['mean_std']:.2f}"
    )
    print(
        f"all fields     modal={stats['mean_field_modal_agreement']:.1%}  "
        f"pairwise={stats['mean_field_pairwise_agreement']:.1%}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
