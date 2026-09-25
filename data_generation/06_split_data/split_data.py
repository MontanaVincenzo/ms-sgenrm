"""Split the evaluated dataset into train / eval / val jsonl files.

Source is the output of the LLM-judge stage (records carrying
`pipeline*.overall_evaluation`). Records with no evaluation on either pipeline
are dropped by default -- they have no SFT target. Records missing any of
their three audio files (input request, pipeline1 tts, pipeline2 tts) are
always dropped, since they can't be used regardless of --keep-unevaluated.

The remaining records are split by topic (`additional_info.topic`) so that no
topic appears in more than one split. Topics are shuffled (seeded, so the split
is reproducible) and assigned whole to val, then eval, until each reaches its
target share; every other topic goes to train. The shares come from the
`data_generation.split` section of the config (train_frac / eval_frac /
val_frac, summing to 1). Because topics are assigned whole, the achieved
fractions can differ slightly from the targets; they are printed at the end.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SPLITS = ("train", "eval", "val")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--limit", type=int, default=None, help="only read the first N records (debugging)")
    p.add_argument(
        "--keep-unevaluated",
        action="store_true",
        help="keep records with no overall_evaluation on either pipeline",
    )
    args = p.parse_args()

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
    data_config = config["data_generation"]
    split_config = data_config["split"]
    data_dir = Path(data_config["data_dir"])

    args.input = data_dir / data_config["final_output_file"]
    args.seed = split_config["seed"]
    for split in SPLITS:
        setattr(args, f"{split}_out", data_dir / split_config[f"{split}_file"])
        setattr(args, f"{split}_frac", float(split_config[f"{split}_frac"]))
    return args


def has_evaluation(rec: dict) -> bool:
    return any(rec.get(pk, {}).get("overall_evaluation") for pk in ("pipeline1", "pipeline2"))


def topic_of(rec: dict) -> str:
    return rec.get("additional_info", {}).get("topic")


def audio_paths(rec: dict):
    yield rec.get("input_request", {}).get("audio_path")
    for pk in ("pipeline1", "pipeline2"):
        yield rec.get(pk, {}).get("tts", {}).get("output")


def has_all_audio(rec: dict) -> bool:
    return all(p and Path(p).exists() for p in audio_paths(rec))


def main(args):
    fracs = {split: getattr(args, f"{split}_frac") for split in SPLITS}
    if any(not 0.0 < f < 1.0 for f in fracs.values()):
        raise SystemExit(f"split fractions must each be in (0, 1), got {fracs}")
    if abs(sum(fracs.values()) - 1.0) > 1e-6:
        raise SystemExit(f"split fractions must sum to 1, got {fracs} (sum={sum(fracs.values())})")

    records = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]
    total = len(records)

    if not args.keep_unevaluated:
        records = [r for r in records if has_evaluation(r)]
    n_unevaluated = total - len(records)

    n_before_audio = len(records)
    records = [r for r in records if has_all_audio(r)]
    n_missing_audio = n_before_audio - len(records)

    # Group by topic and assign whole topics to a split, so no topic appears in two splits.
    by_topic = defaultdict(list)
    for r in records:
        by_topic[topic_of(r)].append(r)
    topics = list(by_topic.keys())
    rng = random.Random(args.seed)
    rng.shuffle(topics)

    # Fill the small splits first (val, then eval) up to their targets; train gets the rest.
    targets = {split: round(len(records) * fracs[split]) for split in ("val", "eval")}
    split_records = {split: [] for split in SPLITS}
    split_topics = {split: 0 for split in SPLITS}
    for topic in topics:
        split = next((s for s in ("val", "eval") if len(split_records[s]) < targets[s]), "train")
        recs = by_topic[topic]
        rng.shuffle(recs)
        split_records[split].extend(recs)
        split_topics[split] += 1

    empty = [split for split in SPLITS if not split_records[split]]
    if empty:
        raise SystemExit(f"split leaves {empty} empty ({len(records)} usable records, {len(topics)} topics)")

    for split in SPLITS:
        path = getattr(args, f"{split}_out")
        with path.open("w", encoding="utf-8") as f:
            for r in split_records[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"read {total} records from {args.input.name}"
        + (f" (dropped {n_unevaluated} unevaluated)" if n_unevaluated else "")
        + (f" (dropped {n_missing_audio} missing audio)" if n_missing_audio else "")
    )
    for split in SPLITS:
        n = len(split_records[split])
        path = getattr(args, f"{split}_out")
        print(
            f"  {split:<5}: {n} -> {path.name} ({n / len(records):.1%}, target {fracs[split]:.0%}; "
            f"{split_topics[split]} topics)"
        )


if __name__ == "__main__":
    main(parse_args())
