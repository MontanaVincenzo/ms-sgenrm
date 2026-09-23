"""Split the evaluated dataset into train / eval / test jsonl files.

Source is the output of the LLM-judge stage (records carrying
`pipeline*.overall_evaluation`). Records with no evaluation on either pipeline
are dropped by default -- they have no SFT target. Records missing any of
their three audio files (input request, pipeline1 tts, pipeline2 tts) are
always dropped, since they can't be used regardless of --keep-unevaluated.
The split is a seeded shuffle so it is reproducible.

Records with `annotation_id` in TEST_ID_RANGE are always routed to the test
set and never appear in train or eval. The remaining records are split by
topic (`additional_info.topic`) so that no topic appears in both train and
eval; --eval-frac controls how many topics are earmarked to the eval side.
Both the train and eval sets are then capped (--train-size / --eval-size),
selected via a round-robin draw across their earmarked topics (one sample
per topic per round) so that topic coverage is maximized before any topic
contributes a second sample.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

TEST_ID_RANGE = range(1949, 2000)  # [1949, 1999] inclusive, reserved for test_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=HERE / "dataset_eval.jsonl")
    p.add_argument("--train-out", type=Path, default=HERE / "train_data.jsonl")
    p.add_argument("--eval-out", type=Path, default=HERE / "eval_data.jsonl")
    p.add_argument("--test-out", type=Path, default=HERE / "test_data.jsonl")
    p.add_argument("--eval-frac", type=float, default=0.10, help="fraction of the non-test pool earmarked (by whole topic) to the eval side")
    p.add_argument("--train-size", type=int, default=700, help="max train records, picked to maximize topic diversity")
    p.add_argument("--eval-size", type=int, default=100, help="max eval records, picked to maximize topic diversity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--keep-unevaluated",
        action="store_true",
        help="keep records with no overall_evaluation on either pipeline",
    )
    return p.parse_args()


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


def diversify_select(by_topic: dict, target: int, rng: random.Random) -> list:
    """Round-robin one record per topic per pass, so topic breadth is
    maximized before any topic contributes a second record."""
    pools = {topic: list(recs) for topic, recs in by_topic.items()}
    for recs in pools.values():
        rng.shuffle(recs)
    topics = list(pools.keys())

    selected = []
    while len(selected) < target:
        rng.shuffle(topics)
        drew_any = False
        for topic in topics:
            if len(selected) >= target:
                break
            if pools[topic]:
                selected.append(pools[topic].pop())
                drew_any = True
        if not drew_any:
            break
    return selected


def main(args):
    if not 0.0 < args.eval_frac < 1.0:
        raise SystemExit("--eval-frac must be in (0, 1)")

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

    test_records = [r for r in records if r.get("annotation_id") in TEST_ID_RANGE]
    pool = [r for r in records if r.get("annotation_id") not in TEST_ID_RANGE]

    # Group the remaining records by topic and assign whole topics to eval,
    # so no topic appears in both train and eval.
    by_topic = defaultdict(list)
    for r in pool:
        by_topic[topic_of(r)].append(r)
    topics = list(by_topic.keys())
    rng = random.Random(args.seed)
    rng.shuffle(topics)

    n_eval_target = round(len(pool) * args.eval_frac)
    if not 0 < n_eval_target < len(pool):
        raise SystemExit(f"split leaves an empty side (n_eval={n_eval_target}, n_total={len(pool)})")

    eval_by_topic = {}
    train_by_topic = {}
    n_eval_earmarked = 0
    for topic in topics:
        if n_eval_earmarked < n_eval_target:
            eval_by_topic[topic] = by_topic[topic]
            n_eval_earmarked += len(by_topic[topic])
        else:
            train_by_topic[topic] = by_topic[topic]

    n_train_pool = sum(len(recs) for recs in train_by_topic.values())
    train_records = diversify_select(train_by_topic, args.train_size, rng)
    n_train_topics = len({topic_of(r) for r in train_records})

    n_eval_pool = sum(len(recs) for recs in eval_by_topic.values())
    eval_records = diversify_select(eval_by_topic, args.eval_size, rng)
    n_eval_topics = len({topic_of(r) for r in eval_records})

    for path, rows in (
        (args.train_out, train_records),
        (args.eval_out, eval_records),
        (args.test_out, test_records),
    ):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"read {total} records from {args.input.name}"
        + (f" (dropped {n_unevaluated} unevaluated)" if n_unevaluated else "")
        + (f" (dropped {n_missing_audio} missing audio)" if n_missing_audio else "")
    )
    print(f"  test:  {len(test_records)} -> {args.test_out.name} (annotation_id in [{TEST_ID_RANGE.start}, {TEST_ID_RANGE.stop - 1}])")
    print(f"  train: {len(train_records)} -> {args.train_out.name} ({n_train_topics} topics, drawn from {n_train_pool} eligible across {len(train_by_topic)} topics)")
    print(f"  eval:  {len(eval_records)} -> {args.eval_out.name} ({n_eval_topics} topics, drawn from {n_eval_pool} eligible across {len(eval_by_topic)} topics)")


if __name__ == "__main__":
    main(parse_args())
