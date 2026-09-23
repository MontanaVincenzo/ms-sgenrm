#!/usr/bin/env python3
"""Concatenate instruction + audio_a + audio_b (2s silence between each) per sample id.

Expects files named "{id}_instruction.wav", "{id}_audio_a.wav", "{id}_audio_b.wav"
in the data directory. Writes "{id}_concatenated.wav" into the same directory.
"""
import argparse
import re
import subprocess
from pathlib import Path

SILENCE_SECONDS = 2
TARGET_SAMPLE_RATE = 24000  # highest rate seen across inputs; others get upsampled


def ffprobe_rate(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def find_ids(data_dir: Path) -> list[str]:
    ids = set()
    for f in data_dir.glob("*_instruction.wav"):
        m = re.match(r"^(.+)_instruction\.wav$", f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=lambda x: (len(x), x))


def concatenate(sample_id: str, data_dir: Path, out_dir: Path, sample_rate: int) -> None:
    instruction = data_dir / f"{sample_id}_instruction.wav"
    audio_a = data_dir / f"{sample_id}_audio_a.wav"
    audio_b = data_dir / f"{sample_id}_audio_b.wav"
    for p in (instruction, audio_a, audio_b):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    out_path = out_dir / f"{sample_id}_concatenated.wav"

    filter_complex = (
        f"[0:a]aresample={sample_rate},aformat=channel_layouts=mono[a0];"
        f"[1:a]aresample={sample_rate},aformat=channel_layouts=mono[a1];"
        f"[2:a]aresample={sample_rate},aformat=channel_layouts=mono[a2];"
        f"anullsrc=r={sample_rate}:cl=mono:d={SILENCE_SECONDS}[sil1];"
        f"anullsrc=r={sample_rate}:cl=mono:d={SILENCE_SECONDS}[sil2];"
        f"[a0][sil1][a1][sil2][a2]concat=n=5:v=0:a=1[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(instruction),
        "-i", str(audio_a),
        "-i", str(audio_b),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"wrote {out_path}")


def combine_all(ids: list[str], out_dir: Path, sample_rate: int) -> None:
    inputs = [out_dir / f"{sample_id}_concatenated.wav" for sample_id in ids]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Missing per-id output: {p}")

    combined_path = out_dir / "all_examples_concatenated.wav"

    filter_parts = []
    concat_inputs = []
    for i in range(len(inputs)):
        filter_parts.append(
            f"[{i}:a]aresample={sample_rate},aformat=channel_layouts=mono[a{i}]"
        )
        concat_inputs.append(f"[a{i}]")
        if i < len(inputs) - 1:
            filter_parts.append(
                f"anullsrc=r={sample_rate}:cl=mono:d={SILENCE_SECONDS}[sil{i}]"
            )
            concat_inputs.append(f"[sil{i}]")

    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_inputs)
    filter_complex += f"concat=n={len(concat_inputs)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", filter_complex, "-map", "[out]", str(combined_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"wrote {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/vmontana/reward_model_icaasp/code/sft/data/concatenated_audios"),
    )
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Defaults to --data-dir")
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--combine", action="store_true",
                         help="Also concatenate all per-id outputs (in id order, 2s silence "
                              "between examples) into all_examples_concatenated.wav")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = find_ids(args.data_dir)
    if not ids:
        raise SystemExit(f"No *_instruction.wav files found in {args.data_dir}")

    for sample_id in ids:
        concatenate(sample_id, args.data_dir, out_dir, args.sample_rate)

    if args.combine:
        combine_all(ids, out_dir, args.sample_rate)


if __name__ == "__main__":
    main()
