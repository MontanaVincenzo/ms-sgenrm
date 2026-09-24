import asyncio
import json
import os
import random
import sys
import time
import wave
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from vllm import SamplingParams
import os

from orpheus_tts_pypi.orpheus_tts import OrpheusModel
# reuse the already-loaded SNAC model + its device from the package
from orpheus_tts_pypi.orpheus_tts.decoder import model as snac_model, snac_device

# Orpheus end-of-speech / end-of-turn tokens (49158 in engine_class.py is a bug: it's a text token)
STOP_TOKEN_IDS = [128258, 128009]
CUSTOM_TOKEN_OFFSET = 10          # <custom_token_N>: audio codes start at N=10
SNAC_CODEBOOK_SIZE = 4096
FRAME = 7                         # tokens per SNAC frame


def token_ids_to_codes(token_ids, tokenizer):
    """vLLM output token ids -> list of 7*k SNAC codes (or None if nothing usable)."""
    toks = tokenizer.convert_ids_to_tokens(token_ids)
    codes = []
    pos = 0
    for t in toks:
        if not (t.startswith("<custom_token_") and t.endswith(">")):
            continue
        try:
            n = int(t[len("<custom_token_"):-1])
        except ValueError:
            continue
        code = n - CUSTOM_TOKEN_OFFSET - (pos % FRAME) * SNAC_CODEBOOK_SIZE
        if code < 0:                       # control tokens (<custom_token_1/2/...>)
            continue
        codes.append(code)
        pos += 1
    n = (len(codes) // FRAME) * FRAME
    return codes[:n] or None


def codes_to_audio_bytes(codes):
    """Full (non-streaming) SNAC decode of a whole utterance -> int16 PCM bytes."""
    l0, l1, l2 = [], [], []
    for j in range(len(codes) // FRAME):
        i = j * FRAME
        f = codes[i:i + FRAME]
        if any(c < 0 or c >= SNAC_CODEBOOK_SIZE for c in f):
            continue  # drop only the bad frame instead of crashing the GPU
        l0.append(f[0])
        l1 += [f[1], f[4]]
        l2 += [f[2], f[3], f[5], f[6]]
    if not l0:
        return b""
    layers = [
        torch.tensor([l0], device=snac_device, dtype=torch.int32),
        torch.tensor([l1], device=snac_device, dtype=torch.int32),
        torch.tensor([l2], device=snac_device, dtype=torch.int32),
    ]
    with torch.inference_mode():
        audio = snac_model.decode(layers).squeeze().detach().cpu().numpy()
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


async def generate_token_ids(model, items, temperature=0.6, top_p=0.8,
                             max_tokens=2048, repetition_penalty=1.3):
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        stop_token_ids=STOP_TOKEN_IDS, repetition_penalty=repetition_penalty,
    )

    async def one(i, prompt, voice):
        prompt_string = model._format_prompt(prompt, voice)
        final = None
        async for out in model.engine.generate(
            prompt=prompt_string, sampling_params=sampling_params, request_id=f"req-{i}"
        ):
            final = out
        return list(final.outputs[0].token_ids) if final is not None else []

    return await asyncio.gather(
        *(one(i, p, v) for i, (_, p, v) in enumerate(items)),
        return_exceptions=True,
    )


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
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

    data_dir = Path(config["data_dir"])
    dataset_path = data_dir / config["input_request_output_file"]

    model = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len=16384,
        max_num_seqs=32,
    )
    voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

    with open(dataset_path, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    samples = [(s["annotation_id"], s["input_request"]["text"], random.choice(voices)) for s in dataset]

    t0 = time.monotonic()
    results = asyncio.run(generate_token_ids(model, samples))
    gen_dt = time.monotonic() - t0

    base_dir = data_dir / "input_audios_original"
    os.makedirs(base_dir, exist_ok=True)
    total_sec, ok, bad = 0.0, 0, 0
    for item, res, (annotation_id, input_request, voice) in zip(dataset, results, samples): 
        if isinstance(res, Exception) or not res:
            bad += 1
            print(f"[{annotation_id}] generation failed: {res!r}")
            continue
        try:
            codes = token_ids_to_codes(res, model.tokenizer)
            pcm = codes_to_audio_bytes(codes) if codes else b""
        except Exception as e:
            bad += 1
            print(f"[{annotation_id}] decode failed: {e!r}")
            continue
        if not pcm:
            bad += 1
            print(f"[{annotation_id}] no audio (voice={voice}): {input_request[:60]!r}")
            continue
        wav_path = os.path.join(base_dir, f"{annotation_id}_{voice}.wav")
        write_wav(wav_path, pcm)
        item["input_request"]["audio_path"] = wav_path
        total_sec += len(pcm) / 2 / 24000
        ok += 1

    print(f"\n{ok} ok / {bad} bad, {total_sec:.1f}s audio | gen {gen_dt:.1f}s")

    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")