"""Re-key a Qwen2.5-Omni Thinker LoRA adapter so vLLM can actually apply it.

The adapter produced by `sft.py` (PEFT over a standalone
`Qwen2_5OmniThinkerForConditionalGeneration`) stores tensors as
`base_model.model.model.layers.*`. vLLM loads the *full* Omni checkpoint and
exposes the text decoder as `language_model.model.layers.*`, remapping only
weights whose name starts with `thinker.`. A bare `model.` prefix is left
untouched, so vLLM finds no matching module and the adapter is a silent no-op.

This script rewrites `base_model.model.model.` -> `base_model.model.thinker.model.`
in every tensor key. vLLM then strips `base_model.model.`, its `hf_to_vllm_mapper`
turns `thinker.model.` into `language_model.model.`, and the module matches.

Usage:
    python convert_adapter_for_vllm.py ./qwen-omni-thinker-sft/checkpoint-290 \
        ./qwen-omni-thinker-sft-vllm
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

OLD_PREFIX = "base_model.model.model."
NEW_PREFIX = "base_model.model.thinker.model."
# files worth copying so the output dir is a self-contained adapter + processor
COPY_FILES = (
    "adapter_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
EXPLICIT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="adapter dir from sft.py")
    ap.add_argument("dst", type=Path, help="output dir for the vLLM-ready adapter")
    args = ap.parse_args()

    src_weights = args.src / "adapter_model.safetensors"
    if not src_weights.is_file():
        raise SystemExit(f"{src_weights} not found")

    args.dst.mkdir(parents=True, exist_ok=True)

    sd = load_file(str(src_weights))
    renamed = 0
    new_sd = {}
    for k, v in sd.items():
        if k.startswith(OLD_PREFIX):
            k = NEW_PREFIX + k[len(OLD_PREFIX):]
            renamed += 1
        new_sd[k] = v
    if renamed == 0:
        raise SystemExit(f"no keys start with {OLD_PREFIX!r}; nothing to do")
    save_file(new_sd, str(args.dst / "adapter_model.safetensors"))
    print(f"re-keyed {renamed}/{len(sd)} tensors -> {args.dst/'adapter_model.safetensors'}")

    for name in COPY_FILES:
        f = args.src / name
        if f.is_file():
            shutil.copy2(f, args.dst / name)

    # Give vLLM's PEFTHelper the plain-list target_modules form (it never sees
    # the renamed tree, and the regex + exclude_modules were only meaningful at
    # training time).
    cfg_path = args.dst / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["target_modules"] = EXPLICIT_TARGET_MODULES
    cfg.pop("exclude_modules", None)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg_path} with explicit target_modules")


if __name__ == "__main__":
    main()
