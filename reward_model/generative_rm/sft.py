"""LoRA SFT of the Qwen2.5-Omni Thinker as a single-pipeline evaluation model.

The model is conditioned on (spoken request, ASR transcription, LLM answer) and
trained to emit the `overall_evaluation` JSON. No pairwise comparison here.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

from utils.create_templates import LEVELS, build_dataset
from utils.data_collator import QwenOmniDataCollator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument(
        "--level",
        choices=LEVELS,
        default=None,
        help=(
            "Override `generative_rm.level` from the config. How much of `overall_evaluation` to train on: "
            "A = overall_score only, "
            "B = A + overall_notes (rationale), "
            "C = B + per-stage assessments and the compounding error-propagation analysis."
        ),
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

    sft_config = config["generative_rm"]
    required = ["model_id", "level", "output_dir", "epochs", "per_device_batch_size", "grad_accum", "learning_rate",
                "warmup_steps", "seed", "use_wandb", "wandb_project", "run_name", "lora"]
    missing = [k for k in required if k not in sft_config]
    if missing:
        print(f"Error: missing keys in 'generative_rm' section of {args.config_path}: {missing}", file=sys.stderr)
        sys.exit(1)
    level_override = args.level
    for key in required:
        setattr(args, key, sft_config[key])
    args.level = level_override or args.level
    if args.level not in LEVELS:
        print(f"Error: generative_rm.level must be one of {LEVELS}, got {args.level!r}", file=sys.stderr)
        sys.exit(1)
    args.epochs = float(args.epochs)
    args.learning_rate = float(args.learning_rate)  # PyYAML reads `1e-4` (no dot) as a string

    # Relative paths (data_dir, output_dir and the audio paths stored in the jsonl, e.g.
    # "data/input_audios_original/x.wav") are relative to the repo root, i.e. the folder containing config.yaml.
    args.repo_root = Path(args.config_path).resolve().parent
    data_config = config["data_generation"]
    args.annotations_file = args.repo_root / data_config["data_dir"] / data_config["split"]["train_file"]
    # Different levels target different output formats, so their adapters must
    # not collide -- keep each level's checkpoints in their own folder.
    args.output_dir = str(args.repo_root / f"{args.output_dir}-level{args.level}")
    return args


def main(args):
    # With >1 visible GPU and no torchrun, Trainer falls back to nn.DataParallel, which splits every tensor on
    # dim 0 -- but Qwen-Omni's audio features are packed per audio clip, not per sample, so the replicas get
    # mismatched audio (IndexError in chunk_and_pad_features). Use one GPU or torchrun (DDP) instead.
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        sys.exit(
            f"Error: {torch.cuda.device_count()} GPUs visible without a distributed launcher, which makes Trainer use "
            "DataParallel (incompatible with packed audio features). Run with CUDA_VISIBLE_DEVICES=0, or "
            "`torchrun --nproc_per_node <N> sft.py ...` for multi-GPU."
        )

    report_to = "wandb" if args.use_wandb else "none"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        # keep wandb run files next to the checkpoints, as the RL script does
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", wandb_dir)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # swap to "sdpa" if flash-attn is not installed
    )
    model.config.use_cache = False

    # LoRA on the language backbone only; the audio encoder and vision tower stay frozen.
    peft_config = LoraConfig(
        r=args.lora["r"],
        lora_alpha=args.lora["alpha"],
        target_modules=r".*\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
        exclude_modules=r".*(audio_tower|visual)\..*",
        lora_dropout=float(args.lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    train_dataset = build_dataset(args.annotations_file, level=args.level, root_dir=args.repo_root)
    collator = QwenOmniDataCollator(processor=processor)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        loss_type="nll",  # trl's default "chunked_nll" patch breaks on Qwen2.5-Omni's partial forward
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to=report_to,
        run_name=args.run_name or os.path.basename(args.output_dir.rstrip("/")),
        packing=False,  # must be off for multimodal
        remove_unused_columns=False,  # keep the "messages" column for the collator
        dataset_kwargs={"skip_prepare_dataset": True},  # the collator does all tokenisation
        label_names=["labels"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main(parse_args())
