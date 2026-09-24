"""LoRA SFT of the Qwen2.5-Omni Thinker as a single-pipeline evaluation model.

The model is conditioned on (spoken request, ASR transcription, LLM answer) and
trained to emit the `overall_evaluation` JSON. No pairwise comparison here.
"""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

from utils.create_templates import LEVELS, build_dataset
from utils.data_collator import QwenOmniDataCollator

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

DEFAULT_OUTPUT_DIR = "./qwen-omni-thinker-sft"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations_file", default="data/train_data.jsonl")
    p.add_argument(
        "--audio_dir",
        default=None,
        help="directory holding the wavs; leave unset if audio_path in the data is absolute",
    )
    p.add_argument(
        "--level",
        choices=LEVELS,
        default="C",
        help=(
            "How much of `overall_evaluation` to train the model on: "
            "A = overall_score only, "
            "B = A + overall_notes (rationale), "
            "C = B + per-stage assessments and the compounding error-propagation analysis."
        ),
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help=f"adapter save dir; defaults to '{DEFAULT_OUTPUT_DIR}-level<LEVEL>'",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="src")
    p.add_argument("--run_name", default=None, help="wandb run name; defaults to the output_dir basename")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


def main(args):
    # Different levels target different output formats, so their adapters must
    # not collide -- keep each level's checkpoints in their own folder.
    args.output_dir = args.output_dir or f"{DEFAULT_OUTPUT_DIR}-level{args.level}"

    report_to = "none" if args.no_wandb else "wandb"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        # keep wandb run files next to the checkpoints, as the RL script does
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", wandb_dir)

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # swap to "sdpa" if flash-attn is not installed
    )
    model.config.use_cache = False

    # LoRA on the language backbone only; the audio encoder and vision tower stay frozen.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r".*\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
        exclude_modules=r".*(audio_tower|visual)\..*",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    train_dataset = build_dataset(args.annotations_file, args.audio_dir, level=args.level)
    collator = QwenOmniDataCollator(processor=processor)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=10,  # ~3% of a 1-epoch run over the current dataset
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
