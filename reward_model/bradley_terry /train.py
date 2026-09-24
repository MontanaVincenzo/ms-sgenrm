from utils.reward_model import BradleyTerryRewardModel
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, get_cosine_schedule_with_warmup
from qwen_omni_utils import process_mm_info  # pip install qwen-omni-utils
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from utils.prepare_data import PairDataset, build_inputs
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from functools import partial
import json
import wandb
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

def load_model(model_id, checkpoint_dir, device):

    # 1. Load base model directly onto this process's GPU
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        **({"local_files_only": True} if checkpoint_dir is not None else {}),
    )
    model.config.use_cache = False

    # 2. Load LoRA adapter on top of the base model (skipped when starting from base).
    #    is_trainable=True so an SFT adapter can be fine-tuned in place under --train_lm
    #    (PeftModel.from_pretrained otherwise loads it in inference_mode).
    if checkpoint_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_dir,
            is_trainable=True,
            device_map={"": device},
        )

    # 3. Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        checkpoint_dir if checkpoint_dir is not None else model_id,
        **({"local_files_only": True} if checkpoint_dir is not None else {}),
    )

    return model, processor



def collate_fn(batch, processor, _printed=[False]):
    """
    Follows the official Qwen2.5-Omni pattern:
      1. build_inputs → apply_chat_template → formatted text string
      2. process_mm_info     → extract audio arrays from conversation dicts
      3. processor(text, audio, ...) → model inputs

    Every item covers the full ASR → LLM → TTS pipeline:
      chosen_request_audio_path, chosen_transcription,
      chosen_response_audio_path, chosen_tts_text
      (and the rejected_* equivalents)
    """
    chosen_texts, rejected_texts = [], []
    chosen_audios, rejected_audios = [], []

    for item in batch:
        chosen_in = build_inputs(
            processor,
            request_audio_path=item["chosen_request_audio_path"],
            transcription=item["chosen_transcription"],
            tts_audio_path=item["chosen_response_audio_path"],
            tts_text=item["chosen_tts_text"],
        )
        rejected_in = build_inputs(
            processor,
            request_audio_path=item["rejected_request_audio_path"],
            transcription=item["rejected_transcription"],
            tts_audio_path=item["rejected_response_audio_path"],
            tts_text=item["rejected_tts_text"],
        )

        chosen_texts.append(chosen_in["text"])
        rejected_texts.append(rejected_in["text"])

        # extend (not append) so the two-audio (request + response) conversation stays flat
        c_audios, _, _ = process_mm_info(chosen_in["conversation"], use_audio_in_video=False)
        r_audios, _, _ = process_mm_info(rejected_in["conversation"], use_audio_in_video=False)
        chosen_audios.extend(c_audios or [])
        rejected_audios.extend(r_audios or [])

    if not _printed[0]:
        print("\n[collate_fn] first sample — chosen:")
        print(chosen_texts[0])
        print("[collate_fn] first sample — rejected:")
        print(rejected_texts[0])
        _printed[0] = True

    inputs_chosen = processor(
        text=chosen_texts,
        audio=chosen_audios or None,
        return_tensors="pt",
        padding=True,
    )
    inputs_rejected = processor(
        text=rejected_texts,
        audio=rejected_audios or None,
        return_tensors="pt",
        padding=True,
    )

    return inputs_chosen, inputs_rejected


def evaluate(rm, eval_dataloader, accelerator):
    rm.eval()
    all_rewards_chosen = []
    all_rewards_rejected = []

    with torch.no_grad():
        for inputs_chosen, inputs_rejected in tqdm(
            eval_dataloader,
            disable=not accelerator.is_local_main_process,
            desc="Evaluating",
        ):
            rewards_chosen = rm(inputs_chosen)
            rewards_rejected = rm(inputs_rejected)

            rewards_chosen = accelerator.gather_for_metrics(rewards_chosen)
            rewards_rejected = accelerator.gather_for_metrics(rewards_rejected)

            all_rewards_chosen.append(rewards_chosen.cpu().float())
            all_rewards_rejected.append(rewards_rejected.cpu().float())

    rewards_chosen = torch.cat(all_rewards_chosen)
    rewards_rejected = torch.cat(all_rewards_rejected)
    margins = rewards_chosen - rewards_rejected

    metrics = {
        "eval/pairwise_accuracy": (margins > 0).float().mean().item(),
        "eval/bt_loss": -F.logsigmoid(margins).mean().item(),
        "eval/mean_margin": margins.mean().item(),
        "eval/mean_reward_chosen": rewards_chosen.mean().item(),
        "eval/mean_reward_rejected": rewards_rejected.mean().item(),
    }

    return metrics


def main(args):
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])

    model, processor = load_model(args.model_id, args.adapter_path, accelerator.local_process_index)

    model.config.use_cache = False

    # Optionally fine-tune the LM backbone via LoRA
    if args.train_lm:
        if isinstance(model, PeftModel):
            # An SFT adapter is already loaded (and active) — continue fine-tuning it
            # in place rather than stacking a second, separate adapter on top of it.
            # Its LoRA params get unfrozen in BradleyTerryRewardModel; --lora_rank/
            # --lora_alpha/--lora_dropout/--lora_target_modules are the SFT adapter's
            # own config and are not reapplied here.
            accelerator.print(
                "--train_lm: continuing to fine-tune the already-loaded SFT adapter in place "
                "(--lora_rank/--lora_alpha/--lora_dropout/--lora_target_modules are ignored — "
                "the SFT adapter's own LoRA config applies)."
            )
        else:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)

        # Gradient checkpointing only helps when the backbone actually backprops.
        # enable_input_require_grads() is required so gradients reach the LoRA
        # params through a checkpointed, otherwise-frozen backbone.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    rm = BradleyTerryRewardModel(model, train_lm=args.train_lm)

    # Optimizer covers the reward head and, when train_lm=True, the LoRA params too.
    # LoRA params use a separate (lower) lr to avoid disturbing pretrained representations.
    head_params = [p for n, p in rm.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_params  = [p for n, p in rm.named_parameters() if p.requires_grad and "lora_" in n]
    param_groups = [{"params": head_params}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lm_lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=0.01,
    )

    collate = partial(collate_fn, processor=processor)

    dataset = PairDataset(args.input_file, audio_dir=args.audio_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    eval_dataset = PairDataset(args.eval_file, audio_dir=args.audio_dir)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rm, optimizer, dataloader, eval_dataloader = accelerator.prepare(rm, optimizer, dataloader, eval_dataloader)

    # Scheduler must be created after prepare() so len(dataloader) reflects
    # the per-process shard size, giving an accurate total_steps count.
    total_steps = (len(dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    accelerator.init_trackers(
        project_name="qwen2-omni-bt-pipeline",
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "mixed_precision": "bf16",
            "train_lm": args.train_lm,
            "lora_rank": args.lora_rank if args.train_lm else None,
            "lora_alpha": args.lora_alpha if args.train_lm else None,
            "lora_target_modules": args.lora_target_modules if args.train_lm else None,
            "model_name": args.model_id,
            "num_epochs": args.num_epochs,
            "label_smoothing": args.label_smoothing,
        },
        init_kwargs={
            "wandb": {
                "entity": "amazon-project",
                "name": f"bt",
            }
        },
    )

    ema_loss = None
    ema_alpha = 0.9
    best_eval_accuracy = None

    for epoch in range(args.num_epochs):
        rm.train()

        progress_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        )

        for step, (inputs_chosen, inputs_rejected) in enumerate(progress_bar):
            with accelerator.accumulate(rm):
                rewards_chosen = rm(inputs_chosen)
                rewards_rejected = rm(inputs_rejected)

                reward_margin = rewards_chosen - rewards_rejected

                # Label-smoothed Bradley-Terry loss:
                #   target p(chosen > rejected) = 1 - label_smoothing  (instead of hard 1.0)
                #   loss = -(1-ε)·log σ(margin) - ε·log σ(-margin)
                bt_loss_per_sample = (
                    -(1 - args.label_smoothing) * F.logsigmoid(reward_margin)
                    - args.label_smoothing * F.logsigmoid(-reward_margin)
                )
                bt_loss = bt_loss_per_sample.mean()
                loss = bt_loss

                accelerator.backward(loss)
                # Clip / step the schedule only on real optimizer steps — under
                # gradient accumulation these must not fire on every micro-batch.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(rm.parameters(), max_norm=1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            margin = reward_margin.mean().item()
            ema_loss = loss.item() if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss.item()
            log_dict = {
                "train/loss": loss.item(),
                "train/loss_ema": ema_loss,
                "train/bt_loss": bt_loss.item(),
                "train/reward_chosen_mean": rewards_chosen.mean().item(),
                "train/reward_rejected_mean": rewards_rejected.mean().item(),
                "train/reward_margin_mean": reward_margin.mean().item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }

            accelerator.log(
                log_dict,
                step=epoch * len(dataloader) + step,
            )

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", margin=f"{margin:.4f}")

        # Save a checkpoint at the end of every epoch so a crash doesn't
        # lose all progress.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_ckpt = accelerator.unwrap_model(rm)
            ckpt_path = output_path / f"bt_reward_head_epoch{epoch + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(unwrapped_ckpt.head.state_dict(), ckpt_path)
            accelerator.print(f"Epoch {epoch + 1} checkpoint saved to {ckpt_path}")
            if args.train_lm:
                lm_ckpt_path = output_path / f"lm_lora_epoch{epoch + 1}"
                unwrapped_ckpt.lm.save_pretrained(lm_ckpt_path)
                accelerator.print(
                    f"Epoch {epoch + 1} LM LoRA checkpoint saved to {lm_ckpt_path} "
                    f"(adapter: {list(unwrapped_ckpt.lm.peft_config.keys())})"
                )

        eval_metrics = evaluate(rm, eval_dataloader, accelerator)
        accelerator.log(eval_metrics, step=(epoch + 1) * len(dataloader))
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch + 1} eval — " + ", ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items()))

            # Track the best checkpoint so far (highest eval pairwise accuracy) and mirror it into best/
            current_accuracy = eval_metrics["eval/pairwise_accuracy"]
            if best_eval_accuracy is None or current_accuracy > best_eval_accuracy:
                best_eval_accuracy = current_accuracy
                best_dir = output_path / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(unwrapped_ckpt.head.state_dict(), best_dir / "bt_reward_head.pt")
                if args.train_lm:
                    unwrapped_ckpt.lm.save_pretrained(best_dir / "lm_lora_adapter")
                    accelerator.print(
                        f"Best LM LoRA adapter saved to {best_dir / 'lm_lora_adapter'} "
                        f"(adapter: {list(unwrapped_ckpt.lm.peft_config.keys())})"
                    )
                with open(best_dir / "metrics.json", "w") as f:
                    json.dump({"epoch": epoch + 1, **eval_metrics}, f, indent=2)
                accelerator.print(
                    f"New best model (epoch {epoch + 1}, eval/pairwise_accuracy={current_accuracy:.4f}) saved to {best_dir}"
                )
        rm.train()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        best_str = f"{best_eval_accuracy:.4f}" if best_eval_accuracy is not None else "n/a"
        accelerator.print(f"Training complete. Best model (eval/pairwise_accuracy={best_str}) saved to {output_path / 'best'}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_file", required=True, help="jsonl file path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Id of the model")
    parser.add_argument("--audio_dir", default=None, help="Dir path of the audios")
    parser.add_argument("--eval_file", type=str, required=True, help="jsonl file for per-epoch evaluation")
    parser.add_argument("--output_path", required=True, help="Checkpoint output folder")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for BT loss (0 = standard, 0.1 = recommended)")
    parser.add_argument("--warmup_steps", type=int, default=25,
                        help="Linear LR warmup steps before cosine decay begins.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to a local SFT LoRA adapter. If omitted, the base model is "
                             "loaded from HuggingFace and only the reward head is trained.")
    parser.add_argument("--train_lm", action="store_true",
                        help="Fine-tune the LM backbone via LoRA in addition to the reward head.")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of module names to apply LoRA to.")
    parser.add_argument("--lm_lr", type=float, default=5e-6,
                        help="Learning rate for the LoRA adapter parameters (default: 5e-6).")

    args = parser.parse_args()
    main(args)