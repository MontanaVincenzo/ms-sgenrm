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
import yaml
import sys

def load_model(model_id, device):

    # 1. Load base model directly onto this process's GPU
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.config.use_cache = False


    # 3. Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

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

    model, processor = load_model(args.model_id, accelerator.local_process_index)

    model.config.use_cache = False

    rm = BradleyTerryRewardModel(model)

    head_params = [p for n, p in rm.named_parameters() if p.requires_grad]
    param_groups = [{"params": head_params}]

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

    bt_config = config["bradley_terry"]
    required = ["seed", "model_id", "output_path", "num_workers", "num_epochs",
                "batch_size", "gradient_accumulation_steps", "lr", "label_smoothing", "warmup_steps"]
    missing = [k for k in required if k not in bt_config]
    if missing:
        print(f"Error: missing keys in 'bradley_terry' section of {args.config_path}: {missing}", file=sys.stderr)
        sys.exit(1)
    for key in required:
        setattr(args, key, bt_config[key])
    # Relative paths (data_dir, and the audio paths stored in the jsonl, e.g. "data/input_audios_original/x.wav")
    # are relative to the repo root, i.e. the folder containing config.yaml -- not to data_dir.
    repo_root = Path(args.config_path).resolve().parent
    data_dir = repo_root / config["data_generation"]["data_dir"]
    setattr(args, "input_file", data_dir / config["data_generation"]["split"]["train_file"])
    setattr(args, "eval_file", data_dir / config["data_generation"]["split"]["eval_file"])
    setattr(args, "audio_dir", repo_root)

    args.lr = float(args.lr)
    args.label_smoothing = float(args.label_smoothing)
    main(args)