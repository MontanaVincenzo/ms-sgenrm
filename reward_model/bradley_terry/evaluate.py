from utils.reward_model import BradleyTerryRewardModel
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from utils.prepare_data import PairDataset, build_inputs
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import json
import sys
import yaml


def load_model(model_id):
    # 1. Load base model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    # 3. Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_id
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

        # extend (not append) so multi-audio conversations (e.g. overall) stay flat
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


def evaluate(rm, dataloader, device):
    """Mirrors train.py's evaluate(): pairwise accuracy and BT loss."""
    rm.eval()
    all_rewards_chosen, all_rewards_rejected = [], []

    with torch.no_grad():
        for inputs_chosen, inputs_rejected in tqdm(dataloader, desc="Evaluating"):
            inputs_chosen = {k: v.to(device) for k, v in inputs_chosen.items()}
            inputs_rejected = {k: v.to(device) for k, v in inputs_rejected.items()}

            rewards_chosen = rm(inputs_chosen)
            rewards_rejected = rm(inputs_rejected)

            all_rewards_chosen.append(rewards_chosen.cpu().float())
            all_rewards_rejected.append(rewards_rejected.cpu().float())

    rewards_chosen = torch.cat(all_rewards_chosen)
    rewards_rejected = torch.cat(all_rewards_rejected)
    margins = rewards_chosen - rewards_rejected

    accuracy = (margins > 0).float().mean().item()
    bt_loss = -F.logsigmoid(margins).mean().item()

    summary = {
        "n_pairs": len(rewards_chosen),
        "pairwise_accuracy": round(accuracy, 4),
        "bt_loss": round(bt_loss, 4),
        "mean_margin": round(margins.mean().item(), 4),
        "std_margin": round(margins.std().item(), 4),
        "mean_reward_chosen": round(rewards_chosen.mean().item(), 4),
        "mean_reward_rejected": round(rewards_rejected.mean().item(), 4),
    }

    return summary, rewards_chosen, rewards_rejected, margins


def main(args):
    device_idx = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_idx}" if isinstance(device_idx, int) else device_idx)

    print(f"Loading model {args.model_id} on {device} ...")
    model, processor = load_model(args.model_id)
    model.config.use_cache = False
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    rm = BradleyTerryRewardModel(model)

    print(f"Loading reward head from {args.reward_head_path} ...")
    state_dict = torch.load(args.reward_head_path, map_location=device)
    rm.head.load_state_dict(state_dict)
    rm = rm.to(device)
    rm.eval()

    dataset = PairDataset(args.input_file, audio_dir=args.audio_dir)
    print(f"Loaded {len(dataset)} test pairs from {args.input_file}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=args.num_workers,
    )

    summary, rewards_chosen, rewards_rejected, margins = evaluate(rm, dataloader, device)

    print("\n=== Evaluation Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output_file:
        output = {
            "args": vars(args),
            "summary": summary,
            "pairs": [
                {
                    "reward_chosen": round(rc.item(), 6),
                    "reward_rejected": round(rr.item(), 6),
                    "margin": round(m.item(), 6),
                    "correct": bool((m > 0).item()),
                }
                for rc, rr, m in zip(rewards_chosen, rewards_rejected, margins)
            ],
        }
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)  # default=str: args holds Path objects
        print(f"Per-pair results saved to {args.output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--split", choices=["train", "eval", "val"], default="val",
                        help="Which split from data_generation.split to evaluate on (default: the held-out val split)")
    parser.add_argument("--reward_head_path", default=None,
                        help="Reward head .pt to evaluate (default: <bradley_terry.output_path>/best/bt_reward_head.pt)")
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
    required = ["model_id", "output_path", "eval_output_file", "batch_size", "num_workers"]
    missing = [k for k in required if k not in bt_config]
    if missing:
        print(f"Error: missing keys in 'bradley_terry' section of {args.config_path}: {missing}", file=sys.stderr)
        sys.exit(1)

    # Relative paths are relative to the repo root (the folder containing config.yaml), as in train.py.
    repo_root = Path(args.config_path).resolve().parent
    data_dir = repo_root / config["data_generation"]["data_dir"]
    args.model_id = bt_config["model_id"]
    args.batch_size = bt_config["batch_size"]
    args.num_workers = bt_config["num_workers"]
    args.input_file = data_dir / config["data_generation"]["split"][f"{args.split}_file"]
    args.audio_dir = repo_root
    args.reward_head_path = (Path(args.reward_head_path) if args.reward_head_path
                             else repo_root / bt_config["output_path"] / "best" / "bt_reward_head.pt")
    args.output_file = repo_root / bt_config["eval_output_file"]
    main(args)
