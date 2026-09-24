import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

class BradleyTerryRewardModel(nn.Module):
    """
    Scalar reward model for Bradley-Terry preference learning on top of Qwen2.5-Omni.

    The full LM (thinker + audio/vision encoders) is frozen by default.
    Only the linear reward head is trained.

    Usage (pairwise BT loss):
        rewards_chosen   = model(inputs_chosen)    # (batch,)
        rewards_rejected = model(inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """

    def __init__(self, base_lm: Qwen2_5OmniThinkerForConditionalGeneration, train_lm: bool = False):
        super().__init__()
        self.lm = base_lm
        self.train_lm = train_lm

        hidden_size = base_lm.config.text_config.hidden_size

        self.head = nn.Linear(hidden_size, 1, bias=False)

        # Freeze the entire base model
        for param in self.lm.parameters():
            param.requires_grad = False

        # When fine-tuning the LM via LoRA, unfreeze only the LoRA parameters
        if train_lm:
            for name, param in self.lm.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        self.head = self.head.to(dtype=torch.bfloat16)

        # We only ever read outputs.hidden_states[-1] at a single pooled
        # position; the model's own lm_head still runs on every forward call
        # and projects every token to a (batch, seq_len, vocab_size) logits
        # tensor we immediately discard. With Omni's ~156k vocab this is the
        # single largest activation in the whole forward pass, and its size
        # scales with the (highly variable, audio-driven) sequence length —
        # that's what was causing OOMs once a shuffled batch hit long audio.
        # hidden_states is produced by the inner transformer independently
        # of lm_head, so replacing it with a no-op is safe here.
        lm_head_owner = self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm
        lm_head_owner.lm_head = nn.Identity()

    def _pool_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Per sequence, return the index of the second-to-last real (non-padding)
        token. The chat template ends every assistant turn with the two tokens
        ``<|im_end|>`` and ``\\n``; we want the ``<|im_end|>`` position, which
        carries the richest summary of the candidate output.

        Works for both left- and right-padded batches (the attention mask is the
        single source of truth), so it does not rely on any assumption about the
        processor's padding side.

        Args:
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            (batch,) long tensor of column indices
        """
        batch, seq_len = attention_mask.shape
        lengths = attention_mask.long().sum(dim=1)               # (batch,)
        if attention_mask[:, 0].min() == 0:
            # left padding: real tokens occupy the tail of every row
            last_real = torch.full((batch,), seq_len - 1, device=attention_mask.device)
        else:
            # right padding: real tokens occupy the head of each row
            last_real = lengths - 1
        target = (last_real - 1).clamp(min=0)                     # the <|im_end|> token
        return target

    def forward(self, inputs: dict) -> torch.Tensor:
        if self.train_lm:
            outputs = self.lm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.lm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden = outputs.hidden_states[-1]  # (B, T, hidden_size)

        idx = self._pool_indices(inputs["attention_mask"])        # (B,)
        idx = idx.to(hidden.device)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        seq_repr = hidden[batch_idx, idx]                         # (B, hidden_size)

        seq_repr = seq_repr.to(dtype=self.head.weight.dtype)
        rewards = self.head(seq_repr).squeeze(-1).float()         # (B,)

        return rewards
    
    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_lm:
            # Keep the LM frozen in eval mode to avoid BatchNorm/dropout side-effects
            self.lm.eval()
        return self

if __name__ == "__main__":
    base_model_id = "/leonardo_work/IscrC_MSMU/models/Qwen2.5-Omni-7B"
    adapter_path = "/leonardo_work/IscrC_MSMU/vmontana/amazon/speech_judge_sft/code/speechjudge_qwen_omni_thinker_lora"
    processor = Qwen2_5OmniProcessor.from_pretrained(
        adapter_path,
        local_files_only=True,
    )
    base_lm = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    base_lm = PeftModel.from_pretrained(
        base_lm,
        adapter_path,
    )

    print(base_lm.config.text_config.hidden_size)

    rm = BradleyTerryRewardModel(base_lm=base_lm)
    
    print("Reward model created successfully.")
    trainable = sum(p.numel() for p in rm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in rm.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)") 