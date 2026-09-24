# Joint Bradley-Terry Reward Model — Architecture

## 1. Overview

This pipeline trains a **joint multi-task reward model** that scores three speech quality dimensions simultaneously: ASR transcription quality, TTS naturalness, and overall spoken-dialogue quality. A single Qwen2.5-Omni-7B backbone is shared across all three tasks, with three independent scalar reward heads trained via Bradley-Terry preference learning.

---

## 2. Backbone

The backbone is **Qwen2.5-Omni-7B** (`Qwen2_5OmniThinkerForConditionalGeneration`), a 7B-parameter multimodal transformer that natively processes both audio and text. Audio is decoded by a Whisper-style audio tower into embeddings that are concatenated with text token embeddings before entering the language model layers.

**Input formatting:**

Audio is extracted from structured conversation dicts via `qwen_omni_utils.process_mm_info()` and tokenized together with text via `Qwen2_5OmniProcessor`. Conversations are formatted with `apply_chat_template(add_generation_prompt=False)` — the full assistant turn is included and no generation prompt is appended, so the model sees the candidate output directly.

**Conversation structure per task** (built by `build_conversation()` in [code/utils/prepare_data.py](code/utils/prepare_data.py)):

| Role | ASR | TTS |
|---|---|---|
| System | *"You are an ASR system. Your task is to transcribe the spoken audio provided by the user as accurately as possible."* | *"You are a TTS system. Your task is to generate natural, expressive speech for the text provided by the user."* |
| User | `[audio]` | `[target text]` |
| Assistant | `[transcription]` | `[generated audio]` |

The **overall** task uses a three-turn multi-turn structure under a single system prompt (*"You are an end-to-end spoken dialogue assistant. You receive spoken audio requests and respond by: (1) transcribing the spoken request accurately, (2) generating a relevant and helpful textual response, (3) synthesizing that response as natural, expressive speech."*):

| Turn | Role | Content |
|---|---|---|
| 1 | User | *"Please transcribe the following spoken request."* + `[request audio]` |
| 1 | Assistant | `[transcription]` |
| 2 | User | *"Based on the transcribed request, provide a helpful and concise textual response."* |
| 2 | Assistant | `[response text]` |
| 3 | User | *"Now synthesize your textual response as natural spoken audio."* |
| 3 | Assistant | `[response audio]` |

The system prompt encodes the task identity; each turn models one sub-step of the spoken dialogue pipeline.

**Optional SFT / LM adapter:**

A LoRA SFT adapter (e.g. from SpeechJudge fine-tuning) can be loaded on top of the base model via `--adapter_path`. Its weights are frozen by default.

When `--train_lm` is set in `train.py`:
- If an SFT adapter was already loaded via `--adapter_path`, its existing LoRA parameters are simply **unfrozen and fine-tuned in place** (the same adapter continues training — a second adapter is *not* stacked on top of it). `--lora_rank`/`--lora_alpha`/`--lora_dropout`/`--lora_target_modules` are ignored in this case, since the SFT adapter's own config already applies.
- If no adapter was loaded (training from the base model), a fresh LoRA adapter is created via `LoraConfig`/`get_peft_model` using `--lora_rank` (default 16), `--lora_alpha` (default 32), `--lora_dropout` (default 0.05), and `--lora_target_modules` (default `q_proj,k_proj,v_proj,o_proj`).

Either way, the LoRA parameters are trained jointly with the reward heads at a lower learning rate (`--lm_lr`, default 5e-6), and saved per epoch to `lm_lora/` (see §7).

At evaluation time, `evaluate.py` reloads a previously-trained LoRA checkpoint (the `lm_lora_adapter/` directory saved by `train.py`) via `--lm_lora_path`, registering it under the adapter name **`reward_lm`** (`model.load_adapter(lm_lora_path, adapter_name="reward_lm")`). This adapter name is an `evaluate.py`-side loading detail — `train.py` does not create or reference an adapter named `reward_lm`.

Optionally, fine-tuned audio tower weights can be restored from `{adapter_path}/audio_tower_finetuned.pt` via `--load_audio_tower`, loaded with `strict=False` into `model.base_model.model.audio_tower` (missing/unexpected keys are printed as warnings, not raised).

---

## 3. Reward Heads

Implemented in `BradleyTerryRewardModel` ([code/utils/reward_model.py](code/utils/reward_model.py)).

**Reward extraction (default configuration):** after the backbone forward pass (frozen unless `--train_lm` unfreezes its LoRA parameters), the last-layer hidden state at position `[-2]` (the final real token before padding) is projected through the task head:

```
h = outputs.hidden_states[-1][:, -2, :]   # (B, hidden_size)
```

Position `-2` is used rather than `-1` because the sequence ends with the [`<|im_end|>`, `\n`] tokens; `<|im_end|>` is the token that usually carries the richest contextual representation of the candidate output. This applies to `asr_head` and `tts_head` always, and to `overall_head` when the model is constructed with the default `multi_token=False` (the only mode `train.py` and `evaluate.py` actually use — neither script passes `multi_token`).

Each task's inputs are forwarded through the LM **independently** — `forward()` makes a separate `self.lm(...)` call per non-`None` argument (`inputs_asr`, `inputs_tts`, `inputs_overall`). The three heads share the same backbone *weights*, not a shared forward pass.

**Optional `multi_token` mode (constructor arg, unused by `train.py`/`evaluate.py`):** if `BradleyTerryRewardModel` is constructed with `multi_token=True`, `overall_head` instead becomes `Linear(3 * hidden_size, 1, bias=False)` and pools by masking on `input_ids == im_end_id` — extracting the hidden state at every `<|im_end|>` position (one per assistant turn in the 3-turn "overall" conversation) and reshaping to `(batch, 3 * hidden_size)` before the head. Note this code path references a module-level `tokenizer` object (`self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")`) that is never defined or imported in `reward_model.py`, so constructing the model with `multi_token=True` currently raises a `NameError`.

### 3.1 Scalar heads (one per task)

| Head | Output | Range |
|---|---|---|
| `asr_head` | `Linear(3584, 1, bias=False)` | unbounded |
| `tts_head` | `Linear(3584, 1, bias=False)` | unbounded |
| `overall_head` | `Linear(3584, 1, bias=False)` (default `multi_token=False`; `Linear(10752, 1, bias=False)` if `multi_token=True`) | unbounded |

`3584` is `base_lm.config.text_config.hidden_size` for Qwen2.5-Omni-7B's text backbone.

### 3.2 Multi-variate TTS regression head

An additional `multi_head = Linear(3584, 5, bias=False)` is attached whenever `num_scores` is truthy (both `train.py` and `evaluate.py` always construct the model with `num_scores=5`); it is only evaluated on the TTS branch. Its output is sigmoid-scaled to the annotation range (1, 5):

$$\hat{s} = 1 + 4 \cdot \sigma\!\left(\text{multi\_head}(h)\right) \in (1, 5)^5$$

The five dimensions correspond to human annotation scores, in this exact order: **pitch**, **intensity**, **naturalness**, **prosody_emotion**, **timing_pacing** (matching the `tts_pitch`/`tts_intensity`/`tts_naturalness`/`tts_prosody_emotion`/`tts_timing_pacing` TSV columns read in `PairDataset`).

**Trainable parameter count (default, `multi_token=False`, `num_scores=5`):**

| Component | Parameters |
|---|---|
| `asr_head` | 3,584 |
| `tts_head` | 3,584 |
| `overall_head` | 3,584 |
| `multi_head` | 17,920 |
| LoRA backbone (optional, `--train_lm`) | ~millions |
| **Total (heads only)** | **28,672** |

(If `multi_token=True` were used, `overall_head` alone would grow to 10,752 params, for a heads-only total of 35,840 — but this mode is not exercised by either training or evaluation script.)

---

## 4. Loss Functions

### 4.1 Label-smoothed Bradley-Terry loss

For each task, given a preference pair with predicted rewards $(r_c, r_r)$, the margin is $\delta = r_c - r_r$ and the per-sample BT loss with label smoothing $\varepsilon$ is:

$$\mathcal{L}_{\text{BT}}^{(i)} = -(1-\varepsilon)\log\sigma(\delta^{(i)}) - \varepsilon\log\sigma(-\delta^{(i)})$$

- $\varepsilon = 0$: hard Bradley-Terry.
- $\varepsilon > 0$: soft targets that prevent overconfidence. Default: $\varepsilon = 0.1$ (`--label_smoothing`).

### 4.2 Metric-weighted BT loss (currently inert for all tasks)

`compute_task_loss()` in `train.py` is generic — it is called for **all three tasks** (asr, tts, overall), and re-weights the BT loss whenever `metrics_chosen` is a `torch.Tensor` rather than the empty-list placeholder:

$$\mathcal{L}_{\text{BT}}^{w} = \text{mean}\!\left[(1 + \lambda_m \cdot g^{(i)}) \cdot \mathcal{L}_{\text{BT}}^{(i)}\right], \qquad \lambda_m = \texttt{--metrics\_weight}\ (\text{default } 2.0)$$

where $g^{(i)}$ is the clamped metric gap:
- **task == "asr":** $g = \text{clip}(\text{WER}_r - \text{WER}_c,\ 0, 1)$
- **any other task (tts, overall):** $g = \text{clip}(m_c - m_r,\ 1, 4)$, where $m$ is whatever value `PairDataset` supplied as the per-item metric (UTMOS for tts, `goodness_score` for overall, by intent)

A margin regression loss anchors the predicted margin to the metric-implied margin:

$$\mathcal{L}_{\text{margin}} = \text{MSE}(\delta,\ t), \qquad \lambda_{\text{margin}} = \texttt{--margin\_loss\_weight}\ (\text{default } 0.1)$$

where the target $t$ differs by task:
- **asr:** $t = s \cdot g$, with $s$ = `--margin_scale` (default 5.0)
- **any other task:** $t = g$ (no scale factor applied)

giving:

$$\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{BT}}^{w} + \lambda_{\text{margin}} \cdot \mathcal{L}_{\text{margin}}$$

**This path is currently unreachable in practice.** `PairDataset.__init__` ([code/utils/prepare_data.py](code/utils/prepare_data.py)) computes real metrics per task (WER for asr, `goodness_score` for overall) but then unconditionally executes `metrics_chosen, metrics_rejected = None, None` immediately after the per-task `if/elif/else` block, on every group, overwriting whatever was just computed. TTS's UTMOS metric computation is additionally commented out at the source, so it was never populated to begin with. As a result, `collate_fn` always receives `None` metrics, `metrics_chosen`/`metrics_rejected` become empty lists (not tensors), and `compute_task_loss` always takes the `else` branch — plain label-smoothed BT loss, for asr, tts, and overall alike. The metric-weighted/margin-loss machinery above describes what the code does *if* metrics were present, not current runtime behavior.

### 4.3 TTS multi-variate regression loss

When ground-truth annotation scores are available (they always are for the `tts` task pairs, via the `scores_chosen`/`scores_rejected` TSV-derived fields), the multi_head is supervised with an L2 loss over all five dimensions simultaneously:

$$\mathcal{L}_{\text{multi}} = \frac{1}{32}\left(\mathbb{E}\!\left[\|\hat{s}_c - s_c\|^2\right] + \mathbb{E}\!\left[\|\hat{s}_r - s_r\|^2\right]\right)$$

where each expectation is `torch.mean(torch.sum((predictions - targets)^2, dim=-1))` (`compute_mse_loss`). The factor $\frac{1}{32}$ scales the loss before it is added into the combined training loss (see §5), weighted by `--lambda_tts_multi`.

---

## 5. Loss Balancing Across Tasks (`--loss_balancing`)

`train.py` does **not** implement homoscedastic/uncertainty-based loss weighting — there is no learned log-variance parameter, no `UncertaintyWeights` module, and no `uncertainty_weights.pt` checkpoint anywhere in the codebase. Instead, `--loss_balancing` (choices `manual` | `pcgrad`, default `manual`) selects between two strategies for combining the three per-task BT losses (§4) and the TTS regression loss (§4.3):

### 5.1 `manual` (default) — fixed weighted sum

$$\mathcal{L} = \lambda_{\text{asr}} \mathcal{L}_{\text{task,asr}} + \lambda_{\text{tts}} \mathcal{L}_{\text{task,tts}} + \lambda_{\text{overall}} \mathcal{L}_{\text{task,overall}} + \lambda_{\text{multi}} \cdot \mathcal{L}_{\text{multi}}$$

All four weights are fixed CLI floats, each defaulting to 1.0: `--lambda_asr`, `--lambda_tts`, `--lambda_overall`, `--lambda_tts_multi`. A single `accelerator.backward(loss)` call backpropagates the combined loss.

### 5.2 `pcgrad` — Project Conflicting Gradients (Yu et al., 2020)

Each task (asr, tts, overall) gets its own forward pass and its own isolated `accelerator.backward()` call. On the parameters shared across tasks (the LoRA backbone — only populated when `--train_lm` is set), gradients are combined via PCGrad: for each task's flattened gradient, the component conflicting (negative cosine similarity) with each other task's gradient is projected away, in random order, then the three (possibly-projected) gradients are summed (`pcgrad_project()`). On the private per-task head parameters (`asr_head`, `tts_head`, `overall_head`, `multi_head`), gradients are simply accumulated per task with no projection, since they never overlap across tasks. `--lambda_tts_multi` still scales the TTS regression loss, which is folded into the `tts` task's own backward pass (so it participates in PCGrad's projection). `pcgrad` requires `--gradient_accumulation_steps 1`; `train.py` raises a `ValueError` at startup otherwise, since PCGrad already performs one full per-task backward pass per optimizer step.

### 5.3 Gradient concordance diagnostic (`--log_grad_concordance`)

Independent of `--loss_balancing`, `--log_grad_concordance` (with `--grad_concordance_every_n_steps`, default 50) periodically does an isolated backward pass per head (asr/tts/overall/tts_multi) and logs the pairwise cosine similarity of their gradients on the shared LoRA backbone to W&B as `grad_concordance/*` (§9). It requires `--train_lm` (there is no shared trainable parameter to compare gradients on otherwise); `train.py` disables it with a warning if requested without `--train_lm`.

---

## 6. Training Setup

### Optimizer and learning rates

**AdamW** (weight decay 0.01) with up to two parameter groups:

| Group | Parameters | LR |
|---|---|---|
| Reward heads | all non-LoRA trainable parameters — `asr_head`, `tts_head`, `overall_head`, `multi_head` | `--lr` (default 1e-4) |
| LoRA backbone (only when `--train_lm` produces trainable `lora_*` params) | `lora_*` parameters | `--lm_lr` (default 5e-6) |

**Cosine schedule with linear warmup** (`get_cosine_schedule_with_warmup`), `--warmup_steps` warmup steps (default 25), decay to 0.

### Data pipeline

Three independent `PairDataset` instances are created — one per task — each yielding `(chosen, rejected)` pairs from TSV annotations. The dataloaders are zipped and iterated together; training stops when the shortest dataloader is exhausted.

Per-GPU, per-task batch size = `--train_batch_size // 3`, so each step processes equal numbers of ASR, TTS, and overall pairs. Evaluation dataloaders use `--eval_batch_size` directly per task (not divided by 3).

The TTS dataloader additionally provides five-dimensional annotation scores per pair (`scores_chosen`, `scores_rejected`) for the regression head.

### Distributed training

Multi-GPU data-parallel training via Accelerate — process count is controlled by the `accelerate launch --num_processes N` invocation, not by a `train.py` CLI flag. `DistributedDataParallelKwargs(find_unused_parameters=True)` is required because `multi_head` is only activated on TTS inputs — ASR and overall forward passes do not use it, which would cause DDP to error without this flag.

### Batch and precision

| Parameter | Value |
|---|---|
| Total train batch size | `--train_batch_size` (default 12) |
| Per-task per-GPU | `train_batch_size // 3` |
| Eval batch size (per task) | `--eval_batch_size` (default 12) |
| Gradient accumulation | `--gradient_accumulation_steps` (default 1; must be 1 when `--loss_balancing pcgrad`) |
| Gradient clipping | max norm 1.0 |
| Mixed precision | bfloat16 |

---

## 7. Checkpointing

After every epoch the main process saves to `{output_path}/epoch{N}/`. Additionally, whenever `eval/overall/pairwise_accuracy` improves on the held-out set, the same files are mirrored to `{output_path}/best/` alongside a `metrics.json`. There is no separate unconditional "final" save at the end of training beyond the last epoch's `epoch{N}/` directory.

| File | Location | Contents |
|---|---|---|
| `asr_head.pt` | `epoch{N}/`, `best/` | ASR scalar head state dict |
| `tts_head.pt` | `epoch{N}/`, `best/` | TTS scalar head state dict |
| `overall_head.pt` | `epoch{N}/`, `best/` | Overall scalar head state dict |
| `tts_multi_head.pt` | `epoch{N}/`, `best/` (if `multi_head` exists) | TTS regression head state dict |
| `lm_lora/` | `epoch{N}/lm_lora/` (when `--train_lm`) | LoRA backbone adapter, saved every epoch via `unwrapped.lm.save_pretrained(...)` |
| `lm_lora_adapter/` | `best/lm_lora_adapter/` (when `--train_lm` and this epoch is a new best) | LoRA backbone adapter, best-epoch copy |
| `metrics.json` | `best/` only | `{"epoch": N, **eval_metrics}` for the best checkpoint |

No `uncertainty_weights.pt` is ever written — that file, and the loss-balancing mechanism it would have stored, does not exist in the current code (see §5).

---

## 8. Evaluation

### Per-epoch evaluation (during `train.py`)

After each epoch, `evaluate()` computes the following on held-out test sets for each task independently:

| Metric | Description |
|---|---|
| `eval/{task}/pairwise_accuracy` | Fraction of pairs where $r_c > r_r$ |
| `eval/{task}/bt_loss` | $-\log\sigma(\delta)$ on test pairs (plain, unweighted — metric-weighted loss is never used at eval time) |
| `eval/{task}/mean_margin` | Mean $r_c - r_r$ |
| `eval/{task}/mean_reward_chosen` | Mean $r_c$ |
| `eval/{task}/mean_reward_rejected` | Mean $r_r$ |
| `eval/tts/mse_loss_chosen`, `eval/tts/mse_loss_rejected` | TTS multi-score regression MSE (only populated for the `tts` task, since only TTS inputs produce `multi_score` in `forward()`) |

The best checkpoint is selected purely by `eval/overall/pairwise_accuracy` (see §7).

### Standalone evaluation script (`code/evaluate.py`)

A separate script for offline evaluation of a saved checkpoint. It loads the base model plus `--adapter_path` (SFT adapter) and optional `--load_audio_tower` / `--lm_lora_path` (reloaded as adapter `reward_lm`, see §2), then loads reward heads from `--reward_head_dir` (`asr_head.pt`, `tts_head.pt`, `overall_head.pt`, and `tts_multi_head.pt` if present — printing a warning and leaving a head randomly initialized if its file is missing). It runs whichever of `--input_file_asr` / `--input_file_tts` / `--input_file_overall` are supplied (each optional; default `None` skips that task) at `--batch_size` (default 4), and prints/optionally writes (`--output_file`) a JSON summary per task (`pairwise_accuracy`, `bt_loss`, `mean_margin`, `std_margin`, `mean_reward_chosen`, `mean_reward_rejected`, and `mse_loss_chosen`/`mse_loss_rejected`/`mse_loss` when TTS scores are available), plus per-pair rewards keyed by `user_id`/`dialogue_id`.

---

## 9. W&B Logging (per step, from `train.py`)

| Key | Description |
|---|---|
| `train/loss` | Combined loss for the step (manual: weighted sum per §5.1; pcgrad: sum of unweighted per-task losses) |
| `train/loss_ema` | EMA of total loss (α = 0.9) |
| `train/bt_loss_{asr,tts,overall}` | Per-task BT loss |
| `train/reward_margin_{asr,tts,overall}` | Mean margin per task |
| `train/weight_{asr,tts,overall}` | Only logged in `manual` mode — the fixed `--lambda_{asr,tts,overall}` value (pcgrad has no fixed weight for these three; PCGrad combines their gradients directly) |
| `train/weight_tts_multi` | Always logged — the fixed `--lambda_tts_multi` value |
| `train/mse_loss_tts_{chosen,rejected}` | TTS regression loss per side |
| `train/tts_multi_{chosen,rejected}_{dim}` | Per-dimension mean predicted scores; `{dim}` ∈ `pitch`, `intensity`, `naturalness`, `prosody_emotion`, `timing_pacing` |
| `train/margin_loss_{asr,tts}` | Metric margin loss — key exists in the code but is never populated under the current `PairDataset` (see §4.2) |
| `train/metrics_gap_{asr,tts}` | Mean metric gap — same caveat as above |
| `train/learning_rate` | Current LR |
| `grad_concordance/{head_a}_vs_{head_b}`, `grad_concordance/mean` | Pairwise cosine similarity between isolated per-head gradients on the shared LoRA backbone; only logged when `--log_grad_concordance` is set (see §5.3) |
