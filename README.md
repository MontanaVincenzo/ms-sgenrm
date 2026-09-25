# ms-sgenrm

Synthetic data generation and reward-model training for spoken ASR → LLM → TTS pipelines.

The repo has two parts:

1. **`data_generation/`** builds a dataset of spoken requests. Each request is answered by two pipelines (`pipeline1`, `pipeline2`), each with its own ASR, LLM and TTS perturbations, and every pipeline is scored by an LLM judge.
2. **`reward_model/`** trains reward models on that dataset, based on Qwen2.5-Omni-7B:
   - `bradley_terry/`: a pairwise Bradley–Terry reward head trained on the frozen model.
   - `generative_rm/`: a LoRA fine-tune (SFT) that writes the judge's evaluation JSON.

All scripts read their settings from **`config.yaml`** at the repo root.

## Setup

There are three virtual environments, because the packages conflict (different `vllm` and `torch` versions). Build them with [uv](https://docs.astral.sh/uv/):

```bash
# Orpheus-TTS must be cloned before building the `orpheus` env
git clone https://github.com/canopyai/Orpheus-TTS.git data_generation/01_input_request/02_tts_synthesis/Orpheus-TTS

./setup_envs.sh                     # all three
./setup_envs.sh training            # or just some of them
FORCE=1 ./setup_envs.sh orpheus     # rebuild an existing one
```

| Env | Location | Python | Used by |
|---|---|---|---|
| `data_generation` | `data_generation/.venv` | 3.12 | data-generation stages 01 (text), 02–06 |
| `orpheus` | `data_generation/01_input_request/02_tts_synthesis/.venv` | 3.11 | stage 01 speech synthesis (Orpheus-TTS) |
| `training` | `reward_model/.venv` | 3.12 | everything in `reward_model/` |

Notes:
- The `data_generation` env installs `vllm-omni` from a local copy (it carries local patches). Point `VLLM_OMNI_SRC=/path/to/vllm-omni` at yours if it isn't at the default path in `setup_envs.sh`.
- Requirements are exact freezes (`data_generation/*_requirements.txt`, `reward_model/training_requirements.txt`).

## Configuration

`config.yaml` has one section per part of the code:

| Section | What it controls |
|---|---|
| `data_generation` | models (`asr_model`, `llm_model`, `tts_model`), `data_dir`, whether inputs are perturbed, and the file name each stage writes |
| `data_generation.split` | train/eval/val fractions (80/15/5 by default), split seed and file names |
| `bradley_terry` | Bradley–Terry training and evaluation hyperparameters, checkpoint and results paths |
| `generative_rm` | SFT hyperparameters, level (A/B/C), LoRA settings, wandb, vLLM adapter folder |
| `generative_rm.eval` | split to evaluate on, results folder, vLLM engine settings |

**Paths:** relative paths in `config.yaml`, and the audio paths stored in the dataset (e.g. `data/input_audios_original/28_dan.wav`), are relative to the repo root, which is the folder containing `config.yaml`. Every script resolves them from there, so you can launch from any folder. The data-generation stages switch into the repo root when they start. Paths you pass on the command line (such as `--votes` or `--src`) are relative to where you launch.

## Data generation

Run the stages in order from the repo root. Each stage reads the previous stage's file in `data_dir` and writes its own:

| # | Stage | Command | Output |
|---|---|---|---|
| 01a | Generate text requests from `data/topics/topics.jsonl` | `data_generation/.venv/bin/python data_generation/01_input_request/01_text_generation/textual_input_request.py --config_path config.yaml --num_samples N` | `dataset.jsonl` |
| 01b | Speak the requests with Orpheus-TTS | `data_generation/01_input_request/02_tts_synthesis/.venv/bin/python data_generation/01_input_request/02_tts_synthesis/tts.py --config_path config.yaml` | `input_audios_*/` |
| 02 | ASR: transcribe, with optional perturbations (silence masking, hallucination) | `data_generation/.venv/bin/python data_generation/02_asr/asr_stage.py --config_path config.yaml` | `dataset_asr.jsonl` |
| 03 | LLM: answer the transcription, with optional perturbations | `data_generation/.venv/bin/python data_generation/03_llm/llm_stage.py --config_path config.yaml` | `dataset_llm.jsonl` |
| 04 | TTS: speak the answer with Qwen3-TTS, with optional perturbations (wrong emotion, added/missing words) | `data_generation/.venv/bin/python data_generation/04_tts/tts_stage.py --config_path config.yaml` | `dataset_tts.jsonl`, `output_audio/` |
| 05 | LLM judge: score each pipeline end to end with `--votes` samples and a majority vote | `data_generation/.venv/bin/python data_generation/05_llm_evaluation/evaluate.py --config_path config.yaml` | `dataset_eval.jsonl`, `dataset_eval_votes.jsonl` |
| 06 | Split into train/eval/val | `data_generation/.venv/bin/python data_generation/06_split_data/split_data.py --config_path config.yaml` | `train_data.jsonl`, `eval_data.jsonl`, `val_data.jsonl` |

Useful extras:
- **Quick test runs:** stages 03–06 accept `--limit N` to process only the first N samples.
- **ASR wrapper:** `data_generation/02_asr/run.sh` runs stage 02 with the vLLM environment variables this machine needs. Extra flags are passed through, and `CONFIG=...` selects another config.
- **Judge settings:** stage 05 also takes `--votes` (default 10) and `--temperature` (default 0.7).
- **Judge on two GPUs:** `data_generation/05_llm_evaluation/run_eval_2gpu.sh` splits the dataset in half and judges each half on its own GPU. Its file paths come from the config.
- **Judge consistency:** `data_generation/.venv/bin/python data_generation/05_llm_evaluation/agreement.py --config_path config.yaml` reports how consistent the judge is with itself. It writes to `data_generation.agreement_output_file`.
- **Topic separation:** stage 06 keeps each topic in a single split, so the achieved fractions can differ slightly from the targets (it prints them). Records with no judge score or with missing audio are dropped; `--keep-unevaluated` keeps the unscored ones.
- **Inspecting data:** `visualize/` folders hold notebooks for browsing each stage's output, and an HTML viewer for the judge's verdicts.

## Reward models

All commands use the `training` env and run from the repo root.

### Bradley–Terry

```bash
# train: data/train_data.jsonl, evaluated on data/eval_data.jsonl after every epoch
reward_model/.venv/bin/python reward_model/bradley_terry/train.py --config_path config.yaml

# evaluate the best checkpoint on the held-out val split
reward_model/.venv/bin/python reward_model/bradley_terry/evaluate.py --config_path config.yaml
#   --split train|eval|val        evaluate on another split
#   --reward_head_path <file.pt>  evaluate a specific epoch's checkpoint
```

- **Pairs:** each record becomes a (chosen, rejected) pair from the two pipelines' judge scores; ties are dropped.
- **Checkpoints:** a head is saved every epoch to `bradley_terry.output_path`, and the best one by eval pairwise accuracy is copied to `<output_path>/best/`.
- **Results:** `evaluate.py` writes the summary and per-pair rewards to `bradley_terry.eval_output_file`.
- **Logging:** training logs to wandb.

### Generative reward model (SFT)

```bash
CUDA_VISIBLE_DEVICES=0 reward_model/.venv/bin/python reward_model/generative_rm/sft.py --config_path config.yaml
#   --level A|B|C   override generative_rm.level
./train.sh                  # same thing, on GPU 0; extra flags are passed through
```

- **Levels:** A = overall score only, B = A + rationale, C = B + per-stage assessments and error-propagation analysis.
- **Output:** the adapter is saved to `<generative_rm.output_dir>-level<LEVEL>`.
- **GPUs:** use **one GPU**, or `torchrun` for several. With several GPUs visible and no `torchrun`, the script refuses to start, because DataParallel breaks Qwen-Omni's audio inputs.

To run the adapter in vLLM, first rename its weights. Without this step vLLM loads the adapter but silently ignores it. The conversion reads `<output_dir>-level<LEVEL>` and writes `<vllm_adapter_dir>-level<LEVEL>`:

```bash
reward_model/.venv/bin/python reward_model/generative_rm/convert_adapter_for_vllm.py --config_path config.yaml
#   --level A|B|C          override generative_rm.level
#   --src <adapter dir>    convert a specific checkpoint-N folder instead

reward_model/.venv/bin/python reward_model/generative_rm/eval.py --config_path config.yaml
#   --level A|B|C          override generative_rm.level (picks the matching adapter)
#   --split train|eval|val override generative_rm.eval.split (default: eval)
#   --no_lora              evaluate the base model instead
#   --adapter <dir>        use a different adapter folder
#   --limit N, --overwrite, --stats_only
```

`eval.py` generates an evaluation for every pipeline and compares its overall score with the judge's. It also reports how often the two agree on which pipeline is better. Results go to `generative_rm.eval.results_dir`:
- `<split>_level<LEVEL>.jsonl`: one evaluation per pipeline.
- `<split>_level<LEVEL>_stats.json`: the summary statistics.

With `--no_lora`, the files are named `<split>_base` instead. If a results file already exists, a rerun continues where it stopped; `--overwrite` starts fresh.

## Repository layout

```
config.yaml                  all settings
setup_envs.sh                builds the three venvs
train.sh                     generative reward model SFT on one GPU
data/                        generated data (gitignored: audio, jsonl)
data_generation/
  01_input_request/          01_text_generation (LLM) + 02_tts_synthesis (Orpheus-TTS)
  02_asr/  03_llm/  04_tts/  pipeline stages, each with prompts/ and visualize/
  05_llm_evaluation/         LLM judge, vote agreement, HTML viewer
  06_split_data/             train/eval/val split
reward_model/
  bradley_terry/             train.py, evaluate.py, utils/ (dataset, reward head)
  generative_rm/             sft.py, eval.py, convert_adapter_for_vllm.py, utils/
```
