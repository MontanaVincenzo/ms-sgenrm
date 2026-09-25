"""Turn the pipeline-evaluation dataset into Qwen2.5-Omni chat examples.

Each training example conditions the model on (spoken request audio, ASR
transcription, LLM answer) and targets the `overall_evaluation` JSON for that
single pipeline. Pipelines are scored individually -- there is no pairwise
comparison at this stage.
"""

import json
from pathlib import Path

from datasets import Dataset

PIPELINE_KEYS = ("pipeline1", "pipeline2")

# Grounding levels: how much of `overall_evaluation` the model is trained to
# reproduce, from bare label (A) up to full rationale + error-propagation
# analysis (C). Each level's target keys and the prompt's "Return ONLY..."
# instruction must stay in lockstep, so both are driven off this one table.
LEVELS = ("A", "B", "C")

_LEVEL_KEYS = {
    "A": ("overall_score",),
    "B": ("overall_notes", "overall_score"),
    "C": ("asr_assessment", "llm_assessment", "tts_assessment", "compounding", "overall_notes", "overall_score"),
}

_LEVEL_INSTRUCTION = {
    "A": "Return ONLY a JSON object with key: overall_score.",
    "B": "Return ONLY a JSON object with keys: overall_notes, overall_score.",
    "C": (
        "Return ONLY a JSON object with keys: asr_assessment, llm_assessment, "
        "tts_assessment, compounding, overall_notes, overall_score."
    ),
}

_SYSTEM_PROMPT_BODY = (
    "You are an evaluator of a spoken cascaded ASR -> LLM -> TTS pipeline. You judge a "
    "single pipeline for one user turn. You are given the user's spoken request, "
    "the ASR transcription the LLM received, the LLM's answer, and the TTS synthesis.\n\n"
    "Evaluation Rules:\n"
    "1. ASR: Grade transcription accuracy against the user's original spoken request.\n"
    "2. LLM: Judge the text answer based strictly on the information available in the ASR transcription, "
    "and check if it ultimately serves the user's true intent.\n"
    "3. TTS: Evaluate the synthesized audio against the text produced by the LLM. Specifically assess:\n"
    "   - Text Fidelity: Any addition, substitution, or omission of words relative to the LLM's text output.\n"
    "   - Voice Style & Delivery:\n"
    "     * Emotional inversion: Mismatch between the tone/emotion of the audio and the textual sentiment (e.g., cheerful tone during serious news).\n"
    "     * Acoustic shift: Sudden or unnatural changes in pitch, timbre, volume, or speaker identity mid-utterance.\n"
    "     * Temporal / rhythm shift: Unnatural pacing, awkward pauses, erratic cadence, or robotic phrasing.\n\n"
)

USER_INSTRUCTION = "Evaluate this pipeline turn and return the evaluation JSON."


def system_prompt(level: str = "C") -> str:
    """System prompt for a given grounding level; the output-keys instruction tracks the level."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return _SYSTEM_PROMPT_BODY + _LEVEL_INSTRUCTION[level]


# Kept for any external importer relying on the old full-detail constant.
SYSTEM_PROMPT = system_prompt("C")


def build_conversation(
    input_audio_path, asr_transcription, llm_answer, tts_audio_path, target: str | None = None, level: str = "C"
):
    """Build the message list for one example. `target` None -> inference prompt."""
    user_content = [
        {"type": "text", "text": USER_INSTRUCTION},
        {"type": "text", "text": "User's spoken request:"},
        {"type": "audio", "audio": str(input_audio_path)},
        {"type": "text", "text": f"ASR transcription:\n{asr_transcription}"},
        {"type": "text", "text": f"LLM answer:\n{llm_answer}"},
        {"type": "text", "text": "TTS Synthesis:"},
        {"type": "audio", "audio": str(tts_audio_path)},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(level)}]},
        {"role": "user", "content": user_content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _filter_target(overall_evaluation: dict, level: str) -> dict | None:
    """Keep only the fields for `level`; None if any of them is missing."""
    filtered = {k: overall_evaluation.get(k) for k in _LEVEL_KEYS[level]}
    if any(v is None for v in filtered.values()):
        return None
    return filtered


def _target_json(overall_evaluation: dict) -> str:
    return json.dumps(overall_evaluation, ensure_ascii=False, indent=2)


def build_dataset(input_file, audio_dir=None, level: str = "C", root_dir=None) -> Dataset:
    """One row per pipeline that has an ASR hypothesis, an LLM answer and a target.

    `level` controls how much of `overall_evaluation` becomes the training target:
    A = overall_score only, B = A + overall_notes, C = B + per-stage assessments
    and the compounding (error-propagation) analysis.

    Audio paths: with `audio_dir`, each file is looked up by basename in that one folder;
    otherwise the stored path is used, resolved against `root_dir` when it is relative.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    with open(input_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    audio_dir = Path(audio_dir) if audio_dir else None
    root_dir = Path(root_dir) if root_dir else Path()

    def resolve(raw):
        return audio_dir / Path(raw).name if audio_dir else root_dir / raw  # absolute raw paths win over root_dir

    rows = []
    for rec in records:
        raw_audio = rec["input_request"]["audio_path"]
        if not raw_audio:
            continue
        audio_path = str(resolve(raw_audio))

        for pk in PIPELINE_KEYS:
            stage = rec.get(pk)
            if not stage:
                continue
            asr_out = stage["asr"]["output"]
            llm_out = stage["llm"]["output"]
            tts_out = stage["tts"]["output"]
            overall_evaluation = stage.get("overall_evaluation")
            if not asr_out or not llm_out or not tts_out or not overall_evaluation:
                continue
            target = _filter_target(overall_evaluation, level)
            if target is None:
                continue

            tts_out = resolve(stage["tts"]["output"])
            if not tts_out.is_file():
                continue
            rows.append(
                {
                    "messages": build_conversation(
                        audio_path, asr_out, llm_out, tts_out, _target_json(target), level=level
                    )
                }
            )

    if not rows:
        raise ValueError(f"no usable (asr, llm, overall_evaluation) triples for level {level} in {input_file}")
    return Dataset.from_list(rows)
