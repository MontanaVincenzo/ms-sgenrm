from torch.utils.data import Dataset
import json
import os
import random

PIPELINE_KEYS = ["pipeline1", "pipeline2"]


def _resolve_audio(path, audio_dir):
    """Absolute paths are kept as-is; relative paths are resolved under audio_dir."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(audio_dir, path)


def _preference_key(overall_evaluation: dict):
    """
    Sort key used to decide which pipeline is 'chosen'.

    Primary: the aggregated integer overall_score (1-5).
    Tie-break: the mean of the per-vote scores, when available — this pulls apart
    pairs that share an integer score but clearly lean one way across the 10 votes.
    """
    voting = overall_evaluation.get("voting", {}).get("overall_score", {})
    score = overall_evaluation["overall_score"]
    return (score, voting.get("mean", score))


class PairDataset(Dataset):
    """
    Yields (chosen, rejected) preference pairs for Bradley-Terry training on the
    full spoken-dialogue pipeline (ASR -> LLM -> TTS).

    Each JSONL line holds one request and two pipeline runs (`pipeline1`,
    `pipeline2`), each with an `asr` / `llm` / `tts` stage output and an
    `overall_evaluation`. The pipeline with the higher overall human score is the
    'chosen' one; pairs that are exactly tied (after the vote-mean tie-break) are
    dropped when `drop_ties` is set.
    """

    def __init__(self, jsonl_file, audio_dir="audios", drop_ties=True, seed=42):
        self.audio_dir = audio_dir
        rng = random.Random(seed)

        with open(jsonl_file, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        self.dataset = []
        n_ties = 0
        for sample in raw:
            k1 = _preference_key(sample["pipeline1"]["overall_evaluation"])
            k2 = _preference_key(sample["pipeline2"]["overall_evaluation"])

            if k1 == k2:
                n_ties += 1
                if drop_ties:
                    continue
                kchosen, krejected = rng.choice(
                    [("pipeline1", "pipeline2"), ("pipeline2", "pipeline1")]
                )
            elif k1 > k2:
                kchosen, krejected = "pipeline1", "pipeline2"
            else:
                kchosen, krejected = "pipeline2", "pipeline1"

            request_audio = _resolve_audio(
                sample["input_request"]["audio_path"], audio_dir
            )

            self.dataset.append({
                "chosen_request_audio_path": request_audio,
                "rejected_request_audio_path": request_audio,
                "chosen_transcription": sample[kchosen]["asr"]["output"],
                "rejected_transcription": sample[krejected]["asr"]["output"],
                "chosen_tts_text": sample[kchosen]["llm"]["output"],
                "rejected_tts_text": sample[krejected]["llm"]["output"],
                "chosen_response_audio_path": _resolve_audio(
                    sample[kchosen]["tts"]["output"], audio_dir
                ),
                "rejected_response_audio_path": _resolve_audio(
                    sample[krejected]["tts"]["output"], audio_dir
                ),
            })

        print(
            f"[PairDataset] {jsonl_file}: {len(self.dataset)} pairs "
            f"({n_ties} ties {'dropped' if drop_ties else 'kept (randomised)'})"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def build_conversation(
    request_audio_path: str,
    transcription: str,
    tts_text: str,
    tts_audio_path: str,
) -> list:
    """
    Build a bare three-turn spoken-dialogue conversation (ASR -> LLM -> TTS)
    compatible with Qwen2.5-Omni's apply_chat_template. No evaluation
    instructions are included; the reward signal comes from BT training.

        turn 1  user/assistant : request audio  -> transcription
        turn 2  user/assistant : (instruction)  -> response text
        turn 3  user/assistant : (instruction)  -> response audio
    """
    if None in (request_audio_path, transcription, tts_text, tts_audio_path):
        raise ValueError(
            "build_conversation requires request audio, transcription, "
            "response text and response audio"
        )

    system_text = (
        "You are an end-to-end spoken dialogue assistant. "
        "You receive spoken audio requests and respond by: "
        "(1) transcribing the spoken request accurately, "
        "(2) generating a relevant and helpful textual response, "
        "(3) synthesizing that response as natural, expressive speech."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Please transcribe the following spoken request."},
            {"type": "audio", "audio": request_audio_path}]},
        {"role": "assistant", "content": [{"type": "text", "text": transcription}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Based on the transcribed request, provide a "
             "helpful and concise textual response."}]},
        {"role": "assistant", "content": [{"type": "text", "text": tts_text}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Now synthesize your textual response as natural spoken audio."}]},
        {"role": "assistant", "content": [{"type": "audio", "audio": tts_audio_path}]},
    ]


def build_inputs(
    processor,
    request_audio_path: str,
    transcription: str,
    tts_audio_path: str,
    tts_text: str,
) -> dict:
    """
    Format the chat-template text string for one reward-model input covering the
    full ASR -> LLM -> TTS pipeline. All four fields are required.

    Audio extraction is handled separately via process_mm_info in collate_fn.
    """
    conversation = build_conversation(
        request_audio_path=request_audio_path,
        transcription=transcription,
        tts_text=tts_text,
        tts_audio_path=tts_audio_path,
    )

    formatted_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=False,
    )

    return {
        "text": formatted_text,
        "conversation": conversation,
    }
