"""Data collator for SFT of the Qwen2.5-Omni Thinker on chat examples.

Each example is ``{"messages": [...]}`` with exactly one audio item in the user
turn. The collator renders the chat, extracts the audio arrays, tokenises text +
audio together, and masks the loss so it is computed only on the assistant JSON.
"""

import torch
from qwen_omni_utils import process_mm_info


class QwenOmniDataCollator:
    def __init__(self, processor, use_audio_in_video: bool = False):
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        # `<|im_start|>assistant` -> first token of the assistant turn header.
        self._assistant_header = processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )

    def __call__(self, examples):
        # `Dataset.from_list` coerces every content dict to the union schema
        # ({type, text, audio}), so text items come back carrying `audio: None`.
        # The Qwen-Omni chat template routes on `'audio' in content` (key
        # presence, not truthiness), so those keys must be dropped or every text
        # item renders as an <|AUDIO|> placeholder.
        conversations = [
            [
                {
                    **msg,
                    "content": [
                        {k: v for k, v in item.items() if v is not None}
                        for item in msg["content"]
                    ],
                }
                for msg in ex["messages"]
            ]
            for ex in examples
        ]

        # Render to text WITHOUT tokenising; the processor call below tokenises and
        # expands each <|AUDIO|> placeholder to match the mel features.
        texts = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=self.use_audio_in_video
        )

        batch = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask everything up to (and including) the assistant header so the loss
        # covers only the assistant response + its <|im_end|>.
        h = len(self._assistant_header)
        tok = self.processor.tokenizer
        for i in range(labels.size(0)):
            seq = batch["input_ids"][i].tolist()
            start = next(
                (j + h for j in range(len(seq) - h + 1) if seq[j : j + h] == self._assistant_header),
                None,
            )
            if start is None:
                raise ValueError(f"assistant header not found in example {i}; check chat template")
            while start < len(seq) and tok.decode([seq[start]]).strip() == "":
                start += 1  # skip the newline right after the header
            labels[i, :start] = -100

        batch["labels"] = labels
        return batch
