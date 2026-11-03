
import json
import tensorflow as tf
from pathlib import Path

from app.model import load_model


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "artifacts" / "vocabulary.json"


class TextGenerator:
    def __init__(self):
        # Load trained model
        self.model, self.config = load_model()

        self.context_length = self.config["context_length"]
        self.vocab_size = self.config["vocab_size"]

        # Load vocabulary
        with open(VOCAB_PATH, "r", encoding="utf-8") as file:
            vocab = json.load(file)

        self.char_to_id = vocab["char_to_id"]

        self.id_to_char = {
            int(idx): char
            for idx, char in vocab["id_to_char"].items()
        }

    def generate(
        self,
        prompt,
        max_new_chars=200,
        temperature=0.8,
        top_k=10
    ):
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        if not 1 <= top_k <= self.vocab_size:
            raise ValueError(
                f"top_k must be between 1 and {self.vocab_size}."
            )

        if max_new_chars < 1:
            raise ValueError("max_new_chars must be at least 1.")

        # Encode prompt
        unknown_chars = [
            char for char in prompt
            if char not in self.char_to_id
        ]

        if unknown_chars:
            raise ValueError(
                f"Unsupported characters in prompt: {unknown_chars}"
            )

        token_ids = [
            self.char_to_id[char] for char in prompt
        ]

        generated = token_ids.copy()

        # Autoregressive generation
        for _ in range(max_new_chars):

            context = generated[-self.context_length:]

            input_ids = tf.constant(
                [context],
                dtype=tf.int32
            )

            logits = self.model(
                input_ids,
                training=False
            )

            next_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            top_values, _ = tf.math.top_k(
                next_logits,
                k=top_k
            )

            threshold = top_values[-1]

            filtered_logits = tf.where(
                next_logits >= threshold,
                next_logits,
                tf.constant(-1e9, dtype=next_logits.dtype)
            )

            # Sample next character
            next_id = tf.random.categorical(
                filtered_logits[tf.newaxis, :],
                num_samples=1
            )[0, 0].numpy()

            generated.append(int(next_id))

        # Decode generated characters
        return "".join(
            self.id_to_char[idx]
            for idx in generated
        )
