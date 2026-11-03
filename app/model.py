
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "artifacts" / "model_config.json"
WEIGHTS_PATH = BASE_DIR / "checkpoints" / "best.weights.h5"


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.ffn = keras.Sequential([
            layers.Dense(ffn_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attention_output = self.attention(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.norm2(x + ffn_output)


class MiniGPT(keras.Model):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_heads,
        ffn_dim,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.context_length = context_length

        self.token_embedding = layers.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = layers.Embedding(
            context_length,
            d_model
        )

        self.transformer_block = TransformerBlock(
            d_model,
            num_heads,
            ffn_dim,
            dropout_rate
        )

        self.final_norm = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        seq_length = tf.shape(inputs)[1]

        positions = tf.range(
            start=0,
            limit=seq_length,
            delta=1
        )

        token_embeddings = self.token_embedding(inputs)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.transformer_block(
            x,
            training=training
        )

        x = self.final_norm(x)

        return self.output_layer(x)


def load_model():
    """Reconstruct MiniGPT and load its trained weights."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {CONFIG_PATH}"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        dropout_rate=config["dropout_rate"]
    )

    # Build model variables before loading weights
    dummy_input = tf.zeros(
        (1, config["context_length"]),
        dtype=tf.int32
    )

    model(dummy_input, training=False)

    # Restore trained parameters
    model.load_weights(str(WEIGHTS_PATH))

    print("MiniGPT model loaded successfully!")
    print("Parameters:", model.count_params())

    return model, config
