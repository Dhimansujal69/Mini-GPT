
import tensorflow as tf
from app.model import load_model

# Load trained model
model, config = load_model()

print("Configuration:", config)

# Create a TensorFlow input tensor
dummy_input = tf.zeros(
    (1, config["context_length"]),
    dtype=tf.int32
)

# Run inference
output = model(dummy_input, training=False)

print("Test output shape:", output.shape)
print("Model test completed successfully!")
