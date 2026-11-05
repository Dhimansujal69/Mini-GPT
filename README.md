# MiniGPT — Shakespeare Text Generation API

A character-level GPT (Generative Pre-trained Transformer) built using **TensorFlow and Keras**, trained on Shakespeare's text, and deployed as a REST API using **FastAPI**.

This project demonstrates the complete workflow of building a small GPT-style language model, training it, generating text using sampling strategies, saving model artifacts, and serving predictions through an API.

## Features

- Character-level text generation using a GPT-style Transformer.
- Token and positional embeddings.
- Causal self-attention to prevent the model from accessing future tokens.
- Feed-forward neural network with residual connections and layer normalization.
- Text generation using temperature and Top-K sampling.
- Model checkpoint and vocabulary loading.
- FastAPI-based REST API.
- Interactive API documentation with Swagger UI.
- Input validation and error handling.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning model |
| NumPy | Numerical operations |
| FastAPI | REST API development |
| Pydantic | Request validation |
| Uvicorn | ASGI server |
| Google Colab | Model training environment |
| Jupyter Notebook | Experimentation |

## Project Architecture

The project follows this workflow:

1. Train a character-level GPT model using Shakespeare's text.
2. Save the trained model weights, vocabulary, and configuration.
3. Reconstruct the model architecture locally.
4. Load the trained weights and vocabulary.
5. Generate text using temperature and Top-K sampling.
6. Expose the generation functionality through FastAPI.

```text
             User Request
                  |
                  v
             FastAPI API
                  |
                  v
          Request Validation
                  |
                  v
           Text Generator
                  |
                  v
             MiniGPT Model
                  |
                  v
         Next Character Prediction
                  |
                  v
          Sampling (Top-K)
                  |
                  v
          Generated Text
                  |
                  v
            JSON Response
```

## Project Structure

```text
mini-gpt/
│
├── app/
│   ├── __init__.py
│   ├── model.py
│   ├── generator.py
│   └── main.py
│
├── artifacts/
│   ├── model_config.json
│   └── vocabulary.json
│
├── checkpoints/
│   └── best.weights.h5
│
├── test_model.py
├── test_generator.py
├── requirements.txt
└── README.md
```

### Important Files

- `app/model.py` — Defines the Transformer block, MiniGPT architecture, and model-loading function.
- `app/generator.py` — Handles tokenization, text generation, temperature, and Top-K sampling.
- `app/main.py` — Defines FastAPI application and API endpoints.
- `artifacts/model_config.json` — Stores model hyperparameters.
- `artifacts/vocabulary.json` — Stores character-to-ID and ID-to-character mappings.
- `checkpoints/best.weights.h5` — Contains the trained model weights.
- `test_model.py` — Tests model loading and output dimensions.
- `test_generator.py` — Tests text generation independently.

## Model Architecture

MiniGPT is a small, decoder-only Transformer language model.

| Parameter | Value |
|---|---:|
| Model type | Decoder-only Transformer |
| Tokenization | Character-level |
| Vocabulary size | 65 |
| Context length | 40 |
| Embedding dimension | 32 |
| Number of attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Total parameters | 18,273 |

### Main Components

1. **Token Embedding:** Converts character IDs into dense vector representations.
2. **Positional Embedding:** Adds information about character positions.
3. **Causal Self-Attention:** Allows each position to attend only to previous and current positions.
4. **Feed-Forward Network:** Learns additional representations through nonlinear transformations.
5. **Residual Connections and Layer Normalization:** Help stabilize training.
6. **Output Projection:** Produces logits for predicting the next character.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd mini-gpt
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify the Model Files

Make sure the following files exist:

```text
artifacts/model_config.json
artifacts/vocabulary.json
checkpoints/best.weights.h5
```

The API requires these files to load the trained model.

## Running the Project

### 1. Test Model Loading

```bash
python test_model.py
```

Expected output:

```text
MiniGPT model loaded successfully!
Parameters: 18273
Test output shape: (1, 40, 65)
Model test completed successfully!
```

### 2. Test Text Generation

```bash
python test_generator.py
```

This tests text generation without starting the API server.

### 3. Start the FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Documentation

FastAPI provides interactive documentation through Swagger UI.

**Swagger UI:** http://127.0.0.1:8000/docs

**ReDoc:** http://127.0.0.1:8000/redoc

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message and API status |
| GET | `/health` | Checks API and model loading status |
| POST | `/generate` | Generates text from a given prompt |

### 1. Home Endpoint

**GET** `/`

Example response:

```json
{
  "message": "Welcome to MiniGPT API",
  "status": "running",
  "documentation": "/docs"
}
```

### 2. Health Check

**GET** `/health`

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Generate Text

**POST** `/generate`

Request body:

```json
{
  "prompt": "ROMEO:",
  "max_new_chars": 200,
  "temperature": 0.8,
  "top_k": 10
}
```

Example response:

```json
{
  "prompt": "ROMEO:",
  "generated_text": "ROMEO: ...",
  "parameters": {
    "max_new_chars": 200,
    "temperature": 0.8,
    "top_k": 10
  }
}
```

The generated text will vary between requests because the model uses probabilistic sampling.

## Text Generation Parameters

| Parameter | Description | Default |
|---|---|---:|
| `prompt` | Starting text for generation | Required |
| `max_new_chars` | Maximum number of new characters | 200 |
| `temperature` | Controls randomness during sampling | 0.8 |
| `top_k` | Number of highest-scoring candidates considered | 10 |

### Understanding Temperature

- Lower temperature produces more predictable and repetitive text.
- Higher temperature produces more diverse and random text.
- A temperature around `0.8` provides a reasonable balance for experimentation.

### Understanding Top-K Sampling

Top-K sampling restricts the next-character selection to the K most likely candidates.

- `top_k = 1` behaves like greedy selection.
- Larger K values allow more diversity.
- The maximum supported value is 65, matching the vocabulary size.

## Training Information

The model was trained on the Shakespeare text dataset using TensorFlow/Keras.

Training configuration:

- Context length: 40
- Batch size: 32
- Embedding dimension: 32
- Attention heads: 4
- Feed-forward dimension: 128
- Epochs: 5

The best checkpoint was saved based on validation loss.

**Note:** This is an educational MiniGPT implementation, not a production-scale language model. Generated text may contain grammatical errors and inconsistencies due to the relatively small model size and limited training.

## Future Improvements

- Increase model depth and embedding dimensions.
- Train for more epochs and experiment with hyperparameters.
- Add streaming text generation.
- Build a frontend interface for interacting with the model.
- Add automated API tests using Pytest.
- Containerize the application using Docker.
- Deploy the API to a cloud platform.

## Learning Outcomes

Through this project, the following concepts were implemented:

- Transformer and decoder-only GPT architecture.
- Self-attention and causal masking.
- Token and positional embeddings.
- Language model training and next-character prediction.
- Cross-entropy loss and optimization.
- Autoregressive text generation.
- Temperature and Top-K sampling.
- Model saving and loading.
- REST API development using FastAPI.
- API request validation and error handling.

## Author

**Sujal Dhiman**

GitHub: [Dhimansujal69](https://github.com/Dhimansujal69)

---

*Built as a hands-on learning project to understand GPT architecture, TensorFlow model development, and API deployment.*
