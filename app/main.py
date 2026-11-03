
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generator import TextGenerator


# Shared generator instance
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    print("Loading MiniGPT...")
    generator = TextGenerator()
    print("MiniGPT API is ready!")

    yield

    print("Shutting down MiniGPT API...")


app = FastAPI(
    title="MiniGPT Text Generation API",
    description="Character-level Shakespeare text generation using TensorFlow.",
    version="1.0.0",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Starting text prompt"
    )

    max_new_chars: int = Field(
        default=200,
        ge=1,
        le=1000
    )

    temperature: float = Field(
        default=0.8,
        gt=0,
        le=2.0
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=65
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to MiniGPT API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None
    }


@app.post("/generate")
def generate_text(request: GenerateRequest):

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:
        generated_text = generator.generate(
            prompt=request.prompt,
            max_new_chars=request.max_new_chars,
            temperature=request.temperature,
            top_k=request.top_k
        )

        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "parameters": {
                "max_new_chars": request.max_new_chars,
                "temperature": request.temperature,
                "top_k": request.top_k
            }
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
