"""FastAPI server for model inference.

Run with:
    uvicorn LILITH.serve:app --host 0.0.0.0 --port 8000

Or:
    python -m LILITH.serve
"""
from __future__ import annotations

from typing import Optional, List
from pathlib import Path

import torch
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None

from .model import GPTDecoder
from .sampling import sample_with_strategy
from .train_phase2 import load_checkpoint


# Request/Response models
class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(100, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.0, le=2.0)
    strategy: str = Field("top_p", description="Sampling strategy")
    top_k: Optional[int] = Field(50, description="Top-k for sampling", ge=1)
    top_p: Optional[float] = Field(0.9, description="Top-p for nucleus sampling", ge=0.0, le=1.0)
    num_return: int = Field(1, description="Number of sequences to return", ge=1, le=10)


class GenerateResponse(BaseModel):
    """Response from text generation."""
    generated_text: List[str]
    prompt: str
    num_tokens: int


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    parameters: int
    vocab_size: int
    max_seq_len: int
    device: str


# Global model storage
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_MODEL_INFO = None


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    global _MODEL, _TOKENIZER, _DEVICE, _MODEL_INFO

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    model, tokenizer, metadata = load_checkpoint(checkpoint_path, device=device_obj)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _DEVICE = device_obj
    _MODEL_INFO = {
        "name": f"LILITH-{model.count_parameters() // 1_000_000}M",
        "parameters": model.count_parameters(),
        "vocab_size": model.config.vocab_size,
        "max_seq_len": model.config.max_seq_len,
        "device": str(device_obj),
    }

    print(f"✓ Model loaded: {_MODEL_INFO['name']}")
    print(f"  Parameters: {_MODEL_INFO['parameters']:,}")
    print(f"  Device: {_MODEL_INFO['device']}")


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="LILITH API",
        description="Language model inference server for LILITH",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        # Try to load from default location
        default_checkpoint = "checkpoints/wikipedia/final_model.pt"
        if Path(default_checkpoint).exists():
            load_model_from_checkpoint(default_checkpoint)
        else:
            print("⚠ No model loaded. Use POST /load_model to load a checkpoint.")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "LILITH API Server",
            "endpoints": {
                "info": "GET /info",
                "generate": "POST /generate",
                "load_model": "POST /load_model",
            }
        }

    @app.get("/info", response_model=ModelInfo)
    async def get_model_info():
        """Get model information."""
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return _MODEL_INFO

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text from prompt."""
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        try:
            # Encode prompt
            prompt_ids = torch.tensor([_TOKENIZER.encode(request.prompt)], device=_DEVICE)

            # Generate
            generated_texts = []
            for _ in range(request.num_return):
                with torch.no_grad():
                    generated = sample_with_strategy(
                        _MODEL,
                        prompt_ids,
                        max_new_tokens=request.max_tokens,
                        strategy=request.strategy,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                    )

                # Decode
                text = _TOKENIZER.decode(generated[0].tolist())
                generated_texts.append(text)

            return GenerateResponse(
                generated_text=generated_texts,
                prompt=request.prompt,
                num_tokens=len(generated[0]),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/load_model")
    async def load_model_endpoint(checkpoint_path: str, device: str = "cuda"):
        """Load a model from checkpoint."""
        try:
            load_model_from_checkpoint(checkpoint_path, device)
            return {"message": f"Model loaded successfully", "info": _MODEL_INFO}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": _MODEL is not None,
        }

else:
    # Stub for when FastAPI is not available
    app = None
    print("FastAPI not available. Install with: pip install fastapi uvicorn")


# CLI entry point
def serve_cli():
    """CLI entry point for serving."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve LILITH model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed. Install with:")
        print("  pip install fastapi uvicorn")
        return

    # Load model
    load_model_from_checkpoint(args.checkpoint, args.device)

    # Start server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    serve_cli()


__all__ = ["app", "load_model_from_checkpoint", "serve_cli"]
