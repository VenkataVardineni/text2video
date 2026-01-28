"""
FastAPI application for sentiment analysis inference.

Loads trained model weights on startup and exposes /predict endpoint.
"""

import os
import sys
from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_classifier import TextClassifier


# Initialize FastAPI app
app = FastAPI(
    title="SmartReview Sentiment Service",
    description="API for sentiment analysis of customer reviews",
    version="1.0.0"
)

# Middleware to add no-cache headers to static files
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

# Global variables for model and vocabulary
model = None
vocab = None
device = None
max_length = 256


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    label: str
    score: float


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from file."""
    vocab_dict = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, idx = parts
                vocab_dict[word] = int(idx)
    return vocab_dict


def text_to_indices(text: str, vocab: Dict[str, int], max_length: int) -> torch.Tensor:
    """Convert text to tensor of indices."""
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    # Pad or truncate to max_length
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [0] * (max_length - len(indices))
    
    return torch.tensor([indices], dtype=torch.long)


def load_model(model_dir: str = "models"):
    """Load trained model and vocabulary."""
    global model, vocab, device, max_length
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    model_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first."
        )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = TextClassifier(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        num_classes=2,
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    max_length = checkpoint.get('max_length', 256)
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_path}. Please train the model first."
        )
    
    vocab = load_vocab(vocab_path)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Vocabulary size: {len(vocab)}")


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    model_dir = os.getenv("MODEL_DIR", "models")
    try:
        load_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    # Serve all static files including assets
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # Also serve assets directory for React build
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
async def root():
    """Serve frontend index page."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {
        "message": "SmartReview Sentiment Service",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None or vocab is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "Model loaded and ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for given text.
    
    Args:
        request: PredictionRequest containing text to analyze
    
    Returns:
        PredictionResponse with label (positive/negative) and confidence score
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Convert text to tensor
        input_tensor = text_to_indices(request.text, vocab, max_length).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            score, predicted = torch.max(probabilities, dim=1)
        
        # Convert to response format
        label = "positive" if predicted.item() == 1 else "negative"
        confidence = score.item()
        
        return PredictionResponse(label=label, score=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

