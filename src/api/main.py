from contextlib import asynccontextmanager
from typing import get_args
from fastapi import FastAPI, HTTPException

from src.api.inference import InferenceService
from src.api.dtos import InferenceRequest, InferenceResponse, HealthResponse
from src.scripts.model_types import ModelType

inference_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global inference_service
    inference_service = InferenceService()
    yield
    inference_service = None


app = FastAPI(
    title="LLimit Model Selection API",
    description="API for running inference on trained model selection models",
    version="1.0.0",
    lifespan=lifespan,
)



@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest) -> InferenceResponse:
    """
    Run inference on a trained model.
    
    Args:
        request: Inference request containing model info, models to score, and prompts
        
    Returns:
        Dictionary with scores for each model across all prompts
        
    Raises:
        HTTPException: If model format is invalid or inference fails
    """
    assert inference_service is not None, "Inference service is not initialized"

    model_type, model_name = _parse_model_type_and_name(request.model)
    
    try:
        scores = inference_service.infer(
            model_type=model_type,
            model_name=model_name,
            models_to_score=request.models_to_score,
            prompts=request.prompts,
            batch_size=request.batch_size,
        )
        
        return InferenceResponse(scores=scores)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "LLimit Model Selection API",
        "docs": "/docs",
        "health": "/health"
    }

def _parse_model_type_and_name(model: str) -> tuple[ModelType, str]:
    if "/" not in model:
        raise HTTPException(
            status_code=400,
            detail="Model must be of the form 'model_type/model_name'"
        )
    
    model_type, model_name = model.split("/", 1)
    if model_type not in get_args(ModelType):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type: {model_type}"
        )
    
    return model_type, model_name
