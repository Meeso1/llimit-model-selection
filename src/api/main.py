from contextlib import asynccontextmanager
from typing import get_args
from fastapi import FastAPI, HTTPException

from src.api.inference import InferenceService
from src.api.dtos import InferenceRequest, InferenceResponse, HealthResponse
from src.models.model_loading import LengthPredictionModelType, ScoringModelType

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
    Run inference on scoring and/or length prediction models.
    
    This unified endpoint can run scoring, length prediction, or both based on which
    models are specified in the request. If a model is not specified, the corresponding
    output field will be None.
    
    Args:
        request: Inference request containing model specs, model names, and prompts
        
    Returns:
        Response with scores and/or predicted_lengths (fields are None if not requested)
        
    Raises:
        HTTPException: If model format is invalid or inference fails
    """
    assert inference_service is not None, "Inference service is not initialized"

    scoring_spec = None
    if request.scoring_model is not None:
        scoring_spec = _parse_scoring_model_type_and_name(request.scoring_model)
    
    length_spec = None
    if request.length_prediction_model is not None:
        length_spec = _parse_length_prediction_model_type_and_name(request.length_prediction_model)
    
    try:
        if scoring_spec is not None:
            scoring_model_type, scoring_model_name = scoring_spec
            scores = inference_service.score(
                model_type=scoring_model_type,
                model_name=scoring_model_name,
                model_names=request.model_names,
                prompts=request.prompts,
                batch_size=request.batch_size,
            )
        else:
            scores = None

        if length_spec is not None:
            length_model_type, length_model_name = length_spec
            lengths = inference_service.predict_lengths(
                model_type=length_model_type,
                model_name=length_model_name,
                model_names=request.model_names,
                prompts=request.prompts,
                batch_size=request.batch_size,
            )
        else:
            lengths = None
        
        return InferenceResponse(scores=scores, predicted_lengths=lengths)
        
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
        "health": "/health",
        "endpoints": {
            "inference": "/infer",
        }
    }

def _parse_length_prediction_model_type_and_name(model: str) -> tuple[LengthPredictionModelType, str]:
    if "/" not in model:
        raise HTTPException(
            status_code=400,
            detail="Model must be of the form 'model_type/model_name'"
        )
    
    model_type, model_name = model.split("/", 1)
    if model_type not in get_args(LengthPredictionModelType):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid length prediction model type: {model_type}"
        )
    
    return model_type, model_name


def _parse_scoring_model_type_and_name(model: str) -> tuple[ScoringModelType, str]:
    if "/" not in model:
        raise HTTPException(
            status_code=400,
            detail="Model must be of the form 'model_type/model_name'"
        )
    
    model_type, model_name = model.split("/", 1)
    if model_type not in get_args(ScoringModelType):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scoring model type: {model_type}"
        )
    
    return model_type, model_name
