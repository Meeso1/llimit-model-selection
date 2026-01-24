from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    model: str = Field(
        ...,
        description="Type and name of the saved model to load (e.g. 'dense_network/model_name')",
        examples=["dense_network/my_model"]
    )
    models_to_score: list[str] = Field(
        ...,
        description="List of model names to score",
        examples=[["gpt-3.5-turbo", "gpt-4", "claude-2"]]
    )
    prompts: list[str] = Field(
        ...,
        description="List of prompts to evaluate",
        examples=[["What is Python?", "Explain machine learning"]]
    )
    batch_size: int = Field(
        default=128,
        description="Batch size for inference",
        ge=1,
        le=1024
    )


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    scores: dict[str, list[float]] = Field(
        ...,
        description="Dictionary mapping model names to their scores for each prompt",
        examples=[{
            "gpt-3.5-turbo": [0.5, 0.3],
            "gpt-4": [0.8, 0.7],
            "claude-2": [0.6, 0.5]
        }]
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str