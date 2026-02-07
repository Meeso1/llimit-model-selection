from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request model for unified inference endpoint."""
    scoring_model: str | None = Field(
        default=None,
        description="Type and name of the scoring model to use (e.g. 'dn_embedding/model_name'). If omitted, scoring will not be performed.",
        examples=["dn_embedding/my_model"]
    )
    length_prediction_model: str | None = Field(
        default=None,
        description="Type and name of the length prediction model to use (e.g. 'dn_embedding_length_prediction/model_name'). If omitted, length prediction will not be performed.",
        examples=["dn_embedding_length_prediction/my_model"]
    )
    model_names: list[str] = Field(
        ...,
        description="List of model names to evaluate",
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
    """Response model for unified inference endpoint."""
    scores: dict[str, list[float]] | None = Field(
        default=None,
        description="Dictionary mapping model names to their scores for each prompt. None if scoring_model was not provided.",
        examples=[{
            "gpt-3.5-turbo": [0.5, 0.3],
            "gpt-4": [0.8, 0.7],
            "claude-2": [0.6, 0.5]
        }]
    )
    predicted_lengths: dict[str, list[float]] | None = Field(
        default=None,
        description="Dictionary mapping model names to their predicted response lengths (in tokens) for each prompt. None if length_prediction_model was not provided.",
        examples=[{
            "gpt-3.5-turbo": [150.5, 200.3],
            "gpt-4": [180.2, 220.7],
            "claude-2": [160.8, 190.4]
        }]
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str