# Inference API

The project provides a FastAPI-based REST API for running model inference. The API behaves similarly to the CLI inference command but is accessible via HTTP requests.

## Running the API

### Local Development

Run the API server locally using uvicorn:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker

Build and run the API using Docker:

```bash
# Build the API image
docker build -f Dockerfile.api -t llimit-api .

# Run the API container
docker run -p 8000:8000 llimit-api
```

For GPU support:

```bash
docker run --gpus all -p 8000:8000 llimit-api
```

## API Endpoints

### Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Root Information

**GET** `/`

Get basic API information and links to documentation.

**Response:**
```json
{
  "message": "LLimit Model Selection API",
  "docs": "/docs",
  "health": "/health"
}
```

### Unified Inference

**POST** `/infer`

Run inference on scoring and/or length prediction models. This unified endpoint can perform scoring, length prediction, or both simultaneously based on which models are specified in the request.

**Request Body:**
```json
{
  "scoring_model": "dn_embedding/my_scoring_model",
  "length_prediction_model": "dn_embedding_length_prediction/my_length_model",
  "model_names": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
  "prompts": ["What is Python?", "Explain machine learning"],
  "batch_size": 128
}
```

**Request Fields:**

- `scoring_model` (string, optional): Type and name of the scoring model to use in the format `"model_type/model_name"`. Example: `"dn_embedding/my_model"`. If omitted, scoring will not be performed and `scores` in the response will be `null`.

- `length_prediction_model` (string, optional): Type and name of the length prediction model to use in the format `"model_type/model_name"`. Example: `"dn_embedding_length_prediction/my_model"`. If omitted, length prediction will not be performed and `predicted_lengths` in the response will be `null`.
  
- `model_names` (array of strings, required): List of model names to evaluate

- `prompts` (array of strings, required): List of prompts to evaluate

- `batch_size` (integer, optional): Batch size for inference processing. Default: 128. Must be between 1 and 1024.

**Response:**

The response always contains both `scores` and `predicted_lengths` fields. Fields corresponding to models that were not requested will be `null`.

**Example 1: Both scoring and length prediction**
```json
{
  "scores": {
    "gpt-3.5-turbo": [0.5, 0.3],
    "gpt-4": [0.8, 0.7],
    "claude-2": [0.6, 0.5]
  },
  "predicted_lengths": {
    "gpt-3.5-turbo": [150.5, 200.3],
    "gpt-4": [180.2, 220.7],
    "claude-2": [160.8, 190.4]
  }
}
```

**Example 2: Only scoring**
```json
{
  "scores": {
    "gpt-3.5-turbo": [0.5, 0.3],
    "gpt-4": [0.8, 0.7]
  },
  "predicted_lengths": null
}
```

**Example 3: Only length prediction**
```json
{
  "scores": null,
  "predicted_lengths": {
    "gpt-3.5-turbo": [150.5, 200.3],
    "gpt-4": [180.2, 220.7]
  }
}
```

**Example 4: Neither model specified**
```json
{
  "scores": null,
  "predicted_lengths": null
}
```

**Error Responses:**

- `400 Bad Request`: Invalid model format, parameters, or wrong model kind
- `404 Not Found`: Model not found

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide an interactive interface to test the API endpoints directly from your browser.

## Example Usage

### Using curl

**Both scoring and length prediction:**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_model": "dn_embedding/my_scoring_model",
    "length_prediction_model": "dn_embedding_length_prediction/my_length_model",
    "model_names": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
    "prompts": ["What is Python?", "Explain machine learning"],
    "batch_size": 128
  }'
```

**Only scoring:**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_model": "dn_embedding/my_model",
    "model_names": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
    "prompts": ["What is Python?", "Explain machine learning"],
    "batch_size": 128
  }'
```

**Only length prediction:**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "length_prediction_model": "dn_embedding_length_prediction/my_model",
    "model_names": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
    "prompts": ["What is Python?", "Explain machine learning"],
    "batch_size": 128
  }'
```

## Features

### Model Caching

The API caches last loaded model in memory to avoid reloading them for subsequent requests. This significantly improves performance for repeated inference calls with the same model.

### Batch Processing

The API supports batch processing through the `batch_size` parameter, which can be tuned based on available memory and desired performance.

### GPU Support

When running with GPU support (via CUDA), the API will automatically utilize available GPUs for inference acceleration.
