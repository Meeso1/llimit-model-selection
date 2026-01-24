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

### Inference

**POST** `/infer`

Run inference on a trained model to score multiple models against multiple prompts.

**Request Body:**
```json
{
  "model": "dense_network/model_name",
  "models_to_score": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
  "prompts": ["What is Python?", "Explain machine learning"],
  "batch_size": 128
}
```

**Request Fields:**

- `model` (string, required): Type and name of the saved model to load in the format `"model_type/model_name"`.
  
- `models_to_score` (array of strings, required): List of model names to evaluate and score

- `prompts` (array of strings, required): List of prompts to evaluate against each model

- `batch_size` (integer, optional): Batch size for inference processing. Default: 128. Must be between 1 and 1024.

**Response:**
```json
{
  "scores": {
    "gpt-3.5-turbo": [0.5, 0.3],
    "gpt-4": [0.8, 0.7],
    "claude-2": [0.6, 0.5]
  }
}
```

Each model maps to an array of scores, one for each prompt in the order they were provided.

**Error Responses:**

- `400 Bad Request`: Invalid model format or parameters
- `404 Not Found`: Model not found

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide an interactive interface to test the API endpoints directly from your browser.

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dense_network/my_model",
    "models_to_score": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
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
