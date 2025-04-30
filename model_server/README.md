# CoreML Model Server

This is a standalone server for hosting the BERTSQUADFP16.mlmodel CoreML model. It provides a REST API for making predictions with the model.

## Setup

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- The BERTSQUADFP16.mlmodel file

### Installation

1. Clone this repository
2. Place your BERTSQUADFP16.mlmodel file in the `model` directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server

#### Directly with Python

```bash
# Set environment variables
export MODEL_PATH=/path/to/your/BERTSQUADFP16.mlmodel
export MODEL_SERVER_API_KEY=your-secure-api-key

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### With Docker

```bash
# Build the Docker image
docker build -t coreml-model-server .

# Run the container
docker run -p 8000:8000 \
  -v /path/to/your/model:/app/model \
  -e MODEL_SERVER_API_KEY=your-secure-api-key \
  coreml-model-server
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the server and whether the model is loaded.

### Model Status

```
GET /model/status
```

Returns detailed information about the model's loading status.

### Prediction

```
POST /predict
```

Makes a prediction using the model.

Request body:
```json
{
  "query_text": "What is AI?",
  "passage_text": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines."
}
```

Response:
```json
{
  "answer": "the simulation of human intelligence processes by machines",
  "confidence": 0.95,
  "start_index": 29,
  "end_index": 79
}
```

## Authentication

The server uses API key authentication. Set the `MODEL_SERVER_API_KEY` environment variable to a secure value, and include it in requests as a Bearer token:

```
Authorization: Bearer your-api-key
```

## Deployment

This server can be deployed to any platform that supports Docker containers or Python applications, such as:

- AWS EC2
- Google Cloud Compute Engine
- Azure Virtual Machines
- Heroku
- Digital Ocean

For production deployments, consider:
- Setting up HTTPS
- Using a more secure authentication method
- Implementing rate limiting
- Setting up monitoring and logging

