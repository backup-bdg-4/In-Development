# Remote Model Server Deployment Guide

This guide explains how to deploy the remote model server and configure the main application to use it.

## Overview

The remote model server is a standalone FastAPI application that loads the CoreML model and provides prediction endpoints. It should be deployed separately from the main application, allowing the main application to offload model inference to this external service.

## Deployment Options

### Option 1: Deploy on a Separate Render Instance

1. Create a new Render Web Service
2. Set the following environment variables:
   - `MODEL_PATH`: Path to the CoreML model file (default: `/app/model/BERTSQUADFP16.mlmodel`)
   - `MODEL_SERVER_API_KEY`: API key for authentication (choose a secure value)

3. Use the following build command:
```bash
pip install -r requirements.txt
mkdir -p /app/model
curl -L "https://www.dropbox.com/scl/fi/w4iclrvil6vh39mg6j7pl/BERTSQUADFP16.mlmodel?rlkey=vbrr9jjvsam1xg9i4i19pkdra&st=ho9dyrm6&dl=1" -o /app/model/BERTSQUADFP16.mlmodel
```

4. Use the following start command:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Option 2: Deploy on AWS EC2

1. Launch an EC2 instance (recommended: t3.medium or higher)
2. Install Docker
3. Clone this repository
4. Build the Docker image:
```bash
docker build -t coreml-model-server .
```

5. Run the container:
```bash
docker run -d -p 8000:8000 \
  -v /path/to/your/model:/app/model \
  -e MODEL_SERVER_API_KEY=your-secure-api-key \
  coreml-model-server
```

### Option 3: Deploy on Google Cloud Run

1. Build and push the Docker image to Google Container Registry:
```bash
gcloud builds submit --tag gcr.io/your-project-id/coreml-model-server
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy coreml-model-server \
  --image gcr.io/your-project-id/coreml-model-server \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="MODEL_SERVER_API_KEY=your-secure-api-key"
```

## Configuring the Main Application

After deploying the remote model server, you need to configure the main application to use it:

1. Set the following environment variables in your main application's deployment:
   - `USE_REMOTE_MODEL_SERVER`: Set to `true`
   - `REMOTE_MODEL_SERVER_URL`: URL of your deployed remote model server (e.g., `https://your-model-server.onrender.com`)
   - `REMOTE_MODEL_SERVER_API_KEY`: The API key you set for the remote model server

2. Verify the connection by checking the logs of your main application. You should see messages like:
   - "Using Remote model server for offloading model to external service"
   - "Remote model server connection initialized successfully"

## Security Considerations

1. Always use HTTPS for communication between the main application and the remote model server
2. Use a strong, unique API key for authentication
3. Consider implementing IP whitelisting if your cloud provider supports it
4. Regularly rotate the API key
5. Monitor the remote model server for unusual activity

## Monitoring and Maintenance

1. Set up monitoring for the remote model server to ensure it's running properly
2. Implement automated restarts if the server becomes unresponsive
3. Set up alerts for high CPU/memory usage
4. Regularly check the logs for errors or warnings

## Troubleshooting

If you encounter issues with the remote model server:

1. Check the logs of both the main application and the remote model server
2. Verify that the API key is correctly set in both applications
3. Ensure the remote model server is accessible from the main application
4. Check that the model file is correctly loaded on the remote model server
5. Verify that the remote model server has sufficient resources (CPU/memory)

## Scaling

If you need to handle more traffic:

1. Deploy multiple instances of the remote model server
2. Set up a load balancer to distribute traffic between instances
3. Consider using a managed service like AWS SageMaker or Google Vertex AI for automatic scaling

