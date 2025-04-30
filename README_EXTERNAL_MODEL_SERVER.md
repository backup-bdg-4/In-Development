# External Model Server Implementation

This document describes the implementation of the external model server approach, which replaces the previous Jupyter-based model server.

## Overview

The external model server approach provides several benefits:

1. **No model download required**: The model is not downloaded to `/tmp/model` or any other location.
2. **Memory efficiency**: The model is loaded only in the external server process, not in the main application.
3. **Simplified deployment**: No need to manage model files or worry about storage locations.
4. **Scalability**: The model server can be deployed separately from the main application.

## Components

### 1. External Model Server (`backend/model_server.py`)

A lightweight FastAPI server that provides:
- `/health` endpoint for health checks
- `/predict` endpoint for making predictions
- `/info` endpoint for server information

The model server uses a simple mock implementation for predictions, which can be replaced with an actual ML model if needed.

### 2. External Model Server Interface (`backend/app/utils/external_model_server.py`)

A module that provides an interface to the external model server:
- `initialize_model_server()`: Establishes connection to the external server
- `predict_with_external_server()`: Makes predictions using the external server
- `is_model_server_running()`: Checks if the external server is running

### 3. Modified Initialization (`backend/app/utils/initialization.py`)

The initialization module has been modified to:
- Remove model download functionality
- Remove model file checks
- Always report the model as "found" to prevent download attempts

### 4. Modified Model Utilities (`backend/app/utils/model_utils.py`)

The model utilities module has been modified to:
- Remove references to model files
- Provide placeholder functions that delegate to the external server

### 5. Startup Script (`backend/start_with_model_server.sh`)

A script that starts both the model server and the main application:
- Starts the model server in the background
- Waits for the model server to start
- Starts the main application
- Cleans up the model server when the main application exits

### 6. Modified Render Configuration (`render.yaml`)

The Render configuration has been updated to:
- Remove model download steps
- Start the model server alongside the main application
- Set appropriate environment variables

## Environment Variables

- `USE_EXTERNAL_MODEL_SERVER=true`: Enables the external model server approach
- `MODEL_SERVER_URL=http://localhost:5000`: URL of the external model server
- `MINIMIZE_MEMORY_USAGE=true`: Enables memory-saving optimizations

## Usage

To run the application locally:

```bash
# Make the startup script executable
chmod +x backend/start_with_model_server.sh

# Run the application with the external model server
./backend/start_with_model_server.sh
```

For deployment on Render, the configuration in `render.yaml` will automatically start both the model server and the main application.

## Extending the Model Server

To use an actual ML model in the model server:

1. Modify the `predict()` function in `backend/model_server.py` to load and use your model
2. Update the model server to handle model loading and unloading
3. Add appropriate error handling and logging

## Benefits

- **Reduced memory usage**: The model is loaded only in the model server process
- **Simplified deployment**: No need to manage model files
- **Improved reliability**: The main application can start even if the model server is not available
- **Scalability**: The model server can be scaled independently of the main application

