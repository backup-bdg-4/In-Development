# Jupyter Model Server for Memory Efficiency

## Overview

The Jupyter Model Server is a solution to the memory usage issues encountered when running the application on Render's Free Tier with its 512MB memory limit. This approach runs the CoreML model in a separate process (Jupyter notebook) to isolate its memory footprint from the main FastAPI application.

## How It Works

1. **Process Isolation**: The model is loaded in a separate Jupyter notebook process, which prevents the model's memory usage from affecting the main FastAPI application.

2. **On-Demand Loading**: The model is only loaded when needed for predictions, and the Jupyter server can be configured to unload the model between requests if extreme memory saving is required.

3. **Single Model Instance**: Only one copy of the model is maintained, eliminating redundant copies that were previously causing memory issues.

4. **Inter-Process Communication**: The FastAPI application communicates with the Jupyter server to make predictions, using a simple HTTP-based protocol.

## Configuration

The Jupyter Model Server can be enabled with the following environment variables:

- `USE_JUPYTER_MODEL_SERVER=true` - Enables the Jupyter model server approach
- `MINIMIZE_MEMORY_USAGE=true` - Enables memory-saving optimizations
- `MODEL_DATA_PATH=/tmp/model` - Sets the location for the model file

## Implementation Details

### Components

1. **jupyter_model_server.py**: Contains the code to start, manage, and communicate with the Jupyter server.

2. **model_server.ipynb**: A Jupyter notebook that loads the model and provides prediction endpoints.

3. **Main Application Integration**: The FastAPI application is modified to use the Jupyter server for predictions when enabled.

### Memory Benefits

- **Isolated Memory Space**: The model's memory footprint is isolated in a separate process.
- **Garbage Collection**: Memory is explicitly freed after predictions.
- **Single Model Copy**: Only one copy of the model is maintained.
- **On-Demand Loading**: The model is only loaded when needed.

## Testing

A test script is provided to verify the Jupyter model server functionality:

```bash
python backend/test_jupyter_model.py
```

## Limitations

1. **Startup Time**: The Jupyter server takes a few seconds to start up.
2. **Complexity**: This approach adds complexity to the application architecture.
3. **Dependencies**: Requires additional dependencies (Jupyter, notebook, ipywidgets).

## Fallback Mechanism

If the Jupyter server is not available or fails to initialize, the application will fall back to the standard model loading approach.