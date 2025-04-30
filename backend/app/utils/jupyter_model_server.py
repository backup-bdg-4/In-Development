"""
Jupyter-based model server for memory-efficient model loading.
This module provides a way to load the model in a separate process
and access it via HTTP, reducing memory usage in the main application.
"""

import os
import sys
import logging
import json
import time
import subprocess
import requests
import threading
import socket
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global variables
JUPYTER_SERVER_PORT = 8888
JUPYTER_SERVER_PROCESS = None
JUPYTER_SERVER_URL = f"http://localhost:{JUPYTER_SERVER_PORT}"
MODEL_SERVER_READY = False
MODEL_SERVER_NOTEBOOK = None

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int = 8888, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def create_model_notebook(model_path: str) -> str:
    """
    Create a Jupyter notebook that loads the model and provides prediction endpoints.
    
    Args:
        model_path: Path to the CoreML model file
        
    Returns:
        Path to the created notebook
    """
    notebook_dir = "/tmp/jupyter"
    os.makedirs(notebook_dir, exist_ok=True)
    
    notebook_path = os.path.join(notebook_dir, "model_server.ipynb")
    
    # Create notebook content
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Model Server Notebook\n",
                    "# This notebook loads the CoreML model and provides prediction endpoints\n",
                    "import os\n",
                    "import sys\n",
                    "import json\n",
                    "import time\n",
                    "import logging\n",
                    "import coremltools as ct\n",
                    "from IPython.display import display, HTML\n",
                    "\n",
                    "# Configure logging\n",
                    "logging.basicConfig(level=logging.INFO)\n",
                    "logger = logging.getLogger('model_server')\n",
                    "\n",
                    f"# Model path\n",
                    f"MODEL_PATH = '{model_path}'\n",
                    "\n",
                    "# Global model variable\n",
                    "model = None\n",
                    "\n",
                    "print(f'Model server notebook initialized. Will load model from {MODEL_PATH}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load the model\n",
                    "def load_model():\n",
                    "    global model\n",
                    "    \n",
                    "    if model is not None:\n",
                    "        print('Model already loaded')\n",
                    "        return True\n",
                    "    \n",
                    "    try:\n",
                    "        print(f'Loading model from {MODEL_PATH}...')\n",
                    "        start_time = time.time()\n",
                    "        \n",
                    "        # Check if file exists\n",
                    "        if not os.path.exists(MODEL_PATH):\n",
                    "            print(f'Error: Model file not found at {MODEL_PATH}')\n",
                    "            return False\n",
                    "        \n",
                    "        # Load the model\n",
                    "        model = ct.models.MLModel(MODEL_PATH)\n",
                    "        load_time = time.time() - start_time\n",
                    "        \n",
                    "        print(f'Model loaded successfully in {load_time:.2f} seconds')\n",
                    "        return True\n",
                    "    except Exception as e:\n",
                    "        print(f'Error loading model: {str(e)}')\n",
                    "        return False\n",
                    "\n",
                    "# Load the model\n",
                    "load_model()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create prediction function\n",
                    "def predict(query_text, passage_text):\n",
                    "    global model\n",
                    "    \n",
                    "    if model is None:\n",
                    "        success = load_model()\n",
                    "        if not success:\n",
                    "            return {'error': 'Model not loaded'}\n",
                    "    \n",
                    "    try:\n",
                    "        # Prepare input for the model\n",
                    "        model_input = {\n",
                    "            'query_text': query_text,\n",
                    "            'passage_text': passage_text\n",
                    "        }\n",
                    "        \n",
                    "        # Make prediction\n",
                    "        prediction = model.predict(model_input)\n",
                    "        \n",
                    "        # Extract start and end indices\n",
                    "        start_idx = int(prediction.get('start_index', 0))\n",
                    "        end_idx = int(prediction.get('end_index', 0))\n",
                    "        \n",
                    "        # Extract the answer from the context\n",
                    "        answer = passage_text[start_idx:end_idx+1] if start_idx <= end_idx else ''\n",
                    "        \n",
                    "        # Calculate confidence score\n",
                    "        confidence = float(prediction.get('confidence', 0.0))\n",
                    "        \n",
                    "        # Return the response\n",
                    "        return {\n",
                    "            'answer': answer,\n",
                    "            'confidence': confidence,\n",
                    "            'start_index': start_idx,\n",
                    "            'end_index': end_idx\n",
                    "        }\n",
                    "    except Exception as e:\n",
                    "        return {'error': f'Prediction error: {str(e)}'}\n",
                    "\n",
                    "# Test the prediction function\n",
                    "test_result = predict('What is AI?', 'Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.')\n",
                    "print('Test prediction result:')\n",
                    "print(test_result)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create API endpoints using IPython widgets\n",
                    "from ipywidgets import widgets\n",
                    "from IPython.display import display\n",
                    "\n",
                    "# Create a function to handle prediction requests\n",
                    "def handle_predict(query, context):\n",
                    "    result = predict(query, context)\n",
                    "    return json.dumps(result)\n",
                    "\n",
                    "# Create a function to check if the model is loaded\n",
                    "def is_model_loaded():\n",
                    "    global model\n",
                    "    return model is not None\n",
                    "\n",
                    "# Display server status\n",
                    "display(HTML('<h3>Model Server Ready</h3>'))\n",
                    "print('Model server is ready to accept requests')\n",
                    "print(f'Model loaded: {is_model_loaded()}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f)
    
    logger.info(f"Created model server notebook at {notebook_path}")
    return notebook_path

def start_jupyter_server(model_path: str) -> bool:
    """
    Start a Jupyter notebook server to host the model.
    
    Args:
        model_path: Path to the CoreML model file
        
    Returns:
        True if server started successfully, False otherwise
    """
    global JUPYTER_SERVER_PROCESS, JUPYTER_SERVER_PORT, JUPYTER_SERVER_URL, MODEL_SERVER_NOTEBOOK
    
    try:
        # Find an available port
        JUPYTER_SERVER_PORT = find_available_port()
        JUPYTER_SERVER_URL = f"http://localhost:{JUPYTER_SERVER_PORT}"
        
        # Create the model notebook
        notebook_path = create_model_notebook(model_path)
        MODEL_SERVER_NOTEBOOK = notebook_path
        
        # Start Jupyter server
        cmd = [
            sys.executable, "-m", "jupyter", "notebook",
            "--no-browser",
            f"--port={JUPYTER_SERVER_PORT}",
            "--ip=0.0.0.0",
            "--NotebookApp.token=''",
            "--NotebookApp.password=''"
        ]
        
        logger.info(f"Starting Jupyter server with command: {' '.join(cmd)}")
        
        # Start the process
        JUPYTER_SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{JUPYTER_SERVER_URL}/api/contents")
                if response.status_code == 200:
                    logger.info(f"Jupyter server started successfully at {JUPYTER_SERVER_URL}")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        
        logger.error("Failed to start Jupyter server within timeout")
        return False
    
    except Exception as e:
        logger.error(f"Error starting Jupyter server: {str(e)}")
        return False

def execute_notebook_cell(notebook_path: str, cell_index: int) -> Dict[str, Any]:
    """
    Execute a cell in the notebook.
    
    Args:
        notebook_path: Path to the notebook
        cell_index: Index of the cell to execute
        
    Returns:
        Dictionary with execution result
    """
    try:
        # Get the notebook content
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Execute the cell using the Jupyter API
        response = requests.post(
            f"{JUPYTER_SERVER_URL}/api/contents/{os.path.basename(notebook_path)}/checkpoints",
            json={"cells": notebook["cells"]}
        )
        
        if response.status_code != 201:
            logger.error(f"Failed to create checkpoint: {response.text}")
            return {"success": False, "error": "Failed to create checkpoint"}
        
        # Execute the cell
        response = requests.post(
            f"{JUPYTER_SERVER_URL}/api/cells/{os.path.basename(notebook_path)}/{cell_index}/execute",
            json={}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to execute cell: {response.text}")
            return {"success": False, "error": "Failed to execute cell"}
        
        return {"success": True, "result": response.json()}
    
    except Exception as e:
        logger.error(f"Error executing notebook cell: {str(e)}")
        return {"success": False, "error": str(e)}

def initialize_model_server(model_path: str) -> bool:
    """
    Initialize the model server by starting Jupyter and executing the notebook.
    
    Args:
        model_path: Path to the CoreML model file
        
    Returns:
        True if initialization was successful, False otherwise
    """
    global MODEL_SERVER_READY, MODEL_SERVER_NOTEBOOK
    
    # Start Jupyter server
    server_started = start_jupyter_server(model_path)
    if not server_started:
        logger.error("Failed to start Jupyter server")
        return False
    
    # Execute the notebook cells to load the model
    for cell_index in range(4):  # Execute all 4 cells
        result = execute_notebook_cell(MODEL_SERVER_NOTEBOOK, cell_index)
        if not result["success"]:
            logger.error(f"Failed to execute cell {cell_index}: {result.get('error', 'Unknown error')}")
            return False
    
    # Mark server as ready
    MODEL_SERVER_READY = True
    logger.info("Model server initialized successfully")
    return True

def predict_with_jupyter(query: str, context: str) -> Dict[str, Any]:
    """
    Make a prediction using the Jupyter-hosted model.
    
    Args:
        query: Query text
        context: Context text
        
    Returns:
        Dictionary with prediction result
    """
    global MODEL_SERVER_READY, MODEL_SERVER_NOTEBOOK
    
    if not MODEL_SERVER_READY:
        logger.error("Model server not ready")
        return {"error": "Model server not ready"}
    
    try:
        # Create a cell with the prediction code
        prediction_code = f"""
        result = predict({json.dumps(query)}, {json.dumps(context)})
        print(json.dumps(result))
        """
        
        # Add the cell to the notebook
        with open(MODEL_SERVER_NOTEBOOK, 'r') as f:
            notebook = json.load(f)
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [prediction_code]
        })
        
        with open(MODEL_SERVER_NOTEBOOK, 'w') as f:
            json.dump(notebook, f)
        
        # Execute the cell
        result = execute_notebook_cell(MODEL_SERVER_NOTEBOOK, len(notebook["cells"]) - 1)
        if not result["success"]:
            logger.error(f"Failed to execute prediction cell: {result.get('error', 'Unknown error')}")
            return {"error": f"Failed to execute prediction: {result.get('error', 'Unknown error')}"}
        
        # Parse the output
        output = result["result"].get("outputs", [])
        for out in output:
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                try:
                    prediction_result = json.loads(out["text"].strip())
                    return prediction_result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse prediction result: {out['text']}")
        
        return {"error": "No valid prediction result found in output"}
    
    except Exception as e:
        logger.error(f"Error making prediction with Jupyter: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

async def predict_with_jupyter_server(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction using the Jupyter-hosted model with async support.
    
    Args:
        model_input: Dictionary with query_text and passage_text
        
    Returns:
        Dictionary with prediction result
    """
    global MODEL_SERVER_READY
    
    if not MODEL_SERVER_READY:
        logger.error("Model server not ready")
        return {"error": "Model server not ready"}
    
    # Extract query and context from model_input
    query = model_input.get('query_text', '')
    context = model_input.get('passage_text', '')
    
    # Use the synchronous function for now
    # In a real implementation, this would make an async HTTP request to the Jupyter server
    return predict_with_jupyter(query, context)

def shutdown_jupyter_server():
    """Shutdown the Jupyter server."""
    global JUPYTER_SERVER_PROCESS, MODEL_SERVER_READY
    
    if JUPYTER_SERVER_PROCESS is not None:
        logger.info("Shutting down Jupyter server")
        JUPYTER_SERVER_PROCESS.terminate()
        JUPYTER_SERVER_PROCESS = None
        MODEL_SERVER_READY = False

def is_jupyter_server_running() -> bool:
    """Check if the Jupyter server is running."""
    global JUPYTER_SERVER_PROCESS, MODEL_SERVER_READY
    
    if JUPYTER_SERVER_PROCESS is None:
        return False
    
    return JUPYTER_SERVER_PROCESS.poll() is None and MODEL_SERVER_READY