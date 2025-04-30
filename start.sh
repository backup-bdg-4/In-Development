#!/bin/bash
# Main startup script for the application

echo "===== SETUP PHASE ====="

# Create essential directories
mkdir -p /tmp/model
mkdir -p /tmp/jupyter

# Set environment variables
export USE_JUPYTER_MODEL_SERVER=true
export MODEL_DATA_PATH=/tmp/model
export MINIMIZE_MEMORY_USAGE=true

echo "Environment variables set:"
echo "USE_JUPYTER_MODEL_SERVER=$USE_JUPYTER_MODEL_SERVER"
echo "MODEL_DATA_PATH=$MODEL_DATA_PATH"
echo "MINIMIZE_MEMORY_USAGE=$MINIMIZE_MEMORY_USAGE"

# Check if model exists in /tmp/model
if [ -f "/tmp/model/BERTSQUADFP16.mlmodel" ]; then
  echo "✅ Model already exists in /tmp/model"
else
  echo "⚠️ Model not found in /tmp/model, checking other locations..."
  
  # Check various locations for the model
  if [ -f "backend/app/model/BERTSQUADFP16.mlmodel" ]; then
    echo "✅ Found model in backend/app/model directory"
    cp backend/app/model/BERTSQUADFP16.mlmodel /tmp/model/
  elif [ -f "backend/BERTSQUADFP16.mlmodel" ]; then
    echo "✅ Found model in backend directory"
    cp backend/BERTSQUADFP16.mlmodel /tmp/model/
  elif [ -f "BERTSQUADFP16.mlmodel" ]; then
    echo "✅ Found model in repository root"
    cp BERTSQUADFP16.mlmodel /tmp/model/
  else
    echo "❌ Model not found in any standard location"
    echo "Please place the BERTSQUADFP16.mlmodel file in one of the following locations:"
    echo "- /tmp/model/"
    echo "- backend/app/model/"
    echo "- backend/"
    echo "- repository root"
    exit 1
  fi
fi

# Check model file size
MODEL_SIZE=$(du -h /tmp/model/BERTSQUADFP16.mlmodel | cut -f1)
echo "Model size: $MODEL_SIZE"
echo "✅ Model successfully stored in /tmp/model"

# Install required packages
echo "Installing required packages..."
pip install -r backend/requirements.txt
pip install jupyter ipywidgets

echo "===== STARTING APPLICATION ====="
echo "Jupyter model server enabled for memory efficiency"

# Start the application
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}

