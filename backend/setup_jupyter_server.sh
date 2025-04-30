#!/bin/bash
# Setup script for Jupyter model server

# Create necessary directories
mkdir -p /tmp/model
mkdir -p /tmp/jupyter

# Set environment variables
export USE_JUPYTER_MODEL_SERVER=true
export MODEL_DATA_PATH=/tmp/model
export MINIMIZE_MEMORY_USAGE=true

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
    exit 1
  fi
fi

# Install required packages for Jupyter
pip install jupyter ipywidgets

# Run the test script to verify Jupyter model server
echo "Testing Jupyter model server..."
cd backend
python test_jupyter_model.py

# If the test was successful, print success message
if [ $? -eq 0 ]; then
  echo "✅ Jupyter model server setup completed successfully"
  echo "You can now start the application with:"
  echo "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 10000"
else
  echo "❌ Jupyter model server setup failed"
  exit 1
fi

