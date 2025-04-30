#!/bin/bash

# Start the model server and main application together
# This script starts both the external model server and the main application

# Set environment variables
export USE_EXTERNAL_MODEL_SERVER=true
export MODEL_SERVER_URL=http://localhost:5000
export MINIMIZE_MEMORY_USAGE=true

# Start the model server in the background
echo "Starting model server..."
python backend/model_server.py &
MODEL_SERVER_PID=$!

# Wait for model server to start
echo "Waiting for model server to start..."
sleep 5

# Start the main application
echo "Starting main application..."
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 10000

# Clean up model server when main app exits
kill $MODEL_SERVER_PID

