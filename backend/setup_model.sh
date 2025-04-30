#!/bin/sh 
echo "===== MODEL SETUP ====="

# Ensure target directories exist
mkdir -p /tmp/model
mkdir -p /app/app/model

# Set the primary model directory to /tmp/model for Render Free Tier compatibility
export MODEL_DATA_PATH=/tmp/model

echo "MODEL_DATA_PATH set to $MODEL_DATA_PATH"

# Prioritize model locations for checking
FOUND=0
MODEL_PATH=""

# Try to find the model in standard locations
for location in "/tmp/model/BERTSQUADFP16.mlmodel" "/app/app/model/BERTSQUADFP16.mlmodel" "/app/backend/BERTSQUADFP16.mlmodel" "/app/BERTSQUADFP16.mlmodel"; do 
  if [ -f "$location" ]; then 
    echo "✅ Found existing model at $location"
    FOUND=1
    MODEL_PATH="$location"
    break 
  fi 
done 

# If model found, make copies for redundancy
if [ $FOUND -eq 1 ]; then
  # Always ensure a copy in /tmp/model (primary location for Render)
  if [ "$MODEL_PATH" != "/tmp/model/BERTSQUADFP16.mlmodel" ]; then
    echo "Copying model to /tmp/model for Render compatibility"
    cp "$MODEL_PATH" /tmp/model/BERTSQUADFP16.mlmodel
  fi
  
  # Also copy to app/model if not already there
  if [ "$MODEL_PATH" != "/app/app/model/BERTSQUADFP16.mlmodel" ]; then
    echo "Copying model to /app/app/model for local access"
    cp "$MODEL_PATH" /app/app/model/BERTSQUADFP16.mlmodel
  fi
# If model not found in any location, try to download it
else
  echo "⚠️ Model not found in any location, running download_model.py" 
  python /app/download_model.py
  
  # Check if download was successful
  if [ -f "/app/app/model/BERTSQUADFP16.mlmodel" ]; then
    echo "Download successful to /app/app/model"
    cp /app/app/model/BERTSQUADFP16.mlmodel /tmp/model/
  elif [ ! -f "/tmp/model/BERTSQUADFP16.mlmodel" ]; then
    echo "⚠️ Download may have failed. Trying direct download as fallback."
    # Last resort - direct download to /tmp/model
    curl -L "https://www.dropbox.com/scl/fi/w4iclrvil6vh39mg6j7pl/BERTSQUADFP16.mlmodel?rlkey=vbrr9jjvsam1xg9i4i19pkdra&st=ho9dyrm6&dl=1" -o /tmp/model/BERTSQUADFP16.mlmodel
  fi
fi

# Final check and file size info
echo "Final model location check:"
if [ -f "/tmp/model/BERTSQUADFP16.mlmodel" ]; then
  echo "✅ MODEL READY in /tmp/model ($(du -h /tmp/model/BERTSQUADFP16.mlmodel | cut -f1))"
  # Set permissions to ensure readability
  chmod 644 /tmp/model/BERTSQUADFP16.mlmodel
else
  echo "❌ ERROR: Model file not found in /tmp/model"
fi

echo "===== MODEL SETUP COMPLETE ====="
