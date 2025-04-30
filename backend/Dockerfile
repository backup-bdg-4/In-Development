FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create all possible model directories
RUN mkdir -p /app/app/model
RUN mkdir -p /tmp/model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DATA_PATH=/app/app/model
ENV BACKDOOR_MODEL_SEARCH_PATHS="/app/app/model,/app,/tmp/model,/app/backend"
ENV ENVIRONMENT=production

# Create a script to find and set up the model
RUN echo '#!/bin/sh \n\
echo "Setting up the model..." \n\
\n\
# Look for the model file in these locations \n\
FOUND=0 \n\
for location in "/app/BERTSQUADFP16.mlmodel" "/app/backend/BERTSQUADFP16.mlmodel" "/app/app/model/BERTSQUADFP16.mlmodel" "/tmp/model/BERTSQUADFP16.mlmodel"; do \n\
  if [ -f "$location" ]; then \n\
    echo "Found existing model at $location" \n\
    FOUND=1 \n\
    break \n\
  fi \n\
done \n\
\n\
# If model not found in any location, try to use download_model.py \n\
if [ $FOUND -eq 0 ]; then \n\
  echo "Model not found in any location, running download_model.py" \n\
  python /app/download_model.py \n\
fi \n\
\n\
# Copy to multiple locations for redundancy \n\
if [ -f "/app/BERTSQUADFP16.mlmodel" ]; then \n\
  cp /app/BERTSQUADFP16.mlmodel /app/app/model/ \n\
  cp /app/BERTSQUADFP16.mlmodel /tmp/model/ \n\
elif [ -f "/app/backend/BERTSQUADFP16.mlmodel" ]; then \n\
  cp /app/backend/BERTSQUADFP16.mlmodel /app/app/model/ \n\
  cp /app/backend/BERTSQUADFP16.mlmodel /tmp/model/ \n\
elif [ -f "/app/app/model/BERTSQUADFP16.mlmodel" ]; then \n\
  cp /app/app/model/BERTSQUADFP16.mlmodel /tmp/model/ \n\
elif [ -f "/tmp/model/BERTSQUADFP16.mlmodel" ]; then \n\
  cp /tmp/model/BERTSQUADFP16.mlmodel /app/app/model/ \n\
fi \n\
\n\
echo "Model setup complete. Checking locations:" \n\
ls -la /app/app/model/ || echo "No models in /app/app/model/" \n\
ls -la /tmp/model/ || echo "No models in /tmp/model/" \n\
ls -la /app/ | grep .mlmodel || echo "No models in /app/" \n\
\n\
' > /app/setup_model.sh

RUN chmod +x /app/setup_model.sh

# Expose the port
EXPOSE 8000

# Run the model setup and start the application
CMD ["/bin/sh", "-c", "/app/setup_model.sh && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
