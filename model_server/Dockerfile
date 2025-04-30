FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directory for the model
RUN mkdir -p /app/model

# Environment variables
ENV MODEL_PATH=/app/model/BERTSQUADFP16.mlmodel
ENV MODEL_SERVER_API_KEY=your-api-key-here

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

