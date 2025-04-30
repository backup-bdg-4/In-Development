import os
import subprocess
import sys
from download_model import check_model

def run_backend():
    """Run the backend server locally"""
    # First, check if the model exists and is valid
    if not check_model():
        print("\nERROR: CoreML model verification failed.")
        print("Make sure the model file is correctly placed in the repository.")
        print("The model should be managed using Git LFS.")
        print("\nIf you're a developer with access to the model file:")
        print("1. Copy the BERTSQUADFP16.mlmodel file to backend/app/model/")
        print("2. Make sure the file is properly tracked by Git LFS")
        print("\nExiting...")
        return False
    
    # Run the backend server
    try:
        print("\nModel verification successful!")
        print("Starting backend server...")
        subprocess.run(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
        return True
    
    except Exception as e:
        print(f"Error starting backend server: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_backend()
    sys.exit(0 if success else 1)

