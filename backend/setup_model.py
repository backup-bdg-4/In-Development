#!/usr/bin/env python3
"""
Setup script for verifying and configuring the CoreML model.
This script helps users verify that the model is correctly installed.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_path():
    """Get the path to where the model should be located"""
    # Define the model directory and file path
    model_data_path = os.environ.get('MODEL_DATA_PATH', None)
    if model_data_path:
        model_path = os.path.join(model_data_path, "BERTSQUADFP16.mlmodel")
    else:
        model_path = os.path.join(os.path.dirname(__file__), "app", "model", "BERTSQUADFP16.mlmodel")
    return model_path

def check_model_file():
    """Check if the model file exists at the expected location"""
    model_path = get_model_path()
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        logger.info(f"✅ Model file found at {model_path} ({file_size:.2f} MB)")
        return True
    else:
        logger.error(f"❌ Model file not found at {model_path}")
        return False

def verify_model():
    """Verify the model by running check_model"""
    try:
        # Import check_model from download_model.py 
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from download_model import check_model
        
        # Check the model
        logger.info("Verifying model...")
        if check_model():
            logger.info("✅ Model verification successful")
            return True
        else:
            logger.error("❌ Model verification failed")
            return False
    
    except ImportError as e:
        logger.error(f"Error importing check_model: {str(e)}")
        return False
    
    except Exception as e:
        logger.error(f"Error verifying model: {str(e)}")
        return False

def setup_git_lfs():
    """Check and setup Git LFS tracking for model files"""
    try:
        import subprocess
        
        # Check if Git LFS is installed
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ Git LFS not installed. Please install it from https://git-lfs.github.com")
            return False
            
        logger.info(f"Git LFS version: {result.stdout.strip()}")
        
        # Check if .gitattributes exists and has the right configuration
        gitattributes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".gitattributes")
        model_tracked = False
        
        if os.path.exists(gitattributes_path):
            with open(gitattributes_path, 'r') as f:
                content = f.read()
                if "*.mlmodel filter=lfs" in content:
                    model_tracked = True
                    logger.info("✅ .mlmodel files are already tracked by Git LFS")
        
        if not model_tracked:
            # Add tracking for .mlmodel files
            subprocess.run(["git", "lfs", "track", "*.mlmodel"], check=True)
            logger.info("✅ Added Git LFS tracking for .mlmodel files")
            
            # Commit the .gitattributes file if it was changed
            if os.path.exists(".git"):
                subprocess.run(["git", "add", ".gitattributes"], check=True)
                logger.info("✅ Added .gitattributes to git")
                
        return True
    
    except Exception as e:
        logger.error(f"Error setting up Git LFS: {str(e)}")
        return False

def create_model_directory():
    """Create the model directory if it doesn't exist"""
    model_path = get_model_path()
    model_dir = os.path.dirname(model_path)
    
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"✅ Created model directory at {model_dir}")
        else:
            logger.info(f"✅ Model directory already exists at {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating model directory: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup script for verifying the CoreML model")
    parser.add_argument("--model-path", help="Custom path to the CoreML model file")
    parser.add_argument("--setup-lfs", action="store_true", help="Setup Git LFS tracking for model files")
    parser.add_argument("--verify-only", action="store_true", help="Only verify the model")
    
    args = parser.parse_args()
    
    print("\n===== CoreML Model Setup Tool =====\n")
    
    # Setup Git LFS if requested
    if args.setup_lfs:
        print(f"\n----- Setting up Git LFS -----")
        if setup_git_lfs():
            print("✅ Git LFS setup successful")
        else:
            print("❌ Failed to setup Git LFS")
            print("\nPlease install Git LFS from https://git-lfs.github.com")
    
    # Create model directory
    if not args.verify_only:
        print(f"\n----- Setting up model directory -----")
        if create_model_directory():
            print("✅ Model directory setup successful")
        else:
            print("❌ Failed to setup model directory")
            return
    
    # Check if model file exists
    print(f"\n----- Checking model file -----")
    if check_model_file():
        print("✅ Model file exists")
    else:
        print("❌ Model file not found")
        print(f"\nThe model file should be placed at: {get_model_path()}")
        print("\nThis model file should be tracked using Git LFS.")
        print("If you have access to the model file, please copy it to the location above.")
        return
    
    # Verify model
    print(f"\n----- Verifying model -----")
    if verify_model():
        print("✅ Model verification successful")
    else:
        print("❌ Model verification failed")
        print("\nThe model file exists but may be corrupted or incompatible.")
        return
    
    print("\n===== Setup Complete =====")
    print("The CoreML model has been successfully verified.")
    print("You can now run the backend server with: python run.py")

if __name__ == "__main__":
    main()

