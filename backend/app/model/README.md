# CoreML Model Directory

This directory is where the CoreML model file (`BERTSQUADFP16.mlmodel`) should be placed.

## Model File

The CoreML model file should be named `BERTSQUADFP16.mlmodel` and placed directly in this directory.
This model is stored using Git LFS to handle its large size properly.

## How to Get the Model

If you don't see the model file here, you can:

1. Make sure you have Git LFS installed: https://git-lfs.github.com
2. Pull the repository with Git LFS enabled: `git lfs pull`
3. If you have the model file separately, copy it to this directory

## Usage

Once the model is correctly placed here, you can:

1. Run the model verification: `python setup_model.py`
2. Start the backend server: `python run.py`

The application will automatically use the model file in this location.
