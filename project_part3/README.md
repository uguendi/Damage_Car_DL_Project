# Car Damage Detection and Severity Analysis System

This project is a deep learning-based system designed to detect vehicle damage and classify its severity. It employs a two-stage pipeline: first, a YOLOv8 model detects the location of damages on a vehicle, and second, a ResNet18 model classifies the detected damage into one of three severity levels: High, Medium, or Low.

## Project Overview

The system is capable of processing individual images via user-friendly web interfaces or handling large datasets through batch processing scripts.

- **Object Detection (YOLOv8)**: Locates damages such as scratches, dents, and smashes.
- **Severity Classification (ResNet18)**: Analyzes the specific damaged area to determine the severity.
- **Visual Output**: Generates annotated images with color-coded bounding boxes representing severity levels (Red for High, Orange for Medium, Yellow for Low).

## Data and Model Setup

**IMPORTANT:** Due to large file sizes, the primary datasets and training data are hosted on Google Drive. 

**[Google Drive Link - Deep Learning Dataset](https://drive.google.com/drive/u/1/folders/1-wpZjvE_s-5yX4lRyXxAlizcbOUU0WHp)**

### Steps to Setup:
1.  Clone this repository.
2.  Download the data folders from the Drive link above.
3.  Place the following folders into the **`project_part3/`** directory (this directory where you found this README):
    *   `archive-2/` (Large dataset for batch processing)
    *   `dataset_raw_bing/` (Raw images for training)
    *   `severity_data/` (Labeled training data)
4.  The pretrained models (`best.pt` and `severity_model_resnet18.pth`) and a small sample dataset (`In_Person/`) are already included in this directory for your convenience.

## Directory Structure and File Descriptions

The project logic is provided as **Jupyter Notebooks** in the `Jupyter Notebook/` folder. The Python scripts descriptions below refer to the logic contained within these notebooks.

### Web Applications

*   **`Jupyter Notebook/app.ipynb`** (corresponds to `app.py`)
    *   **Framework**: Gradio
    *   **Functionality**: Allows users to upload an image of a car. The app processes the image and displays the result with bounding boxes and a text summary of detected damages.

*   **`Jupyter Notebook/app_streamlit.ipynb`** (corresponds to `app_streamlit.py`)
    *   **Framework**: Streamlit
    *   **Functionality**: A modern web interface for damage detection.

### Batch Processing Pipelines

*   **`Jupyter Notebook/test_pipeline.ipynb`**
    *   **Purpose**: General-purpose batch processing for testing the full pipeline.
    *   **Input Source**: `In_Person/` (Included in repo as sample data).
    *   **Output Destination**: `In_Person_Results/` (Generated locally).

*   **`Jupyter Notebook/predict_archive2.ipynb`**
    *   **Purpose**: Optimized for processing large datasets.
    *   **Input Source**: `archive-2/image/` (Download from Drive).
    *   **Output Destination**: `archive2_results/` (Generated locally).

### Data Preparation and Utilities

*   **`Jupyter Notebook/crop_damages.ipynb`**
    *   **Purpose**: Extracts damaged regions from full car images.
    *   **Input Source**: `dataset_raw_bing/` (Download from Drive).

*   **`Jupyter Notebook/label_cropped_dataset.ipynb`**
    *   **Purpose**: Auto-labels cropped images using the severity model.
    *   **Input Source**: `cropped_damages/`.

### Training and Models

*   **`Jupyter Notebook/train_severity.ipynb`**
    *   **Purpose**: Scripts to train the ResNet18 severity classification model.
    *   **Input Data**: `severity_data/` (Download from Drive).

*   **`best.pt`**: Pre-trained YOLOv8 model (Included in Repo).
*   **`severity_model_resnet18.pth`**: Trained ResNet18 model (Included in Repo).

## Installation and Requirements

To run this project, you need Python installed along with the following dependencies:

```bash
pip install torch torchvision ultralytics opencv-python gradio streamlit pillow numpy tqdm scikit-learn matplotlib seaborn
```

## Usage

This project is now structured as Jupyter Notebooks.

1.  Open the terminal and navigate to the `project_part3` directory.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
3.  Open the desired notebook from the `Jupyter Notebook/` folder (e.g., `Jupyter Notebook/test_pipeline.ipynb`).
4.  Run the first cell (Setup Cell) to initialize the correct paths.
5.  Run the subsequent cells to execute the code.

