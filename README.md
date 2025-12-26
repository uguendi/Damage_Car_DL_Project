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
3.  Place the following folders into the root directory of the project:
    *   `archive-2/` (Large dataset for batch processing)
    *   `dataset_raw_bing/` (Raw images for training)
    *   `severity_data/` (Labeled training data)
4.  The pretrained models (`best.pt` and `severity_model_resnet18.pth`) and a small sample dataset (`In_Person/`) are already included in this repository for your convenience.

## Directory Structure and File Descriptions

### Web Applications

These files provide graphical user interfaces for interacting with the models.

*   **`app.py`**
    *   **Framework**: Gradio
    *   **Functionality**: Allows users to upload an image of a car. The app processes the image and displays the result with bounding boxes and a text summary of detected damages.
    *   **Input**: User-uploaded image.
    *   **Output**: Annotated image and severity summary displayed in the browser.

*   **`app_streamlit.py`**
    *   **Framework**: Streamlit
    *   **Functionality**: A modern web interface for damage detection. It includes a legend for severity colors and counts the total number of damages found.
    *   **Input**: User-uploaded image.
    *   **Output**: Annotated image with severity legend.

### Batch Processing Pipelines

These scripts are designed to process folders of images automatically.

*   **`test_pipeline.py`**
    *   **Purpose**: General-purpose batch processing for testing the full pipeline.
    *   **Input Source**: `In_Person/` (Included in repo as sample data).
    *   **Output Destination**: `In_Person_Results/` (Generated locally).
    *   **Process**: Detects damages -> Crops -> Classifies Severity -> Saves original image with annotations.

*   **`predict_archive2.py`**
    *   **Purpose**: Optimized for processing large datasets.
    *   **Input Source**: `archive-2/image/` (Download from Drive).
    *   **Output Destination**: `archive2_results/` (Generated locally).

### Data Preparation and Utilities

*   **`crop_damages.py`**
    *   **Purpose**: Extracts damaged regions from full car images.
    *   **Input Source**: `dataset_raw_bing/` (Download from Drive).
    *   **Output Destination**: `cropped_damages/`.

*   **`label_cropped_dataset.py`**
    *   **Purpose**: Auto-labels cropped images using the severity model.
    *   **Input Source**: `cropped_damages/`.
    *   **Output Destination**: `cropped_damages_results/`.

### Training and Models

*   **`train_severity.py`**
    *   **Purpose**: Scripts to train the ResNet18 severity classification model.
    *   **Input Data**: `severity_data/` (Download from Drive).
    *   **Output**: Saves the best model weights to `severity_model_resnet18.pth`.

*   **`best.pt`**: Pre-trained YOLOv8 model (Included in Repo).
*   **`severity_model_resnet18.pth`**: Trained ResNet18 model (Included in Repo).

## Installation and Requirements

To run this project, you need Python installed along with the following dependencies:

```bash
pip install torch torchvision ultralytics opencv-python gradio streamlit pillow numpy tqdm scikit-learn matplotlib seaborn
```

## Usage

### Running the Web Apps

To launch the Gradio interface:
```bash
python app.py
```

To launch the Streamlit interface:
```bash
streamlit run app_streamlit.py
```

### Running Batch Processing

To process the "In_Person" folder:
```bash
python test_pipeline.py
```

To process the large archive:
```bash
python predict_archive2.py
```

### Training the Model

If you have a dataset arranged in `severity_data/` and want to retrain the classifier:
```bash
python train_severity.py
```
