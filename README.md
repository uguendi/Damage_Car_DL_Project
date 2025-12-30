# Deep Learning-Based Vehicle Damage Region Detection and Severity Classification

## Project Overview
This project provides a comprehensive deep learning solution for automated vehicle damage assessment. The system is designed to traverse the entire pipeline from raw data acquisition to end-user application. It utilizes **YOLOv8** for precise damage localization and **ResNet18** for classifying damage severity into three levels: **High**, **Medium**, and **Low**.

## Project Structure
The project is organized into three distinct phases, each handling a critical stage of the pipeline:

### [Phase 1: Dataset Creation](./project_part_1)
**Goal:** Build a robust and diverse training dataset.
*   **Hybrid Approach:** Combines labeled benchmark data (CarDD & VehiDE), high-volume web-scraped images (Bing), and manual field data.
*   **Key Tool:** `collect_damage_data.py` (Custom web scraper with multi-threading).
*   **Output:** A diverse raw image dataset ready for annotation.

### [Phase 2: Damage Detection (YOLOv8)](./project_part_2)
**Goal:** Train a model to locate damaged regions on a vehicle.
*   **Model:** YOLOv8 (State-of-the-art object detection).
*   **Features:**
    *   Data preprocessing pipelines.
    *   Hyperparameter optimization (`Final_Optimization&Analysis.ipynb`).
    *   Classes: `dent`, `scratch`, `processed_damage`, etc.

### [Phase 3: Severity Analysis System](./project_part3)
**Goal:** A complete end-to-end system for damage analysis and user interaction.
*   **Pipeline:**
    1.  **Detection:** Identify damage location (YOLOv8).
    2.  **Classification:** Determine severity (ResNet18).
*   **Applications:**
    *   **Gradio App:** Interactive drag-and-drop web interface.
    *   **Streamlit App:** Modern dashboard for damage assessment.
    *   **Batch Processing:** Automated pipeline for large datasets.

## Getting Started

### 1. Requirements
*   Python 3.8+
*   Please check the individual `requirements.txt` files in each project folder for specific dependencies (e.g., `ultralytics`, `gradio`, `bing-image-downloader`).

### 2. Workflow
1.  **Acquire Data (Phase 1):** Use the scripts in `project_part_1` to gather your initial raw dataset if needed.
2.  **Train Detector (Phase 2):** Follow the notebooks in `project_part_2` to preprocess data and train your YOLOv8 model on Kaggle.
3.  **Deploy System (Phase 3):** Download the trained weights and dataset from Drive, then run the web apps in `project_part3` to see the results in action.

## Resources
*   **Google Drive (Datasets & Models):** [https://drive.google.com/drive/folders/1-wpZjvE_s-5yX4lRyXxAlizcbOUU0WHp?usp=sharing]
*   **GitHub Repository:** [https://github.com/uguendi/Damage_Car_DL_Project/invitations]

## Acknowledgements
*   CarDD Dataset (USTC) & VehiDE Dataset (Kaggle)
*   Project Team Members:
    * Ahmad Zahir RAHIMI
    * Aleyna Begüm ORMAN
    * Uğur ENDİRLİK
