# YOLOv8 Car Damage Detection Project

This project uses YOLOv8 to detect car damages and specific damage areas. It consists of two main parts:
1.  **Training (`yolo_training_2.ipynb`)**: Data preprocessing and model training.
2.  **Optimization & Analysis (`Analysis&Optimization.ipynb`)**: Hyperparameter optimization and result visualization.

## 1. Data Setup (Important!)

Before running any code, you must set up the dataset on Kaggle or any other platform.

1.  **Prepare your Data**:
    *   Take your data from your **Google Drive** folder.
    *   Ensure it contains the images and the two JSON annotation files:
        *   `0Train_via_annos.json`
        *   `0Val_via_annos.json`
2.  **Upload to Kaggle**:
    *   Go to Kaggle and create a **New Dataset**.
    *   Upload your images and JSON files.
    *   **Name the dataset**: `car-damage-labeled-yolo`
    *   *Note: If you name it something else, you will need to change the `RAW_ROOT` path in the code.*

## 2. How to Run the Code

### Part 1: Training (`yolo_training_2.ipynb`)
This notebook handles data splitting, configuration, and training the YOLOv8 model.

1.  **Open** `yolo_training_2.ipynb` in Kaggle.
2.  **Add Data**: Click "Add Input" and select the `car-damage-labeled-yolo` dataset you created.
3.  **Run All Cells**:
    *   The code will automatically create `train/val` folders in `/kaggle/working/`.
    *   It converts JSON annotations to YOLO format (`.txt`).
    *   It trains the model for 50 epochs (default).If you want to try more epochs, you can change the `epochs` parameter in the code.
4.  **Output**:
    *   The trained model weights will be saved at: `/kaggle/working/runs/detect/car_damage_7cls_final/weights/best.pt`
    *   **Save Version**: Click "Save Version" -> "Save & Run All (Commit)" to finish the training and save the outputs.

### Part 2: Optimization & Analysis (`Analysis&Optimization.ipynb`)
This notebook performs hyperparameter optimization (tuning) and analyzes results.

1.  **Open** `Analysis&Optimization.ipynb` in Kaggle.
2.  **Add Data (Crucial)**:
    *   You **MUST** add the `car-damage-labeled-yolo` dataset (same as Part 1).
    *   *Why?* Optimization involves re-training the model with different parameters, so it needs access to the images and labels.
3.  **YAML Configuration**:
    *   You do **not** need to upload a YAML file.
    *   The code in this notebook will **automatically generate** a new `damage_config.yaml` file for the current session (just like in Part 1).
4.  **Run All Cells**:
    *   The notebook will run multiple training experiments to find the best hyperparameters.
    *   It will also visualize the final results (`results.png`, `confusion_matrix.png`, etc.).

## File Structure

*   `yolo_training_2.ipynb`: Main training script.
*   `Analysis&Optimization.ipynb`: Result analysis and optimization script.
*   `damage_config.yaml`: Generated automatically during training.

## Classes
The model detects the following 7 classes:
0.  **tear_crack** (rach)
1.  **scratch** (tray_son)
2.  **shattered_glass** (vo_kinh)
3.  **missing_part** (mat_bo_phan)
4.  **puncture** (thung)
5.  **dent** (mop_lom)
6.  **broken_light** (be_den)
