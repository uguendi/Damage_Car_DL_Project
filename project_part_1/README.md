# Deep Learning Project: Vehicle Damage Detection - Phase 1

## Overview
This folder contains the code for **Phase 1: Dataset Creation**. The goal of this phase is to collect a diverse dataset of vehicle images categorized by damage severity (Minor, Moderate, Severe, General) to train the deep learning models in subsequent phases.

## Structure
*   `collect_damage_data.py`: The main Python script that utilizes the `bing-image-downloader` library to fetch images.
*   `requirements.txt`: List of dependencies required to run the script.
*   `dataset_raw_bing/`: (Generated) The directory where downloaded images are stored, organized by category.

## Prerequisites
Ensure you have Python installed. Install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run
Navigate to this directory and execute the script:

```bash
python collect_damage_data.py
```

## Search Queries
The script uses a predefined list of search queries mapped to damage categories:
*   **Minor:** `car dent`, `car scratches`, etc.
*   **Moderate:** `car bumper hanging off`, `broken car side mirror`, etc.
*   **Severe:** `totaled car front end damage`, `car rollover damage`, etc.
*   **General:** `minor damage vehicle images`, `slightly damage vehicle images`, etc.

### Modifying Queries
You can manually modify the search queries by editing the `damage_queries` dictionary in `collect_damage_data.py`.

## Configuration
*   **Threads:** The script runs sequentially for stability but can be modified for parallelism.
*   **Limit:** Currently set to download **125 images** per query.
*   **Timeout:** 60 seconds per image download attempt.
