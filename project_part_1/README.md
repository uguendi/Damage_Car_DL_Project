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
The script is configured for parallel execution to speed up data collection. You can adjust the following parameters directly in the `collect_damage_data.py` file:

*   **Parallelism (`max_workers`)**: Controlled by the `max_workers` parameter in the `ThreadPoolExecutor` line (at the bottom of the script).
    *   *Current Value:* `5` (Downloads 5 queries simultaneously).
    *   *How to change:* Increase for faster downloads if your internet is fast; decrease if you experience timeouts or errors.

*   **Limit (`limit`)**: Controlled by the `limit` parameter inside the `download_query` function.
    *   *Current Value:* `125` images per query.
    *   *How to change:* Set to your desired number of images.

*   **Timeout (`timeout`)**: Controlled by the `timeout` parameter inside the `download_query` function.
    *   *Current Value:* `60` seconds.
    *   *How to change:* Increase if you have a slow connection to prevent giving up on downloads too early.

## Disclaimer
*   **Data Quality**: The script uses Bing Image Search, which retrieves images based on text queries without strict content filtering. Consequently, the downloaded dataset may contain irrelevant or low-quality images. **Manual verification is required** to filter out unusable images and ensure the dataset is meaningful for training.
*   **Download Counts**: The target limit (e.g., 125 images) is a maximum threshold. The actual number of images downloaded may be lower due to factors such as broken links, connection timeouts, or a lack of relevant search results. This is normal behavior.
