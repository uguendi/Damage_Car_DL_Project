import os
from bing_image_downloader import downloader

import concurrent.futures

def download_query(query):
    """
    Function to download images for a single query.
    Designed to be run in a separate thread.
    """
    output_dir = "dataset_raw_bing"
    print(f"--- Starting download for query: '{query}' ---")
    try:
        downloader.download(
            query,
            limit=125,  # <--- LIMIT: Number of images to download per query
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60, # <--- TIMEOUT: Max time to wait for connection in seconds
            verbose=False # Set to False for cleaner output during parallel execution
        )
        print(f"--- Finished download for query: '{query}' ---")
    except Exception as e:
        print(f"Error downloading images for query '{query}': {e}")

def collect_damage_data():
    # Define damage categories and specific search queries
    damage_queries = {
        "Minor": [
            "car dent",
            "car scratches",
            "car paint chips",
            "car door ding",
            "car bumper minor dent",
            "car door key scratch",
            "vehicle fender fender bender",
            "car paint chip close up"
        ],
        "Moderate": [
            "car bumper hanging off",
            "smashed car tail light",
            "broken car side mirror",
            "cracked car windshield accident",
            "side impact car collision"
        ],
        "Severe": [
            "car rollover damage",
            "totaled car front end damage",
            "car deployed airbags accident",
            "wrecked car engine bay",
            "salvage car auction lot"
        ],
        "General": [
            "minor damage vehicle images",
            "slightly damage vehicle images",
            "severeyly damage vehicle images"
        ]
    }

    # Flatten the dictionary into a single list of queries for parallel processing
    all_queries = []
    for queries in damage_queries.values():
        all_queries.extend(queries)

    print(f"Starting parallel download for {len(all_queries)} queries...")
    
    # <--- PARALLELISM: max_workers controls how many downloads happen at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_query, all_queries)

    print("\nData collection complete.")

if __name__ == "__main__":
    # Instructions for the user
    print("To run this script, make sure you have installed the library:")
    print("pip install bing-image-downloader")
    print("----------------------------------------------------------")
    
    collect_damage_data()
