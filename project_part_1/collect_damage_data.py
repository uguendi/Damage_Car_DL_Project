import os
from bing_image_downloader import downloader

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

    output_dir = "dataset_raw_bing"

    # Iterate through categories and queries
    for severity, queries in damage_queries.items():
        print(f"--- Collecting images for severity: {severity} ---")
        for query in queries:
            print(f"Downloading images for query: '{query}'")
            try:
                downloader.download(
                    query,
                    limit=125,
                    output_dir=output_dir,
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                    verbose=True
                )
            except Exception as e:
                print(f"Error downloading images for query '{query}': {e}")

    print("\nData collection complete.")

if __name__ == "__main__":
    # Instructions for the user
    print("To run this script, make sure you have installed the library:")
    print("pip install bing-image-downloader")
    print("----------------------------------------------------------")
    
    collect_damage_data()
