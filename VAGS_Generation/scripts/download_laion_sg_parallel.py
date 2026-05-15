import os
import shutil
import json
import requests
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Adding progress bar for better tracking

# https://huggingface.co/datasets/mengcy/LAION-SG

# Define the dataset name and the local directory to save the files
dataset_name = "mengcy/LAION-SG"
local_dir = "/path/to/datasets/laion_sg"

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Create the directory structure
for split in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(local_dir, split), exist_ok=True)

# Function to download and save a single item
def process_item(item, split, local_dir):
    image_id = item['img_id']
    image_url = item['url']
    json_data = item

    image_path = os.path.join(local_dir, split, f"{image_id}.jpg")
    json_path = os.path.join(local_dir, split, f"{image_id}.json")

    # Skip if image already exists
    if os.path.exists(image_path):
        return f"Skipped {image_id}"

    # Download image
    try:
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()
        with open(image_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    except requests.exceptions.TooManyRedirects:
        return f"TooManyRedirects: {image_url}"
    except requests.exceptions.RequestException as e:
        return f"Request failed for {image_url}: {e}"
    except TimeoutError:
        return f"TimeoutError for {image_url}"

    # Save JSON
    try:
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)
    except Exception as e:
        return f"JSON save failed for {image_id}: {e}"

    return f"Processed {image_id}"

# Process each split in parallel
def process_split(split, dataset, local_dir, max_workers=10):
    print(f"Starting {split} split processing...")
    items = list(dataset[split])
    total_items = len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_item, item, split, local_dir): idx 
            for idx, item in enumerate(items)
        }
        
        # Process results as they complete with progress bar
        completed = 0
        with tqdm(total=total_items, desc=f"Processing {split}") as pbar:
            for future in as_completed(future_to_item):
                result = future.result()
                completed += 1
                pbar.update(1)
                
                # Log every 1000 items or on error
                if "Processed" not in result and "Skipped" not in result:
                    print(result)
                elif completed % 1000 == 0:
                    print(f"{split} progress: {completed}/{total_items}")

# Execute for each split
for split in dataset.keys():
# for split in ['validation', 'test']:
    process_split(split, dataset, local_dir, max_workers=10)

print("Dataset downloaded and organized successfully.")