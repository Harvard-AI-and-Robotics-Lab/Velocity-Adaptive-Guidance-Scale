import os
import shutil
import json
import requests
from datasets import load_dataset

# https://huggingface.co/datasets/mengcy/LAION-SG

# Define the dataset name and the local directory to save the files
dataset_name = "mengcy/LAION-SG"
local_dir = "/path/to/datasets/laion_sg"

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Create the directory structure
os.makedirs(os.path.join(local_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(local_dir, 'validation'), exist_ok=True)
os.makedirs(os.path.join(local_dir, 'test'), exist_ok=True)

# Function to download and save files, ignoring TooManyRedirects errors
# def download_and_save_file(url, destination):
#     try:
#         response = requests.get(url, stream=True, timeout=15)
#         response.raise_for_status()  # raises an exception for 4xx/5xx responses
#     except requests.exceptions.TooManyRedirects:
#         print(f"TooManyRedirects: {url}, skipping.")
#         return
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed for {url}: {e}")
#         return

#     try:
#         with open(destination, 'wb') as file:
#             shutil.copyfileobj(response.raw, file)
#     except TimeoutError:
#         print(f"TimeoutError: The read operation timed out when downloading {url}, skipping.")
#         return
    
def download_and_save_file(url, destination):
    try:
        response = requests.get(url, stream=True, timeout=15)  # Increased timeout to 30 seconds
        response.raise_for_status()
    except requests.exceptions.TooManyRedirects:
        print(f"TooManyRedirects: {url}, skipping.")
        return
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return

    try:
        with open(destination, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    except TimeoutError: # This TimeoutError is for file operations, not network read timeout anymore
        print(f"TimeoutError: The read operation timed out when downloading {url}, skipping.")
        return
    except urllib3.exceptions.ReadTimeoutError as e: # Catch ReadTimeoutError explicitly here as well for clarity, though requests.get timeout should handle it
        print(f"ReadTimeoutError:  {url}, skipping. Error: {e}")
        return

# Process each split in the dataset
for split in dataset.keys():
    if split in ['train', 'validation']:
        continue
    for idx, item in enumerate(dataset[split]):
        if idx % 1000 == 0:
            print(f"Processing {split} split: {idx}/{len(dataset[split])}")
        image_id = item['img_id']
        image_url = item['url']
        json_data = item  # Assuming the item itself is the JSON data

        # Define paths for image and JSON files
        image_path = os.path.join(local_dir, split, f"{image_id}.jpg")
        json_path = os.path.join(local_dir, split, f"{image_id}.json")

        # if the file image_path exists, continue the loop (skip the file)
        if os.path.exists(image_path):
            continue  

        # Download and save the image
        download_and_save_file(image_url, image_path)

        # Save the JSON data
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)

print("Dataset downloaded and organized successfully.")
