import os
import json
import random
from pathlib import Path

def extract_random_caption(local_dir, splits=['train', 'validation'], text='captions'):
    """
    Extract a random caption from JSON files in specified splits and save to .txt files.
    
    Args:
        local_dir (str): Base directory containing split subdirectories
        splits (list): List of subdirectory names to process ['train', 'validation']
    """
    # Ensure local_dir is a Path object
    local_dir = Path(local_dir)
    
    # Process each split
    for split in splits:
        split_dir = local_dir / split
        
        # Check if directory exists
        if not split_dir.exists() or not split_dir.is_dir():
            print(f"Directory {split_dir} does not exist or is not a directory. Skipping.")
            continue
            
        print(f"Processing {split} split...")
        
        # Get all JSON files in the directory
        json_files = list(split_dir.glob('*.json'))
        
        if not json_files:
            print(f"No JSON files found in {split_dir}. Skipping.")
            continue
            
        # Process each JSON file
        for json_file in json_files:
            try:
                # Read JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if 'captions' key exists and has content
                captions = data.get(text, [])
                if not captions: # or not isinstance(captions, list):
                    # print(f"No valid captions found in {json_file}. Skipping.")
                    continue

                if isinstance(captions, list):
                    # Pick a random caption
                    random_caption = random.choice(captions)
                else:
                    random_caption = captions
                
                # Create corresponding .txt filename
                txt_file = json_file.with_suffix('.txt')
                
                # Write the random caption to txt file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(random_caption)
                    
                print(f"Processed {json_file.name} -> {txt_file.name}")
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON file {json_file}. Skipping.")
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}. Skipping.")
    
    print("Caption extraction completed.")

# Example usage
if __name__ == "__main__":
    # local_dir = "/path/to/datasets/coco17"  # Replace with your path
    # splits = ['train', 'validation']
    # text = 'captions'

    local_dir = "/path/to/datasets/laion_sg"  # Replace with your path
    # splits = ['train', 'validation', 'test']
    splits = ['train', 'validation', 'test']
    text = 'caption_ori'

    extract_random_caption(local_dir, splits, text=text)