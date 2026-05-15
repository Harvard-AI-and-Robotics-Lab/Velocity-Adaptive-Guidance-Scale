import os
import shutil
import json
import pdb

def copy_and_rename_images(source_base_path, destination_path, split):
    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)
    
    # Path to train folder
    train_path = os.path.join(source_base_path, split)

    # Check if train folder exists
    if not os.path.exists(train_path):
        print(f"Error: Train folder not found at {train_path}")
        return
    
    # Iterate through numbered folders
    for folder in os.listdir(train_path):
        # try:
        # Verify folder name is a number
        int(folder)
        img_name = f"{folder}.jpg"
        
        # Source image path
        source_image = os.path.join(train_path, folder, 'image.jpg')
        
        # Destination image path with new name
        dest_image = os.path.join(destination_path, img_name)
        
        # Copy and rename if source image exists
        if os.path.exists(source_image):
            shutil.copy2(source_image, dest_image)
            print(f"Copied: {source_image} -> {dest_image}")
        else:
            print(f"Warning: Image not found in folder {folder}")

        # Path to output.json
        json_file = os.path.join(train_path, folder, 'output.json')
        
        if os.path.exists(json_file):
            # Read and extract caption from JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
                caption = data.get('caption', '')
            
            # Save caption to numbered text file
            output_file = os.path.join(destination_path, f"{folder}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(caption)
                
        # except ValueError:
        #     # Skip folders that aren't numbers
        #     print(f"Skipping non-numeric folder: {folder}")
        # except Exception as e:
        #     print(f"Error processing folder {folder}: {str(e)}")


# Example usage
split = 'test'

# source_path = "/path/to/data/richhf_18k"  # Current directory or path to your base folder
source_path = "/path/to/datasets/richhf_18k_dataset/richhf_18k"  # Current directory or path to your base folder
# destination_path = f"/path/to/data/richhf_18k/{split}_for_simpletuner"  # Where you want the renamed images
destination_path = f"/path/to/datasets/richhf_18k_dataset/richhf_18k/{split}_for_simpletuner"  # Where you want the renamed images

copy_and_rename_images(source_path, destination_path, split)