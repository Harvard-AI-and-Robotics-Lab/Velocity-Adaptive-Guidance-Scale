import os
import json

def iterate_subfolders(base_path):
    # List of main splits
    splits = ['train', 'dev', 'test']
    
    # Dictionary to store all paths
    all_paths = {}
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split} folder not found")
            continue
            
        # Get all numeric subfolders
        subfolders = []
        for folder in os.listdir(split_path):
            if os.path.isdir(os.path.join(split_path, folder)):
                try:
                    # Verify the folder name is a number
                    int(folder)
                    subfolders.append(folder)
                except ValueError:
                    continue
                    
        # Sort subfolders numerically
        subfolders.sort(key=lambda x: int(x))
        all_paths[split] = subfolders
        
        # Print found subfolders
        print(f"\nFound {len(subfolders)} subfolders in {split}:")
        print(subfolders)
    
    return all_paths

def extract_caption(json_file_path, output_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the caption
    caption = data['caption']
    
    # Write the caption to a new file
    with open(output_file_path, 'w') as f:
        f.write(caption)

# Example usage
base_folder = "/path/to/datasets/richhf_18k_dataset/richhf_18k"
paths = iterate_subfolders(base_folder)

# Example of how to iterate through all subfolders
for split, subfolders in paths.items():
    for subfolder in subfolders:
        full_path = os.path.join(base_folder, split, subfolder, 'output.json')
        print(f"Processing: {full_path}")
        # Do something with each subfolder here

for split, subfolders in paths.items():
    print(f'# of {split} samples {len(subfolders)}')