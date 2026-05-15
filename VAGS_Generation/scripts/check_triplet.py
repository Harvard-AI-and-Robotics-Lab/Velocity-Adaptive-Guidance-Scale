import os
import argparse
from PIL import Image

# Define the extensions of interest
IMAGE_EXTS = {'.jpg', '.png'}
REQUIRED_EXTS = {'.json', '.txt'}

def check_triplets_in_directory(dirpath, filenames):
    groups = {}
    # Build groups for files with required extensions
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in IMAGE_EXTS or ext in REQUIRED_EXTS:
            base = os.path.splitext(filename)[0]
            groups.setdefault(base, set()).add(ext)

    # For each group that has an image, check if the required files exist
    for base, exts in groups.items():
        if exts & IMAGE_EXTS:
            missing = [req for req in REQUIRED_EXTS if req not in exts]
            if missing:
                # Build the full paths for available files in the group:
                files_found = [os.path.join(dirpath, base + ext) for ext in exts]
                print(f"Incomplete triplet for base '{base}' in {dirpath}:")
                print(f"  Files present: {files_found}")
                print(f"  Missing extensions: {missing}\n")

def main():
    # parser = argparse.ArgumentParser(
    #     description="Traverse input folder recursively and show incomplete file triplets."
    # )
    # parser.add_argument(
    #     "input_folder",
    #     type=str,
    #     help="Path to the input folder (e.g., /path/to/datasets/laion_sg)"
    # )
    # args = parser.parse_args()

    input_folder = "/path/to/datasets/laion_sg"
    # input_folder = "/path/to/datasets/coco17"

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        return

    for dirpath, _, filenames in os.walk(input_folder):
        check_triplets_in_directory(dirpath, filenames)
        

if __name__ == "__main__":
    main()