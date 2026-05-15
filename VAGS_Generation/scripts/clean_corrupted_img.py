import os
import argparse
from PIL import Image

def check_and_delete_corrupted_images(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.png')):
                filepath = os.path.join(dirpath, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # verify() is used to check file integrity
                except Exception as e:
                    print(f"Deleting corrupted image: {filepath} (Error: {e})")
                    try:
                        os.remove(filepath)
                    except Exception as remove_error:
                        print(f"Failed to delete {filepath}: {remove_error}")

                    # Delete associated .json and .txt files
                    base, _ = os.path.splitext(filepath)
                    for ext in ['.json', '.txt']:
                        associated_file = base + ext
                        if os.path.exists(associated_file):
                            try:
                                os.remove(associated_file)
                                print(f"Deleted associated file: {associated_file}")
                            except Exception as extra_remove_error:
                                print(f"Failed to delete {associated_file}: {extra_remove_error}")

def main():
    # parser = argparse.ArgumentParser(description="Recursively remove corrupted jpg/png images from a given folder.")
    # parser.add_argument("folder", type=str, help="Path to the folder to be scanned.")
    # args = parser.parse_args()
    # args.folder
    input_folder = "/path/to/datasets/laion_sg"
    # input_folder = "/path/to/datasets/coco17"
    
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        return

    check_and_delete_corrupted_images(input_folder)

if __name__ == "__main__":
    main()