# python merge_images.py /path/to/folder1 /path/to/folder2 /path/to/output_folder
import os
import argparse
from PIL import Image

def merge_images(folder1, folder2, output_folder, match_method="filename"):
    """
    Merge matching images from two folders side by side.
    
    Parameters:
    - folder1: Path to the first folder containing images
    - folder2: Path to the second folder containing images
    - output_folder: Path to save merged images
    - match_method: How to match images across folders ('filename' by default)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all files from folder1
    files1 = os.listdir(folder1)
    # Filter for image files (adjust extensions as needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files1 = [f for f in files1 if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Process each image from folder1
    for image_file in image_files1:
        # Find the matching file in folder2
        matching_file = image_file  # For filename matching
        
        # Check if matching file exists in folder2
        if matching_file in os.listdir(folder2):
            # Open both images
            img1 = Image.open(os.path.join(folder1, image_file))
            img2 = Image.open(os.path.join(folder2, matching_file))
            
            # Resize images to same height if they differ
            if img1.height != img2.height:
                # Calculate new dimensions for img2
                new_height = img1.height
                ratio = new_height / img2.height
                new_width = int(img2.width * ratio)
                img2 = img2.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a new image with enough width for both images side by side
            merged_width = img1.width + img2.width
            merged_height = max(img1.height, img2.height)
            merged_img = Image.new('RGB', (merged_width, merged_height))
            
            # Paste both images side by side
            merged_img.paste(img1, (0, 0))
            merged_img.paste(img2, (img1.width, 0))
            
            # Save the merged image
            output_filename = f"merged_{image_file}"
            output_path = os.path.join(output_folder, output_filename)
            merged_img.save(output_path)
            print(f"Merged {image_file} successfully.")
        else:
            print(f"No matching file found for {image_file} in {folder2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge matching images from two folders side by side.")
    parser.add_argument("folder1", help="Path to the first folder containing images")
    parser.add_argument("folder2", help="Path to the second folder containing images")
    parser.add_argument("output_folder", help="Path to save merged images")
    
    args = parser.parse_args()
    
    merge_images(args.folder1, args.folder2, args.output_folder)
    print(f"All matching images have been merged and saved to {args.output_folder}")