# python scripts/merge_multiple_images_side_by_side.py /path/to/folder1 /path/to/folder2 /path/to/output_folder
# python scripts/merge_multiple_images_side_by_side.py /path/to/folder1 /path/to/folder2 /path/to/folder3 /path/to/output_folder
#!/usr/bin/env python3
# python scripts/merge_multiple_images_side_by_side.py data4paper/image_caption/no_reweight data4paper/image_caption/lognorm data4paper/image_caption/modesample data4paper/image_caption/cosmap data4paper/image_caption/proposed data4paper/image_caption/merged 'dsf'
# python scripts/merge_multiple_images_side_by_side.py data4paper/image_caption/no_reweight data4paper/image_caption/lognorm data4paper/image_caption/modesample data4paper/image_caption/cosmap data4paper/image_caption/proposed data4paper/image_caption/merged 'dsf'
# python scripts/merge_multiple_images_side_by_side.py data4paper/qual_examples/no_reweight data4paper/qual_examples/lognorm data4paper/qual_examples/modesample data4paper/qual_examples/cosmap data4paper/qual_examples/proposed data4paper/qual_examples/merged 'dsf'
import os
import argparse
from PIL import Image

def merge_images(input_folders, output_folder, match_method="filename"):
    """
    Merge matching images from multiple folders side by side.
    
    Parameters:
    - input_folders: List of paths to folders containing images
    - output_folder: Path to save merged images
    - match_method: How to match images across folders ('filename' by default)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all files from first folder
    if not input_folders:
        print("No input folders provided.")
        return
    
    # Use first folder as reference for finding matching files
    reference_folder = input_folders[0]
    files = os.listdir(reference_folder)
    
    # Filter for image files (adjust extensions as needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Process each image from reference folder
    for image_file in image_files:
        # Check if this file exists in all folders
        matching_images = []
        
        # Collect matching images from all folders
        all_matched = True
        for folder in input_folders:
            if image_file in os.listdir(folder):
                img_path = os.path.join(folder, image_file)
                try:
                    img = Image.open(img_path)
                    matching_images.append(img)
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    all_matched = False
                    break
            else:
                print(f"No matching file found for {image_file} in {folder}")
                all_matched = False
                break
        
        # Only proceed if we found a match in all folders
        if not all_matched or not matching_images:
            continue
        
        # Resize all images to the same height (using the first image as reference)
        reference_height = matching_images[0].height
        for i in range(1, len(matching_images)):
            if matching_images[i].height != reference_height:
                # Calculate new dimensions
                ratio = reference_height / matching_images[i].height
                new_width = int(matching_images[i].width * ratio)
                matching_images[i] = matching_images[i].resize((new_width, reference_height), Image.LANCZOS)
        
        # Calculate total width and max height
        total_width = sum(img.width for img in matching_images)
        max_height = max(img.height for img in matching_images)
        
        # Create a new image with enough width for all images side by side
        merged_img = Image.new('RGB', (total_width, max_height))
        
        # Paste all images side by side
        current_x = 0
        for img in matching_images:
            merged_img.paste(img, (current_x, 0))
            current_x += img.width
        
        # Save the merged image
        output_filename = f"merged_{image_file}"
        output_path = os.path.join(output_folder, output_filename)
        merged_img.save(output_path)
        print(f"Merged {image_file} successfully from {len(matching_images)} folders.")
    
    print(f"All matching images have been merged and saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge matching images from multiple folders side by side.")
    parser.add_argument("folders", nargs='+', help="Paths to folders containing images")
    parser.add_argument("output_folder", help="Path to save merged images")
    
    args = parser.parse_args()
    
    # Make sure we have at least one input folder
    if len(args.folders) < 1:
        parser.error("You must provide at least one input folder")
    
    
    # The last argument is the output folder
    output_folder = args.folders.pop()
    input_folders = args.folders
    # print(output_folder, input_folders)
    # sys.exit()
    
    
    merge_images(input_folders, output_folder)