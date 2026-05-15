import os
from pathlib import Path
from PIL import Image
import argparse


def merge_images(input_folder, subfolders, output_dir, direction='horizontal', spacing=10):
    """
    Merge images with the same name from multiple subfolders.
    
    Args:
        input_folder (str): Path to the main input folder
        subfolders (list): List of subfolder names to merge images from
        output_dir (str): Path to save merged images
        direction (str): 'horizontal' or 'vertical' merge direction
        spacing (int): Pixels of spacing between images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert input_folder to Path object
    input_path = Path(input_folder)
    
    # Validate that all subfolders exist
    subfolder_paths = []
    for subfolder in subfolders:
        subfolder_path = input_path / subfolder
        if not subfolder_path.exists():
            print(f"Warning: Subfolder '{subfolder}' does not exist. Skipping.")
        else:
            subfolder_paths.append(subfolder_path)
    
    if not subfolder_paths:
        print("Error: No valid subfolders found.")
        return
    
    # Get all image files from the first subfolder as reference
    first_subfolder = subfolder_paths[0]
    image_files = set()
    
    # Collect all unique image names across all subfolders
    for subfolder_path in subfolder_paths:
        if subfolder_path.exists():
            for file in subfolder_path.glob('*.png'):
                image_files.add(file.name)
            for file in subfolder_path.glob('*.jpg'):
                image_files.add(file.name)
            for file in subfolder_path.glob('*.jpeg'):
                image_files.add(file.name)
    
    print(f"Found {len(image_files)} unique image files to process.")
    
    # Process each image
    merged_count = 0
    for image_name in sorted(image_files):
        images_to_merge = []
        found_in_folders = []
        
        # Collect images with the same name from each subfolder
        for subfolder_path in subfolder_paths:
            image_path = subfolder_path / image_name
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    images_to_merge.append(img)
                    found_in_folders.append(subfolder_path.name)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        
        # Debug output
        if len(images_to_merge) > 0:
            print(f"'{image_name}': found in {len(images_to_merge)} folders {found_in_folders}")
        
        # Skip if we don't have at least one image
        if not images_to_merge:
            continue
        
        # Skip if we only have one image (nothing to merge)
        if len(images_to_merge) == 1:
            print(f"Only one image found for '{image_name}'. Copying to output.")
            output_path = Path(output_dir) / image_name
            images_to_merge[0].save(output_path)
            merged_count += 1
            continue
        
        # Merge images
        try:
            if direction == 'horizontal':
                merged_image = merge_horizontal(images_to_merge, spacing)
            else:
                merged_image = merge_vertical(images_to_merge, spacing)
            
            # Save merged image
            output_path = Path(output_dir) / image_name
            merged_image.save(output_path)
            merged_count += 1
            print(f"Merged {len(images_to_merge)} images for '{image_name}'")
            
        except Exception as e:
            print(f"Error merging images for '{image_name}': {e}")
    
    print(f"\nSuccessfully merged {merged_count} images to '{output_dir}'")


def merge_horizontal(images, spacing=0):
    """Merge images horizontally."""
    # Get dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # Calculate total width and max height
    total_width = sum(widths) + spacing * (len(images) - 1)
    max_height = max(heights)
    
    # Create new image with white background
    merged = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images
    x_offset = 0
    for img in images:
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        merged.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    
    return merged


def merge_vertical(images, spacing=0):
    """Merge images vertically."""
    # Get dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # Calculate max width and total height
    max_width = max(widths)
    total_height = sum(heights) + spacing * (len(images) - 1)
    
    # Create new image with white background
    merged = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste images
    y_offset = 0
    for img in images:
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        merged.paste(img, (0, y_offset))
        y_offset += img.height + spacing
    
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge images with the same name from multiple subfolders.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the main input folder')
    parser.add_argument('--subfolders', type=str, nargs='+', required=True,
                        help='List of subfolder names to merge images from')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save merged images')
    parser.add_argument('--direction', type=str, default='horizontal',
                        choices=['horizontal', 'vertical'],
                        help='Direction to merge images (default: horizontal)')
    parser.add_argument('--spacing', type=int, default=10,
                        help='Pixels of spacing between images (default: 10)')
    
    args = parser.parse_args()
    
    merge_images(
        input_folder=args.input_folder,
        subfolders=args.subfolders,
        output_dir=args.output_dir,
        direction=args.direction,
        spacing=args.spacing
    )