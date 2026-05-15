import os
from PIL import Image

# --- User Inputs ---
input_folders = [
    'output_baseline/coco17_sd35_no_reweight/checkpoint-26000/generated_images_coco17',
    'output_baseline/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17',
    'output_baseline/coco17_sd35_modesample/checkpoint-26000/generated_images_coco17',
    'output_baseline/coco17_sd35_cosmap/checkpoint-26000/generated_images_coco17',
    'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/generated_images_coco17'
]

input_ids = [
    '000000557672', '000000396338', '000000399655', '000000386912', '000000063602', '000000051008', '000000105912', '000000371552', '000000365886', '000000094185', '000000322829'
]

output_folder = 'data4paper/merged_images'
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

# --- Script Logic ---

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

# Process each image ID
for image_id in input_ids:
    images_to_merge = []
    
    # Collect images from each input folder
    for folder in input_folders:
        image_path = os.path.join(folder, f"{image_id}.png")
        try:
            img = Image.open(image_path)
            images_to_merge.append(img)
        except FileNotFoundError:
            print(f"Warning: Image not found and will be skipped: {image_path}")
            # To maintain layout, you could append a blank image placeholder
            # For simplicity, we just skip it here.
            continue

    if not images_to_merge:
        print(f"No images found for ID {image_id}. Skipping.")
        continue

    # Assuming all images have the same dimensions
    # Get width and height from the first image
    width, height = images_to_merge[0].size
    total_width = width * len(images_to_merge)

    # Create a new blank image with the calculated total width
    merged_image = Image.new('RGB', (total_width, height))

    # Paste each image side-by-side
    current_x = 0
    for img in images_to_merge:
        merged_image.paste(img, (current_x, 0))
        current_x += width

    # Save the final merged image
    output_path = os.path.join(output_folder, f"{image_id}_merged.png")
    merged_image.save(output_path)
    print(f"Successfully merged images for ID {image_id} into {output_path}")

print("\nProcessing complete.")