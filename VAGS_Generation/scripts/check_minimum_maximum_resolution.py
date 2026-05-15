import os
from PIL import Image

def check_image_resolutions(input_folder):
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_exts):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        min_width = min(min_width, width)
                        min_height = min(min_height, height)
                        max_width = max(max_width, width)
                        max_height = max(max_height, height)
                except Exception as e:
                    print(f"[WARN] Skipped {img_path}: {e}")

    print(f"Minimum resolution: {min_width} x {min_height}")
    print(f"Maximum resolution: {max_width} x {max_height}")

# Example usage:
# input_folder = "/path/to/datasets/coco17/train"
input_folder = "/path/to/datasets/flickr30k/train"
check_image_resolutions(input_folder)
