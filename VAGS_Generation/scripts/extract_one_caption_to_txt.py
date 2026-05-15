import os
import shutil

def process_image_caption_dataset(input_folder: str, output_folder: str):
    """
    Copies images and the first caption from corresponding text files
    from an input folder to an output folder.

    Args:
        input_folder (str): The path to the source folder containing images and text files.
        output_folder (str): The path to the destination folder.
    """
    # 1. Create the output directory if it doesn't exist to prevent errors
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory ensured at: '{output_folder}'")

    # A set of common image file extensions for quick lookup
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'}
    
    processed_files = 0
    copied_images = 0

    # 2. Iterate over all files in the source directory
    for filename in os.listdir(input_folder):
        # Split the filename into its base name and extension
        file_base, file_extension = os.path.splitext(filename)
        
        # Check if the file is an image by its extension
        if file_extension.lower() in image_extensions:
            # --- Handle Image File ---
            src_image_path = os.path.join(input_folder, filename)
            dest_image_path = os.path.join(output_folder, filename)
            
            # Copy the image file, preserving metadata
            shutil.copy2(src_image_path, dest_image_path)
            copied_images += 1
            
            # --- Handle Corresponding Text File ---
            txt_filename = file_base + '.txt'
            src_txt_path = os.path.join(input_folder, txt_filename)
            
            # Check if the corresponding .txt file exists
            if os.path.exists(src_txt_path):
                try:
                    with open(src_txt_path, 'r', encoding='utf-8') as f_in:
                        # Read only the very first line from the file
                        first_caption = f_in.readline().strip()
                    
                    # If a caption was actually read (i.e., the file wasn't empty)
                    if first_caption:
                        dest_txt_path = os.path.join(output_folder, txt_filename)
                        with open(dest_txt_path, 'w', encoding='utf-8') as f_out:
                            # Write the single caption to the new file
                            f_out.write(first_caption + '\n')
                        processed_files += 1
                        
                except Exception as e:
                    print(f"Could not process file {txt_filename}. Error: {e}")

    print(f"\nProcessing complete! ✨")
    print(f"Total images copied: {copied_images}")
    print(f"Total caption files processed: {processed_files}")


if __name__ == '__main__':
    # --- Configuration ---
    # ⚠️ IMPORTANT: Replace these paths with your actual folder paths
    # source_folder = '/path/to/datasets/flickr30k/train'
    # destination_folder = '/path/to/datasets/flickr30k/train_1stcaption'
    
    source_folder = '/path/to/datasets/coco17/validation'
    destination_folder = '/path/to/datasets/coco17/validation_1stcaption'

    if not os.path.isdir(source_folder):
        print(f"Error: The source folder '{source_folder}' does not exist. Please check the path.")
    else:
        process_image_caption_dataset(source_folder, destination_folder)