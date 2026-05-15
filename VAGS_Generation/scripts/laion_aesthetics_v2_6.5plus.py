import os

from dagshub.streaming import DagsHubFilesystem
from PIL import Image

import pdb


def convert_to_jpg_if_needed(filename):
    # Split filename into the base and extension.
    # os.path.splitext splits on the LAST dot, preserving dots in the base.
    base, ext = os.path.splitext(filename)

    # Check if the extension is already .jpg (case-insensitive).
    if ext.lower() == '.jpg':
        print(f"'{filename}' already has a .jpg extension. No conversion needed.")
        return filename

    # Define the new filename: if there's no extension or the extension is not .jpg,
    # we save the image as <base>.jpg (keeping any dots in the base name).
    new_filename = base + ".jpg"

    try:
        # Open the original image.
        img = Image.open(filename)

        # Convert to RGB (useful if the image has transparency or is in a different mode)
        rgb_img = img.convert("RGB")

        # Save the image as JPEG.
        rgb_img.save(new_filename, "JPEG")
        print(f"Converted '{filename}' to '{new_filename}'.")

        # Delete the original file.
        os.remove(filename)
        print(f"Deleted the original file '{filename}'.")

        return new_filename

    except Exception as e:
        print(f"Error processing '{filename}': {e}")

        return None


if __name__ == '__main__':

    local_path = '/path/to/datasets/laion_aesthetics'

    # Setup data streaming from DagsHub
    fs = DagsHubFilesystem(local_path,
                           repo_url='https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus')
    fs.install_hooks()

    # Get all images + labels.tsv file
    files = fs.listdir(os.path.join(local_path, 'data'))

    # Get the data for the first 5 images in the labels.tsv file
    with fs.open(os.path.join(local_path, 'data', 'labels.tsv')) as tsv:
        for i, row in enumerate(tsv.readlines()[:128]):
            row = row.strip()
            img_file, caption, score, url = row.split('\t')

            # Load the image file
            img_path = os.path.join(local_path, 'data', img_file)
            img = Image.open(img_path)

            # Change image filename if necessary
            new_img_path = convert_to_jpg_if_needed(img_path)

            print(f'No. {i}: {img_file} has a size of {img.size} and an aesthetics score of {score}')

            if new_img_path is not None:
                # Save caption to a file
                with open(new_img_path.replace('.jpg', '.txt'), "w") as file:
                    file.write(caption)
