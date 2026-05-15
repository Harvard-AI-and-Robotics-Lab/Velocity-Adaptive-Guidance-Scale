import os
import random
import shutil
from pathlib import Path

def select_random_pairs(src_folder, dst_folder, ratio=0.1, seed=42):
    """
    Randomly select a percentage of (image_id.jpg, image_id.txt) pairs from src_folder
    and copy them to dst_folder.

    Args:
        src_folder (str or Path): Source directory containing .jpg and .txt pairs.
        dst_folder (str or Path): Destination directory to save the selected pairs.
        ratio (float): Proportion of pairs to select (default=0.1 for 10%).
        seed (int): Random seed for reproducibility.
    """
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Collect base image IDs that have both .jpg and .txt
    image_ids = [
        p.stem for p in src_folder.glob("*.jpg")
        if (src_folder / f"{p.stem}.txt").exists()
    ]
    print(f"Found {len(image_ids)} valid pairs.")

    # Randomly sample 10%
    random.seed(seed)
    n_select = max(1, int(len(image_ids) * ratio))
    selected_ids = random.sample(image_ids, n_select)
    print(f"Selecting {n_select} pairs ({ratio*100:.1f}%).")

    # Copy files to destination
    for img_id in selected_ids:
        for ext in [".jpg", ".txt"]:
            src_path = src_folder / f"{img_id}{ext}"
            dst_path = dst_folder / f"{img_id}{ext}"
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

    print(f"Copied {n_select} pairs to {dst_folder}")

# Example usage
if __name__ == "__main__":

    # input_folder = "/path/to/datasets/coco17/train_1stcaption"
    # output_folder = "/path/to/datasets/coco17/train_1stcaption_part01"
    # ratio = 0.1 

    input_folder = "/path/to/datasets/coco17/validation_bestcaption"
    output_folder = "/path/to/datasets/coco17/validation_bestcaption_part02"
    ratio = 0.2

    select_random_pairs(
        src_folder=input_folder,
        dst_folder=output_folder,
        ratio=ratio
    )
