import argparse
import os
import shutil
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# from data_handler import FairGenMed
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# import torch_fidelity
# from train_text_to_image import normalize_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.multimodal import clip_score
from functools import partial

import clip
from typing import List, Union
from pathlib import Path

import pdb
import csv  # Added for CSV output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on Image Generation")
    parser.add_argument("--gt_img_dir", type=str, default='')
    parser.add_argument("--gen_img_dir", type=str, default='')
    parser.add_argument("--prompt_dir", type=str, default='')
    parser.add_argument("--output", type=str, default='evaluation_results.csv') # Changed default to .csv
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling") # Added seed
    args = parser.parse_args()
    return args

# Options
# ”openai/clip-vit-base-patch16”
# ”openai/clip-vit-base-patch32”
# ”openai/clip-vit-large-patch14-336”
# ”openai/clip-vit-large-patch14”
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    # images_int = (images * 255).astype("uint8")
    # clip_scores = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()

    clip_scores = clip_score_fn(images, list(prompts)).detach()

    return round(float(clip_scores), 4)

# def calculate_clip_score(
#     images: List[Union[str, Path, Image.Image]], 
#     prompts: List[str],
#     clip_model: str = "ViT-B/32",
#     batch_size: int = 50,
#     device: str = None
# ) -> float:
#     """Calculate CLIP score between images and text prompts.
    
#     Args:
#         images: List of image paths or PIL Image objects
#         prompts: List of text prompts
#         clip_model: CLIP model to use (default: "ViT-B/32")
#         batch_size: Batch size for processing (default: 50)
#         device: Device to use (default: None, will use CUDA if available)
    
#     Returns:
#         float: Average CLIP score between images and prompts
    
#     Raises:
#         ValueError: If number of images and prompts don't match
#         TypeError: If images contains unsupported types
#     """
#     # Validate inputs
#     if len(images) != len(prompts):
#         raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
    
#     # Set device
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device(device)
    
#     # Load CLIP model
#     model, preprocess = clip.load(clip_model, device=device)
#     model.eval()
    
#     # Process inputs in batches
#     score_sum = 0.0
#     n_samples = len(images)
    
#     with torch.no_grad():
#         for i in range(0, n_samples, batch_size):
#             batch_images = images[i:i + batch_size]
#             batch_prompts = prompts[i:i + batch_size]
            
#             # Process images
#             processed_images = []
#             for img in batch_images:
#                 if isinstance(img, (str, Path)):
#                     img = Image.open(img).convert('RGB')
#                 elif isinstance(img, Image.Image):
#                     img = img.convert('RGB')
#                 elif isinstance(img, torch.Tensor):
#                     # If tensor is in range [0, 1], scale to [0, 255]
#                     if img.max() <= 1.0:
#                         img = (img * 255).to(torch.uint8)
#                     # Ensure tensor is in correct format (B, C, H, W) or (C, H, W)
#                     if img.dim() == 4:
#                         img = img.squeeze(0)  # Remove batch dimension if present
#                     # Convert to PIL Image
#                     img = transforms.ToPILImage()(img)
#                 else:
#                     raise TypeError(f"Unsupported image type: {type(img)}")
#                 processed_images.append(preprocess(img))
            
#             # Convert to tensors
#             image_tensor = torch.stack(processed_images).to(device)

#             # text_tokens = clip.tokenize(batch_prompts).to(device)

#             # Truncate prompts to fit CLIP's context length (77 tokens)
#             truncated_prompts = [prompt[:77] if isinstance(prompt, str) else prompt for prompt in batch_prompts]
#             try:
#                 text_tokens = clip.tokenize(truncated_prompts).to(device)
#             except RuntimeError as e:
#                 # If still too long, try more aggressive truncation
#                 print(f"Warning: Truncating prompts further due to length: {e}")
#                 truncated_prompts = [prompt[:50] if isinstance(prompt, str) else prompt for prompt in batch_prompts]
#                 text_tokens = clip.tokenize(truncated_prompts).to(device)
            
#             # Get features
#             image_features = model.encode_image(image_tensor)
#             text_features = model.encode_text(text_tokens)
            
#             # Normalize features
#             image_features = image_features / image_features.norm(dim=1, keepdim=True)
#             text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
#             # Calculate batch scores
#             logit_scale = model.logit_scale.exp()
#             scores = logit_scale * (image_features * text_features).sum(dim=1)
#             score_sum += scores.sum().item()
    
#     return score_sum / n_samples


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, prompt_dir, transform=None, filter_by_basenames=None):
        self.img_dir = img_dir
        self.prompt_dir = prompt_dir
        self.transform = transform
        self.images = []
        self.prompts = []
        self.basenames = []  # Store the base filenames of valid samples

        # Use a set for efficient lookup if a filter list is provided
        filter_set = set(filter_by_basenames) if filter_by_basenames is not None else None

        potential_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])

        for img_filename in potential_images:
            base_name = os.path.splitext(img_filename)[0]

            # 🚀 Synchronization logic: skip if not in the filter list
            if filter_set is not None and base_name not in filter_set:
                continue

            prompt_filename = f"{base_name}.txt"
            img_path = os.path.join(self.img_dir, img_filename)
            prompt_path = os.path.join(self.prompt_dir, prompt_filename)

            if os.path.exists(prompt_path):
                try:
                    # image = Image.open(img_path).convert('RGB')
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    # If all checks pass, add the sample
                    self.images.append(img_filename)
                    self.prompts.append(prompt_filename)
                    self.basenames.append(base_name) # 👈 Add the valid basename

                except (IOError, SyntaxError, OSError, Image.UnidentifiedImageError) as e:
                    print(f"⚠️ Warning: Skipping corrupted image: {img_path}. Error: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        prompt_name = os.path.join(self.prompt_dir, self.prompts[idx])

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        with open(prompt_name, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()

        return image, prompt

def calculate_all_metrics(gt_full_dataset, gen_subset, device):
    """
    Calculates FID, MIFID, IS, and CLIP scores.
    
    FID/MIFID: Compares the full ground-truth dataset to the generated subset.
    IS/CLIP: Calculated only on the generated subset.
    """
    
    # --- Create DataLoaders ---
    # We create separate dataloaders for each metric group to avoid
    # issues with exhausted iterators.
    
    # For FID
    gt_dataloader_fid = DataLoader(gt_full_dataset, batch_size=64, num_workers=0, shuffle=False)
    gen_dataloader_fid = DataLoader(gen_subset, batch_size=64, num_workers=0, shuffle=False)
    
    # For MIFID
    gt_dataloader_mifid = DataLoader(gt_full_dataset, batch_size=64, num_workers=0, shuffle=False)
    gen_dataloader_mifid = DataLoader(gen_subset, batch_size=64, num_workers=0, shuffle=False)
    
    # For Inception Score
    gen_dataloader_is = DataLoader(gen_subset, batch_size=64, num_workers=0, shuffle=False)
    
    # For CLIP Score
    gen_dataloader_clip = DataLoader(gen_subset, batch_size=64, num_workers=0, shuffle=False)
    
    # --- Initialize Metrics ---
    fid = FrechetInceptionDistance().to(device)
    fid.set_dtype(torch.float64)
    mifid = MemorizationInformedFrechetInceptionDistance().to(device)
    inception = InceptionScore(splits=10).to(device)
    clip_scores_all = []

    # --- 1. FID Calculation ---
    for batch in gen_dataloader_fid:
        try:
            fid.update(batch[0].to(device), real=False)
        except Exception as e:
            print(f"Error updating FID (gen): {e}")
            continue
    for batch in gt_dataloader_fid:
        try:
            fid.update(batch[0].to(device), real=True)
        except Exception as e:
            print(f"Error updating FID (gt): {e}")
            continue
    fid_metric = fid.compute().item()

    # --- 2. MIFID Calculation ---
    for batch in gen_dataloader_mifid:
        try:
            mifid.update(batch[0].to(device), real=False)
        except Exception as e:
            print(f"Error updating MIFID (gen): {e}")
            continue
    for batch in gt_dataloader_mifid:
        try:
            mifid.update(batch[0].to(device), real=True)
        except Exception as e:
            print(f"Error updating MIFID (gt): {e}")
            continue
    mifid_metric = mifid.compute().item()

    # --- 3. Inception Score Calculation ---
    for batch in gen_dataloader_is:
        try:
            inception.update(batch[0].to(device))
        except Exception as e:
            print(f"Error updating IS: {e}")
            continue
    inception_metric_tuple = inception.compute()
    inception_metric = inception_metric_tuple[0].item()

    # --- 4. CLIP Score Calculation ---
    for batch in gen_dataloader_clip:
        try:
            clip_scores = calculate_clip_score(*batch)
            clip_scores_all.append(clip_scores)
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            continue
            
    clip_scores_avg = 0.0
    if clip_scores_all:
        clip_scores_avg = sum(clip_scores_all) / len(clip_scores_all)

    return {'fid': fid_metric, 'mifid': mifid_metric, 'is': inception_metric, 'clip_score': clip_scores_avg}


def plot_imgs(actual_images, generated_images, column_names, filename):
    fig, axs = plt.subplots(2, len(column_names), figsize=(15, 6))

    for i, img in enumerate(actual_images):
        axs[0, i].imshow(img)
        axs[0, i].axis('off')  # Turn off axis
        if i == 0:
            axs[0, i].set_ylabel('Actual')

    for i, img in enumerate(generated_images):
        axs[1, i].imshow(img)
        axs[1, i].axis('off')  # Turn off axis
        if i == 0:
            axs[1, i].set_ylabel('Generated')

    for i in range(len(column_names)):
        axs[0, i].set_title(column_names[i], fontsize=6)

    plt.tight_layout()
    plt.savefig(f'grid_{filename}.png')


if __name__ == "__main__":
    args = parse_args()

    # Set seed for reproducible sampling
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
    ])

    # --- 1. Load and synchronize FULL datasets ---
    print("Loading and validating generated images...")
    # Load all valid generated images
    full_gen_dataset = CustomDataset(img_dir=args.gen_img_dir, 
                                     prompt_dir=args.prompt_dir, 
                                     transform=transform)
    print(f"Found {len(full_gen_dataset)} valid generated samples.")

    # Get the list of base filenames from the valid generated samples
    valid_basenames = full_gen_dataset.basenames

    print("Loading ground-truth images and synchronizing...")
    # Load GT images, filtered to *only* those that exist in the generated set
    full_gt_dataset = CustomDataset(img_dir=args.gt_img_dir, 
                                    prompt_dir=args.prompt_dir, 
                                    transform=transform, 
                                    filter_by_basenames=valid_basenames)
    print(f"Synchronized ground-truth dataset to {len(full_gt_dataset)} samples.")

    # Safety check
    assert len(full_gen_dataset) == len(full_gt_dataset)
    print("✅ Datasets synchronized successfully!")
    
    base_sample_size = len(full_gen_dataset)
    if base_sample_size == 0:
        print("Error: No matching image-prompt pairs found. Exiting.")
        exit()
        
    # Define proportions to evaluate
    proportions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # Added 1.0 for the full set
    all_results = []
    
    print(f'\nProcessing {args.gen_img_dir} for proportions: {proportions}')

    # --- 2. Loop through proportions and calculate metrics ---
    for prop in proportions:
        subset_size = int(base_sample_size * prop)
        if subset_size == 0:
            print(f"Skipping proportion {prop}: subset size is 0.")
            continue
            
        print(f"\n--- Calculating metrics for proportion: {prop} (Sample size: {subset_size}) ---")
        
        # Create random indices to sample the *generated* dataset
        indices = torch.randperm(base_sample_size)[:subset_size]
        
        # Create the subset for the generated images
        gen_subset = Subset(full_gen_dataset, indices)
        
        # Calculate metrics: comparing full GT (real) vs. gen_subset (fake)
        metrics_dict = calculate_all_metrics(full_gt_dataset, gen_subset, device)
        
        # Store results
        row_data = {
            'proportion': prop,
            'sample_size': subset_size,
            'fid': f"{metrics_dict['fid']:.2f}",
            'mifid': f"{metrics_dict['mifid']:.2f}",
            'is': f"{metrics_dict['is']:.2f}",
            'clip_score': f"{metrics_dict['clip_score']:.2f}"
        }
        all_results.append(row_data)
        
        print(f"Results for {prop}: {row_data}")

    # --- 3. Save all results to a CSV file ---
    
    # Use the parent directory of gen_img_dir for the output file
    parent_dir = os.path.dirname(args.gen_img_dir)
    if not parent_dir: # Handle case where gen_img_dir is a local path like 'gen_imgs/'
        parent_dir = '.'
        
    # Use the basename from the output argument, ensuring it's a .csv
    output_filename = os.path.basename(args.output)
    if not output_filename.endswith('.csv'):
        output_filename = f"{os.path.splitext(output_filename)[0]}.csv"
        
    result_path = os.path.join(parent_dir, output_filename)
    
    print(f"\nSaving all results to {result_path}")
    
    if all_results:
        headers = all_results[0].keys()
        with open(result_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)
    else:
        print("No results to save.")
        
    print("✅ Evaluation complete.")