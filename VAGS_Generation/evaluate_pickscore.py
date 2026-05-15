#!/usr/bin/env python3
"""
PickScore Evaluator

This script computes PickScore for a collection of image-caption pairs and calculates
the average score. It takes an image folder containing image files with format [id].png
and a caption folder containing caption files with format [id].txt.

PickScore is a scoring function for evaluating the quality of images generated from text prompts,
finetuned from CLIP-H using the Pick-a-Pic dataset.
"""

import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Compute PickScore for image-caption pairs")
    parser.add_argument(
        "--image_folder", 
        type=str, 
        required=True,
        help="Path to folder containing image files ([id].png, [id].jpg, etc.)"
    )
    parser.add_argument(
        "--caption_folder", 
        type=str, 
        required=True,
        help="Path to folder containing caption files ([id].txt)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="pickscore_results.txt",
        help="File to save results to"
    )
    return parser.parse_args()

def load_models(device):
    """Load the PickScore model and processor"""
    print("Loading models...")
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    
    return processor, model

def get_image_caption_pairs(image_folder, caption_folder):
    """
    Find all matching image-caption pairs
    Returns a list of tuples (image_path, caption_path)
    """
    pairs = []
    
    # Get all image files and their base IDs
    image_files = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # Extract the ID (everything before the extension)
            file_id = os.path.splitext(filename)[0]
            image_files[file_id] = os.path.join(image_folder, filename)
    
    # Find matching caption files
    for file_id, image_path in image_files.items():
        caption_path = os.path.join(caption_folder, f"{file_id}.txt")
        if os.path.exists(caption_path):
            pairs.append((image_path, caption_path))
    
    print(f"Found {len(pairs)} image-caption pairs")
    return pairs

def batch_items(items, batch_size):
    """Split items into batches of size batch_size"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def compute_pickscore(processor, model, image_path, caption, device):
    """
    Compute PickScore for a single image-caption pair
    Returns a scalar value representing the score
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs
    image_inputs = processor(
        images=image,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=caption,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        # Get embeddings
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
        # Compute score
        score = model.logit_scale.exp() * (text_embs @ image_embs.T)[0][0]
        
    return score.cpu().item()

def compute_batch_pickscore(processor, model, batch_pairs, device):
    """
    Compute PickScore for a batch of image-caption pairs
    Returns a list of scores
    """
    batch_images = []
    batch_captions = []
    
    # Load images and captions
    for image_path, caption_path in batch_pairs:
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        image = Image.open(image_path).convert("RGB")
        
        batch_images.append(image)
        batch_captions.append(caption)
    
    # Process inputs
    image_inputs = processor(
        images=batch_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=batch_captions,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        # Get embeddings
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
        # Calculate similarity scores for each image-caption pair
        scores = []
        for i in range(len(batch_pairs)):
            # Compute one-to-one scores
            score = model.logit_scale.exp() * torch.dot(text_embs[i], image_embs[i])
            scores.append(score.cpu().item())
        
    return scores

def main():
    args = parse_args()
    
    # Load models
    processor, model = load_models(args.device)
    
    # Get image-caption pairs
    pairs = get_image_caption_pairs(args.image_folder, args.caption_folder)
    
    if not pairs:
        print("No matching image-caption pairs found. Exiting.")
        return
    
    # Compute scores
    all_scores = []
    all_file_ids = []
    
    print(f"Computing PickScore for {len(pairs)} image-caption pairs...")
    for batch in tqdm(list(batch_items(pairs, args.batch_size))):
        try:
            batch_scores = compute_batch_pickscore(processor, model, batch, args.device)
            all_scores.extend(batch_scores)
            
            # Extract file IDs for reporting
            for image_path, _ in batch:
                file_id = os.path.splitext(os.path.basename(image_path))[0]
                all_file_ids.append(file_id)
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    # Calculate average score
    if all_scores:
        average_score = np.mean(all_scores)
        print(f"Average PickScore: {average_score:.4f}")
        
        # Save detailed results
        with open(args.output_file, 'w') as f:
            f.write(f"Average PickScore: {average_score:.4f}\n\n")
            f.write("Individual scores:\n")
            for file_id, score in zip(all_file_ids, all_scores):
                f.write(f"{file_id}: {score:.4f}\n")
        
        print(f"Results saved to {args.output_file}")
    else:
        print("No scores were calculated. Check for errors above.")

if __name__ == "__main__":
    main()