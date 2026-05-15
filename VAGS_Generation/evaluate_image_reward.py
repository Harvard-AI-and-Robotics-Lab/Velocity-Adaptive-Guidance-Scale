#!/usr/bin/env python3
"""
Script to compute ImageReward scores for a set of images and their corresponding captions.
ImageReward is a human preference reward model for text-to-image evaluation.

Usage:
    python compute_image_reward.py --image_dir path/to/images --caption_dir path/to/captions [--output results.json] [--batch_size 8]
"""

import os
import argparse
import json
from glob import glob
import torch
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Compute ImageReward scores for images and captions")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing image files ([id].png, [id].jpg, etc.)")
    parser.add_argument("--caption_dir", type=str, required=True, help="Directory containing caption files ([id].txt)")
    parser.add_argument("--output", type=str, default="image_reward_scores.json", help="Output JSON file for results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    return parser.parse_args()

def find_image_caption_pairs(image_dir, caption_dir):
    """Find matching image and caption pairs based on file ID."""
    # Get all image files
    image_extensions = ["png", "jpg", "jpeg", "webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, f"*.{ext}")))
    
    # Extract IDs from image filenames
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    image_dict = dict(zip(image_ids, image_files))
    
    # Get all caption files
    caption_files = glob(os.path.join(caption_dir, "*.txt"))
    caption_ids = [os.path.splitext(os.path.basename(f))[0] for f in caption_files]
    caption_dict = dict(zip(caption_ids, caption_files))
    
    # Find matching pairs
    pairs = []
    for id in set(image_ids).intersection(set(caption_ids)):
        pairs.append((id, image_dict[id], caption_dict[id]))
    
    return pairs

def read_caption(caption_file):
    """Read caption from a text file."""
    with open(caption_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def compute_scores(pairs, batch_size=8):
    """Compute ImageReward scores for image-caption pairs."""
    try:
        import ImageReward as reward
    except ImportError:
        raise ImportError("ImageReward package not found. Install it with: pip install image-reward")
    
    # Load the model
    model = reward.load("ImageReward-v1.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}
    
    # Process in batches
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        batch_ids = [p[0] for p in batch_pairs]
        batch_images = [p[1] for p in batch_pairs]
        batch_captions = [read_caption(p[2]) for p in batch_pairs]
        
        # Compute scores
        with torch.no_grad():
            for j, (id, image, caption) in enumerate(zip(batch_ids, batch_images, batch_captions)):
                try:
                    score = model.score(caption, image)
                    results[id] = float(score)
                except Exception as e:
                    print(f"Error processing {id}: {e}")
                    results[id] = None
    
    return results

def main():
    args = parse_args()
    print(f"Finding image-caption pairs in {args.image_dir} and {args.caption_dir}...")
    pairs = find_image_caption_pairs(args.image_dir, args.caption_dir)
    print(f"Found {len(pairs)} image-caption pairs.")
    
    if not pairs:
        print("No matching image-caption pairs found. Please check directory paths.")
        return
    
    print("Computing ImageReward scores...")
    scores = compute_scores(pairs, args.batch_size)
    
    # Calculate statistics
    valid_scores = [s for s in scores.values() if s is not None]
    if valid_scores:
        avg_score = np.mean(valid_scores)
        median_score = np.median(valid_scores)
        min_score = np.min(valid_scores)
        max_score = np.max(valid_scores)
        
        print(f"Results:")
        print(f"Average score: {avg_score:.4f}")
        print(f"Median score: {median_score:.4f}")
        print(f"Min score: {min_score:.4f}")
        print(f"Max score: {max_score:.4f}")
        
        # Save results
        output = {
            "individual_scores": scores,
            "statistics": {
                "count": len(valid_scores),
                "average": float(avg_score),
                "median": float(median_score),
                "min": float(min_score),
                "max": float(max_score)
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {args.output}")
    else:
        print("No valid scores computed.")

if __name__ == "__main__":
    main()
