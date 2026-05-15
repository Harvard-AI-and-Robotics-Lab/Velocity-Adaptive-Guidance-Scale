# python scripts/merge_image_text.py --json_file output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/coco17_clair_results.json --img_folder output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/generated_images_coco17 --output_folder data4paper/image_caption/proposed
import json
import os
from PIL import Image, ImageDraw, ImageFont

json_file = 'output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/coco17_clair_results.json'
# Load JSON data from file (replace with json.loads() if using a string)
with open(json_file, 'r') as file:
    data = json.load(file)

# Function to process and save images with scores >= 90
def process_high_similarity_images(image_folder, output_folder, threshold=90):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Font settings (adjust as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 100)  # Use a common font, adjust size as needed
    except:
        font = ImageFont.load_default()  # Fallback to default font if Arial is unavailable
    
    # Traverse all caption evaluations
    for image_id, evaluation in data['caption_evaluations'].items():
        similarity_score = evaluation['similarity']['score']
        if similarity_score >= threshold:  # Include scores >= 90
            # Construct image file path
            image_path = os.path.join(image_folder, f"{image_id}.png")
            output_path = os.path.join(output_folder, f"{image_id}.png")
            
            # Load image
            try:
                img = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # Get candidate caption and score
            candidate_caption = f"BLIP Caption: {evaluation['candidate']}"
            score_text = f"CLAIR: {similarity_score}"
            
            # Create drawing context
            draw = ImageDraw.Draw(img)
            
            # Calculate text sizes
            caption_bbox = draw.textbbox((0, 0), candidate_caption, font=font)
            score_bbox = draw.textbbox((0, 0), score_text, font=font)
            caption_width = caption_bbox[2] - caption_bbox[0]
            score_width = score_bbox[2] - score_bbox[0]
            text_height = max(caption_bbox[3] - caption_bbox[1], score_bbox[3] - score_bbox[1])
            
            # Position text at the bottom of the image, side-by-side
            img_width, img_height = img.size
            padding = 10
            total_text_width = caption_width + score_width + padding
            x_start = (img_width - total_text_width) // 2  # Center horizontally
            y_pos = img_height - text_height - padding  # Near the bottom
            
            # Draw background rectangle for better readability (optional)
            draw.rectangle(
                [(x_start - padding, y_pos - padding), 
                 (x_start + total_text_width + padding, y_pos + text_height + padding)],
                fill=(0, 0, 0, 128)  # Semi-transparent black
            )
            
            # Draw candidate caption
            draw.text((x_start, y_pos), candidate_caption, font=font, fill="white")
            
            # Draw score next to caption
            draw.text((x_start + caption_width + padding, y_pos), score_text, font=font, fill="white")
            
            # Save modified image
            try:
                img.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error saving image {output_path}: {e}")
            
            # Close the image
            img.close()

# Main execution
if __name__ == "__main__":
    # Example folder paths (replace with your actual paths)
    image_folder = "output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/generated_images_coco17"  # e.g., "C:/images"
    output_folder = "data4paper/image_caption/proposed"  # e.g., "C:/output"
     
    # Process images
    process_high_similarity_images(image_folder, output_folder)