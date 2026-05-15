# python analysis/put_sim_on_images.py output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/coco17_caption_results_sim.json output_iccv/coco17_proposed_lambda_curva_01/checkpoint-26000/generated_images_coco17 data4paper/qual_examples/proposed
# python analysis/put_sim_on_images.py output_iccv/coco17_no_reweight/checkpoint-26000/coco17_caption_results_sim.json output_iccv/coco17_no_reweight/checkpoint-26000/generated_images_coco17 data4paper/qual_examples/no_reweight
# python analysis/put_sim_on_images.py output_iccv/coco17_lognorm/checkpoint-26000/coco17_caption_results_sim.json output_iccv/coco17_lognorm/checkpoint-26000/generated_images_coco17 data4paper/qual_examples/lognorm
# python analysis/put_sim_on_images.py output_iccv/coco17_modesample/checkpoint-26000/coco17_caption_results_sim.json output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco17 data4paper/qual_examples/modesample
# python analysis/put_sim_on_images.py output_iccv/coco17_cosmap/checkpoint-26000/coco17_caption_results_sim.json output_iccv/coco17_cosmap/checkpoint-26000/generated_images_coco17 data4paper/qual_examples/cosmap
import json
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from pathlib import Path

def process_captions_with_images(json_file_path, image_folder, output_folder="output_images"):
    """
    Process captions from a JSON file and add metrics to images with improved visual design.
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        if "captions" not in data:
            print("Error: JSON file does not contain 'captions' key")
            return
        
        # Create output directory if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_folder}")
        
        captions = data["captions"]
        print(f"Found {len(captions)} captions")
        
        # Try to load fonts (with fallbacks)
        # Font handling specifically optimized for Ubuntu
        try:
            # Ubuntu-specific font locations (comprehensive list)
            ubuntu_font_paths = [
                '/usr/share/fonts/truetype/dejavu/',
                '/usr/share/fonts/truetype/ubuntu/',
                '/usr/share/fonts/truetype/liberation/',
                '/usr/share/fonts/truetype/freefont/',
                '/usr/share/fonts/truetype/',
                '/usr/share/fonts/TTF/',
                '/usr/local/share/fonts/',
                os.path.expanduser('~/.fonts/'),
                os.path.expanduser('~/.local/share/fonts/'),
                ''  # Current directory
            ]
            
            # Fonts commonly available on Ubuntu
            ubuntu_font_names = [
                'DejaVuSans.ttf',
                'Ubuntu-R.ttf',
                'UbuntuMono-R.ttf',
                'FreeSans.ttf',
                'LiberationSans-Regular.ttf',
                'NotoSans-Regular.ttf',
                'OpenSans-Regular.ttf',
                'Lato-Regular.ttf'
            ]
            
            # Debug: List all ubuntu font paths to check if they exist
            print("Checking Ubuntu font paths:")
            for path in ubuntu_font_paths:
                if os.path.exists(path):
                    print(f"  Found directory: {path}")
                    # List first 5 font files in the directory
                    try:
                        files = [f for f in os.listdir(path) if f.endswith('.ttf') or f.endswith('.TTF')][:5]
                        if files:
                            print(f"    Contains font files: {', '.join(files)}")
                    except Exception as e:
                        print(f"    Error listing directory: {str(e)}")
                else:
                    print(f"  Directory not found: {path}")
            
            # Try to find and load a font
            font_found = False
            for path in ubuntu_font_paths:
                if not os.path.exists(path):
                    continue
                    
                # First try our preferred list
                for font_name in ubuntu_font_names:
                    try:
                        font_path = os.path.join(path, font_name)
                        if os.path.exists(font_path):
                            # Test loading to verify this works
                            title_font = ImageFont.truetype(font_path, 36)
                            heading_font = ImageFont.truetype(font_path, 32)
                            text_font = ImageFont.truetype(font_path, 28)
                            metrics_font = ImageFont.truetype(font_path, 30)
                            print(f"Successfully loaded font: {font_path}")
                            font_found = True
                            break
                    except Exception as e:
                        print(f"Error loading font {font_path}: {str(e)}")
                        continue
                
                # If no preferred font found, try any TTF in the directory
                if not font_found and os.path.exists(path):
                    try:
                        files = [f for f in os.listdir(path) if f.endswith('.ttf') or f.endswith('.TTF')]
                        for font_file in files:
                            try:
                                font_path = os.path.join(path, font_file)
                                title_font = ImageFont.truetype(font_path, 36)
                                heading_font = ImageFont.truetype(font_path, 32)
                                text_font = ImageFont.truetype(font_path, 36)
                                metrics_font = ImageFont.truetype(font_path, 30)
                                print(f"Successfully loaded font: {font_path}")
                                font_found = True
                                break
                            except Exception:
                                continue
                    except Exception:
                        pass
                
                if font_found:
                    break
            
            # If still no font found, try installing a font with apt
            if not font_found:
                print("No TrueType fonts found. Attempting to install fonts...")
                try:
                    # Include code to create an embedded font so we don't rely on system
                    from PIL import ImageFont, ImageDraw
                    # Using the default font with increased size
                    title_font = ImageFont.load_default()
                    heading_font = ImageFont.load_default()
                    text_font = ImageFont.load_default()
                    metrics_font = ImageFont.load_default()
                    print("Using PIL default font with increased size.")
                    font_found = True
                except Exception as e:
                    print(f"Error using default font: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Warning: Font loading error: {str(e)} - using default font")
            # Use default bitmap font as last resort
            title_font = ImageFont.load_default()
            heading_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            metrics_font = ImageFont.load_default()
        
        # Process each image
        for image_id, caption_data in captions.items():
            # Find image file (try both jpg and png)
            image_path = Path(image_folder) / f"{image_id}.jpg"
            if not image_path.exists():
                image_path = Path(image_folder) / f"{image_id}.png"
                if not image_path.exists():
                    print(f"Warning: Image file for {image_id} not found. Skipping.")
                    continue
            
            try:
                print(f"Processing image: {image_id}")
                img = Image.open(image_path)
                
                # Get original image dimensions
                width, height = img.size
                
                # Calculate new dimensions with slightly larger caption area for bigger font
                original_ratio = height / width
                caption_height = int(width * 0.15)  # Increased from 0.3 to accommodate larger font
                new_height = height + caption_height
                
                # Create new image with gradient background
                new_img = Image.new('RGB', (width, new_height), color=(245, 245, 245))
                new_img.paste(img, (0, 0))
                
                # Create drawing context
                draw = ImageDraw.Draw(new_img)
                
                # Add a subtle gradient transition between image and text area
                for y in range(20):
                    alpha = y / 20
                    color = (
                        int(245 * alpha + 0 * (1 - alpha)),
                        int(245 * alpha + 0 * (1 - alpha)),
                        int(245 * alpha + 0 * (1 - alpha))
                    )
                    draw.line([(0, height - 20 + y), (width, height - 20 + y)], fill=color)
                
                # Define metrics
                bleu = caption_data.get('BLEU-4', 0)
                meteor = caption_data.get('METEOR', 0)
                rouge = caption_data.get('ROUGE-L', 0)
                
                # Calculate color based on metrics average
                avg_metric = (bleu + meteor + rouge) / 3
                # Color gradient from red (low) to green (high)
                metric_color = (
                    int(255 * (1 - avg_metric)),  # R
                    int(255 * avg_metric),        # G
                    0                            # B
                )
                
                # Define text content
                generated_text = caption_data.get('generated', 'No generated caption')
                ground_truth_text = caption_data.get('ground_truth', 'No ground truth caption')
                
                # Define minimal spacing and padding
                padding = 10
                x_start = padding
                section_gap = 10
                text_y_start = height + padding
                
                # Draw caption boxes with improved visual design
                # Start position right after the image
                current_y = height + padding
                
                # Generated Caption - combined header and content
                # Text content with maximized width usage
                max_width = width - (padding * 2)
                wrap_width = int(max_width / 1)  # More chars per line to use full width
                generated_text_with_header = f"Generated Caption: {generated_text}"
                generated_wrapped = textwrap.fill(generated_text_with_header, width=wrap_width)
                
                # First line in blue (for the header part)
                first_line = generated_wrapped.split('\n')[0]
                header_end_index = first_line.find(': ') + 2  # Position after ": "
                
                if header_end_index > 1:  # If header is found
                    # Draw header part in blue
                    draw.text(
                        (x_start, current_y),
                        first_line[:header_end_index],
                        font=text_font,
                        fill=(0, 0, 100)  # Dark blue
                    )
                    
                    # Draw content part in black
                    header_width = text_font.getbbox(first_line[:header_end_index])[2]
                    draw.text(
                        (x_start + header_width, current_y),
                        first_line[header_end_index:],
                        font=text_font,
                        fill=(0, 0, 0)  # Black
                    )
                    
                    # Draw remaining lines if any
                    remaining_lines = generated_wrapped.split('\n')[1:]
                    line_height = text_font.getbbox('A')[3]
                    
                    for i, line in enumerate(remaining_lines):
                        draw.text(
                            (x_start, current_y + ((i + 1) * line_height)),
                            line,
                            font=text_font,
                            fill=(0, 0, 0)  # Black
                        )
                    
                    # Update y position
                    line_count = len(generated_wrapped.split('\n'))
                    current_y += (line_count * line_height) + (section_gap // 2)
                
                # Ground Truth Caption - combined header and content
                ground_truth_text_with_header = f"Ground Truth Caption: {ground_truth_text}"
                ground_truth_wrapped = textwrap.fill(ground_truth_text_with_header, width=wrap_width)
                
                # First line with colored header
                first_line = ground_truth_wrapped.split('\n')[0]
                header_end_index = first_line.find(': ') + 2  # Position after ": "
                
                if header_end_index > 1:  # If header is found
                    # Draw header part in green
                    draw.text(
                        (x_start, current_y),
                        first_line[:header_end_index],
                        font=text_font,
                        fill=(0, 100, 0)  # Dark green
                    )
                    
                    # Draw content part in black
                    header_width = text_font.getbbox(first_line[:header_end_index])[2]
                    draw.text(
                        (x_start + header_width, current_y),
                        first_line[header_end_index:],
                        font=text_font,
                        fill=(0, 0, 0)  # Black
                    )
                    
                    # Draw remaining lines if any
                    remaining_lines = ground_truth_wrapped.split('\n')[1:]
                    line_height = text_font.getbbox('A')[3]
                    
                    for i, line in enumerate(remaining_lines):
                        draw.text(
                            (x_start, current_y + ((i + 1) * line_height)),
                            line,
                            font=text_font,
                            fill=(0, 0, 0)  # Black
                        )
                    
                    # Update y position
                    line_count = len(ground_truth_wrapped.split('\n'))
                    current_y += (line_count * line_height) + (section_gap // 2)
                
                # Metrics with header and proper spacing
                # Draw header part in red
                metrics_header = "Similarity Metrics: "
                draw.text(
                    (x_start, current_y),
                    metrics_header,
                    font=text_font,
                    fill=(100, 0, 0)  # Dark red
                )
                
                # Calculate positions for properly spaced metrics
                header_width = text_font.getbbox(metrics_header)[2]
                metric_x = x_start + header_width
                
                # Calculate total available width
                available_width = width - (padding * 2) - header_width
                
                # Fixed width for each metric to prevent overlap
                metric_width = available_width / 3  # Divide space into 3 equal parts
                
                # Draw each metric with proper spacing
                metrics = [
                    ("BLEU-4", bleu),
                    ("METEOR", meteor),
                    ("ROUGE-L", rouge)
                ]
                
                for i, (name, value) in enumerate(metrics):
                    # Position for this metric
                    pos_x = metric_x + (i * metric_width)
                    
                    # Use a beautiful constant color regardless of value
                    # Deep blue-purple - a visually appealing color for all metrics
                    color = (75, 92, 196)  # Rich indigo blue
                    
                    # Format metric text
                    metric_text = f"{name}: {value:.2f}"
                    
                    # Draw metric name in black
                    name_width = text_font.getbbox(f"{name}: ")[2]
                    draw.text(
                        (pos_x, current_y),
                        f"{name}: ",
                        font=text_font,
                        fill=(0, 0, 0)  # Black
                    )
                    
                    # Draw value in color
                    draw.text(
                        (pos_x + name_width, current_y),
                        f"{value:.2f}",
                        font=text_font,
                        fill=color  # Beautiful color based on value
                    )
                
                # Update position
                current_y += text_font.getbbox(metrics_header)[3] + 10
                
                # Save the annotated image with high quality
                output_path = Path(output_folder) / f"{image_id}_annotated.png"
                new_img.save(output_path, format="PNG", quality=95)
                print(f"Saved annotated image to: {output_path}")
                
            except Exception as e:
                print(f"Error processing image {image_id}: {str(e)}")
        
        print("Processing complete!")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: '{json_file_path}' is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python put_sim_on_images.py <path_to_json_file> <path_to_image_folder> <path_to_output_folder>")
        return
    
    json_file_path = sys.argv[1]
    image_folder = sys.argv[2]
    output_folder = sys.argv[3]
    
    process_captions_with_images(json_file_path, image_folder, output_folder)

if __name__ == "__main__":
    main()