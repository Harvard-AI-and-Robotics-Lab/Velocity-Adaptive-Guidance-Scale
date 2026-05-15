# python analysis/visualize_intermediate_noise.py --caption_folder /path/to/captions --output_folder /path/to/output
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from captions and save intermediate steps")
    parser.add_argument("--caption_folder", type=str, required=True, help="Folder containing caption files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save generated images")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of captions to process in parallel")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale for generation")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Step 1: Load the pipeline
    # model_id = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    if 'checkpoint' in args.model_path:
        pipe.load_lora_weights(args.model_path)
    pipe = pipe.to("cuda")
    
    # Get all caption files
    caption_files = [f for f in os.listdir(args.caption_folder) if f.endswith(".txt")]
    
    # Process each caption file
    # for caption_file in tqdm(caption_files, desc="Processing captions"):
    for idx, caption_file in enumerate(tqdm(caption_files, desc="Processing captions")):
        if idx > 0 and idx % 200 == 0:
            break
        # Extract ID from filename
        image_id = os.path.splitext(caption_file)[0]
        
        # Read the caption
        with open(os.path.join(args.caption_folder, caption_file), "r") as f:
            prompt = f.read().strip()
        
        # Save intermediate images for specific steps 
        # Note: diffusion steps are zero-indexed internally, so adjusting accordingly
        # We'll save steps 0, 9, 19, 29, 39, 49 which correspond to steps 1, 10, 20, 30, 40, 50
        steps_to_save = [0, 9, 19, 29, 39, 49]
        steps_to_display = [1, 10, 20, 30, 40, 50]  # For display purposes
        saved_images = {}
        
        # Define callback function to save intermediate steps
        def save_intermediate_latents(pipeline, step, timestep, callback_kwargs):
            # Only save the steps we're interested in
            if step in steps_to_save:
                # Extract latents from callback_kwargs
                latents = callback_kwargs['latents']
                
                # Decode latents to an image using the pipeline's VAE
                with torch.no_grad():
                    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                
                # Postprocess the image to PIL format
                image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
                
                # Save the image to our dictionary
                saved_images[step] = image
                
                print(f"Processed intermediate image for {image_id} at step {step+1} (internal step {step})")
            
            # Return callback_kwargs to prevent the error
            return callback_kwargs
        
        # Generate the image with the callback
        print(f"Generating image for caption: {prompt}")
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,  # Set to 50 as per requirement
            guidance_scale=args.guidance_scale,
            callback_on_step_end=save_intermediate_latents,
        ).images[0]
        
        # Save the final image
        final_image_path = os.path.join(args.output_folder, f"{image_id}.jpg")
        # image.save(final_image_path)
        print(f"Saved final image: {final_image_path}")
        
                    # Create a side-by-side comparison of the intermediate results
        if any(step in saved_images for step in steps_to_save):
            # Get the dimensions of the images (use the first available image)
            first_available_step = next(step for step in steps_to_save if step in saved_images)
            width, height = saved_images[first_available_step].size
            
            # Create a new image with enough space for all our images side by side
            combined_image = Image.new('RGB', (width * len(steps_to_save), height))
            
            # Paste each image into the combined image (if available)
            for i, step in enumerate(steps_to_save):
                if step in saved_images:
                    combined_image.paste(saved_images[step], (i * width, 0))
                else:
                    # Create a blank image with text indicating missing step
                    blank = Image.new('RGB', (width, height), color=(200, 200, 200))
                    combined_image.paste(blank, (i * width, 0))
                    
            # Also include the final image as the last step if it wasn't captured
            if 49 not in saved_images and len(steps_to_save) > 0:
                # Use the final generated image for the last position
                combined_image.paste(image, ((len(steps_to_save) - 1) * width, 0))
            
            # Save the combined image
            combined_image_path = os.path.join(args.output_folder, f"intermediate_{image_id}.jpg")
            combined_image.save(combined_image_path)
            print(f"Saved combined intermediate image: {combined_image_path}")

if __name__ == "__main__":
    main()