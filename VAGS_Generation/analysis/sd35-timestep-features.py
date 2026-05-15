import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Step 1: Load the pipeline
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Step 2: Define the callback function to save intermediate latents
def save_intermediate_latents(pipeline, step, timestep, callback_kwargs):
    # Extract latents from callback_kwargs
    latents = callback_kwargs['latents']
    
    # Decode latents to an image using the pipeline's VAE
    with torch.no_grad():
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
    # Postprocess the image to PIL format
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    # Save the image
    save_path = f"intermediate_step_{step}.jpg"
    image.save(save_path)
    print(f"Saved intermediate image at step {step}: {save_path}")
    
    # Return callback_kwargs to prevent the error
    return callback_kwargs

# Step 3: Generate the image with the callback
prompt = "A serene landscape with mountains and a lake"
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=7.0,
    callback_on_step_end=save_intermediate_latents,
).images[0]

# Step 4: Save the final image
image.save("final_image.jpg")
print("Saved final image: final_image.jpg")