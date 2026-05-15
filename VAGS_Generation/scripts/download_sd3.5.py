from diffusers import StableDiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-3.5-large"

# Load the scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token='')

# Load the pipeline with the scheduler
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
