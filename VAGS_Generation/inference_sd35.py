import argparse
import os
import time
import torch
from typing import Optional, Dict

from diffusers import (
    StableDiffusion3Pipeline,
    # classic diffusion-family schedulers
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
from torch.utils.data import Dataset, DataLoader

# --- Try to import Flow Matching schedulers (available in newer diffusers) ---
FM_EULER = None
FM_HEUN = None
try:
    from diffusers import FlowMatchEulerDiscreteScheduler as _FM_EULER_CLS  # type: ignore
    FM_EULER = _FM_EULER_CLS
except Exception:
    FM_EULER = None

try:
    from diffusers import FlowMatchHeunDiscreteScheduler as _FM_HEUN_CLS  # type: ignore
    FM_HEUN = _FM_HEUN_CLS
except Exception:
    FM_HEUN = None


def _available(cls) -> bool:
    return cls is not None


def _collect_scheduler_aliases() -> Dict[str, str]:
    """Build alias map based on what's importable in this environment."""
    aliases = {
        # diffusion-family (score-matching) schedulers
        "euler": "EulerDiscreteScheduler",
        "euler_discrete": "EulerDiscreteScheduler",
        "euler_ancestral": "EulerAncestralDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpmpp": "DPMSolverMultistepScheduler",
        "dpmpp_2m": "DPMSolverMultistepScheduler",
        "dpmsolver": "DPMSolverMultistepScheduler",
        "heun": "HeunDiscreteScheduler",
        "lms": "LMSDiscreteScheduler",
        "unipc": "UniPCMultistepScheduler",
        "ddim": "DDIMScheduler",
        "pndm": "PNDMScheduler",
    }
    # flow-matching schedulers (added only if installed)
    if _available(FM_EULER):
        aliases.update({
            "flowmatch_euler": "FlowMatchEulerDiscreteScheduler",
            "fm_euler": "FlowMatchEulerDiscreteScheduler",
            "flow_euler": "FlowMatchEulerDiscreteScheduler",
            "flowmatch": "FlowMatchEulerDiscreteScheduler",  # convenient default
        })
    if _available(FM_HEUN):
        aliases.update({
            "flowmatch_heun": "FlowMatchHeunDiscreteScheduler",
            "fm_heun": "FlowMatchHeunDiscreteScheduler",
            "flow_heun": "FlowMatchHeunDiscreteScheduler",
        })
    return aliases


SCHEDULER_ALIASES = _collect_scheduler_aliases()


def get_scheduler(name: str, pipe_scheduler_config):
    name = (name or "").lower()
    target = SCHEDULER_ALIASES.get(name)
    if target is None:
        fm_options = [k for k in SCHEDULER_ALIASES.keys() if k.startswith("flow")]
        all_opts = ", ".join(sorted(SCHEDULER_ALIASES.keys()))
        msg = (f"Unknown --scheduler '{name}'. Choose one of: {all_opts}."
               f"{' (Flow-matching options: ' + ', '.join(sorted(fm_options)) + ')' if fm_options else ''}")
        raise ValueError(msg)

    # Map class name to actual constructor
    if target == "EulerDiscreteScheduler":
        return EulerDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "EulerAncestralDiscreteScheduler":
        return EulerAncestralDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "DPMSolverMultistepScheduler":
        return DPMSolverMultistepScheduler.from_config(pipe_scheduler_config)
    if target == "HeunDiscreteScheduler":
        return HeunDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "LMSDiscreteScheduler":
        return LMSDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "UniPCMultistepScheduler":
        return UniPCMultistepScheduler.from_config(pipe_scheduler_config)
    if target == "DDIMScheduler":
        return DDIMScheduler.from_config(pipe_scheduler_config)
    if target == "PNDMScheduler":
        return PNDMScheduler.from_config(pipe_scheduler_config)
    if target == "FlowMatchEulerDiscreteScheduler":
        if FM_EULER is None:
            raise RuntimeError("FlowMatchEulerDiscreteScheduler not available in this diffusers version.")
        return FM_EULER.from_config(pipe_scheduler_config)  # type: ignore
    if target == "FlowMatchHeunDiscreteScheduler":
        if FM_HEUN is None:
            raise RuntimeError("FlowMatchHeunDiscreteScheduler not available in this diffusers version.")
        return FM_HEUN.from_config(pipe_scheduler_config)  # type: ignore
    raise RuntimeError("Internal error: unmapped scheduler target")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with SD3.5 and a selectable scheduler, including flow-matching compatible options."
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="Hugging Face token")

    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing .txt prompts")
    parser.add_argument("--split", type=str, default="test", help="train | val | validation | dev | test")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to LoRA or checkpoints (if contains 'checkpoint', load as LoRA)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output images")

    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale (CFG)")
    parser.add_argument("--image_width", type=int, default=512, help="Output width")
    parser.add_argument("--image_height", type=int, default=512, help="Output height")

    parser.add_argument("--num_prompts_per_run", type=int, default=1,
                        help="Number of prompts to process at a time from each batch")
    parser.add_argument("--reverse_mode", action="store_true",
                        help="Reverse order of prompts (default: False)")

    # Default to flowmatch_euler if available, else fall back to the model's default-like choice
    default_sched = "flowmatch_euler" if "flowmatch_euler" in SCHEDULER_ALIASES else "dpmpp"
    parser.add_argument("--scheduler", type=str, default=default_sched,
                        help=("Scheduler to use. Available: " + ", ".join(sorted(SCHEDULER_ALIASES.keys()))))
    parser.add_argument("--seed", type=int, default=227, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()
    return args


class PromptDataset(Dataset):
    def __init__(self, data_folder: str, reverse_mode: bool = False):
        super().__init__()
        self.data_folder = data_folder
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".txt")]
        if reverse_mode:
            self.file_paths = self.file_paths[::-1]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "r") as fh:
            content = fh.read().strip()
        file_idx = os.path.basename(file_path).split(".")[0]
        return content, file_idx


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()

    # Infer split from prompt_dir path if present
    for split in ["train", "val", "validation", "dev", "test"]:
        if split in args.prompt_dir:
            args.split = split
            break

    set_seed(args.seed)
    start_time = time.time()

    dataset = PromptDataset(args.prompt_dir, reverse_mode=args.reverse_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Compose final output directory
    # args.output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        # token=args.huggingface_token,  # Uncomment if needed for private models
    )

    # Select scheduler from CLI
    if args.scheduler != "none":
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # Optionally load LoRA weights if path is a HF checkpoint
    if "checkpoint" in args.model_path:
        pipe.load_lora_weights(args.model_path)

    pipe = pipe.to(args.device)

    # Generate images
    for batch in dataloader:
        batch_prompts, batch_prompt_ids = batch
        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts, batch_prompt_ids):
                out_path = f"{args.output_dir}/{_id}.png"
                if os.path.exists(out_path):
                    print(f"Skipping {out_path}")
                    continue
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                )
                image = result.images[0]
                image.save(out_path)

        elif args.num_prompts_per_run > 1:
            prompts_per_run = [batch_prompts[i:i + args.num_prompts_per_run]
                               for i in range(0, len(batch_prompts), args.num_prompts_per_run)]
            prompt_ids_per_run = [batch_prompt_ids[i:i + args.num_prompts_per_run]
                                  for i in range(0, len(batch_prompt_ids), args.num_prompts_per_run)]

            for prompts, prompt_ids in zip(prompts_per_run, prompt_ids_per_run):
                paths = [f"{args.output_dir}/{_id}.png" for _id in prompt_ids]
                if all(os.path.exists(p) for p in paths):
                    print(f"Skipping {paths}")
                    continue
                result = pipe(
                    prompt=list(prompts),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                )
                for image, _id in zip(result.images, prompt_ids):
                    image.save(f"{args.output_dir}/{_id}.png")
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}.")

    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600.0
    with open(os.path.join(args.output_dir, "runtime_inference.log"), "w") as f:
        f.write(f"Runtime duration: {duration_hours:.4f} hours\n")


if __name__ == "__main__":
    main()
