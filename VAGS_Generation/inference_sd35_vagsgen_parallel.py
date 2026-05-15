"""
Inference with SD3.5 using VAG-Gen (Velocity-Adaptive Guidance Scale for
Generation).

VAG-Gen replaces the constant classifier-free guidance (CFG) scale lambda with
a per-step adaptive scale

    lambda_i = lambda * exp( kappa * (2*sigma_i - 1) * s_i ),

where
    sigma_i = 1 - t_i           (signal level, monotone in denoising progress)
    s_i     = cos( v_uncond, v_cond )   (cosine sim of the two CFG velocities)

Setting kappa = 0 recovers standard (constant) CFG.

Implementation strategy
-----------------------
The stock StableDiffusion3Pipeline applies a fixed guidance_scale internally.
We subclass it and override __call__ so that, inside the denoising loop, we:
  1. run the usual single batched-of-two forward pass to get (v_uncond, v_cond)
  2. compute sigma_i, s_i, and lambda_i (per-sample)
  3. assemble the guided velocity with lambda_i
  4. advance one scheduler (Euler / flow-match) step
The scheduler itself is untouched; only the guidance combination changes.
This adds one inner-product per step and no extra forward passes.
"""

import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

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
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)

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


# ---------------------------------------------------------------------------
# VAG-Gen pipeline
# ---------------------------------------------------------------------------
class VAGGenSD3Pipeline(StableDiffusion3Pipeline):
    """
    Stable Diffusion 3 pipeline that replaces the constant CFG scale with the
    per-step adaptive VAG-Gen scale.

    Extra call-time kwargs
    ----------------------
    vag_kappa : float
        Modulation strength kappa. kappa = 0 reduces to standard CFG.
    vag_enabled : bool
        If False, falls back to standard CFG (equivalent to kappa = 0, but
        skips the cosine computation).
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        # --- VAG-Gen specific ---
        vag_kappa: float = 0.0,
        vag_enabled: bool = True,
        **kwargs,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs (same as the stock pipeline)
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Figure out batching
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Standard CFG active iff guidance_scale > 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=(
                joint_attention_kwargs.get("scale", None)
                if joint_attention_kwargs is not None
                else None
            ),
        )

        if do_classifier_free_guidance:
            # Concatenate [uncond; cond] along the batch dimension.
            prompt_embeds_cat = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_cat = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        else:
            prompt_embeds_cat = prompt_embeds
            pooled_prompt_embeds_cat = pooled_prompt_embeds

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Normalising constant so that t_normalised lies in [0, 1].
        # FlowMatchEulerDiscreteScheduler stores timesteps in [0, num_train_timesteps)
        # with num_train_timesteps = 1000; a plain EulerDiscreteScheduler also
        # uses the same convention. We read it from the scheduler config.
        t_max = float(getattr(self.scheduler.config, "num_train_timesteps", 1000))

        # 5. Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop with VAG-Gen
        effective_batch = batch_size * num_images_per_prompt
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand the latents for CFG: [uncond; cond]
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                else:
                    latent_model_input = latents

                # broadcast timestep across batch
                timestep_in = t.expand(latent_model_input.shape[0])

                # Single forward pass yielding (v_uncond, v_cond)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_in,
                    encoder_hidden_states=prompt_embeds_cat,
                    pooled_projections=pooled_prompt_embeds_cat,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Guidance combination
                if do_classifier_free_guidance:
                    v_uncond, v_cond = noise_pred.chunk(2, dim=0)

                    # ---- VAG-Gen adaptive scale ----
                    if vag_enabled and vag_kappa != 0.0:
                        # Per-sample cosine similarity s_i in [-1, 1].
                        # Flatten each sample to a vector and compute cosine.
                        v_u_flat = v_uncond.reshape(v_uncond.shape[0], -1).float()
                        v_c_flat = v_cond.reshape(v_cond.shape[0], -1).float()
                        s_i = torch.nn.functional.cosine_similarity(
                            v_u_flat, v_c_flat, dim=1, eps=1e-8
                        )  # shape: (effective_batch,)

                        # Signal level sigma_i = 1 - t_i (t normalised to [0,1]).
                        # t is a scalar tensor on `device`.
                        t_norm = (t.float() / t_max).to(s_i.device)
                        sigma_i = 1.0 - t_norm  # scalar in [0,1]

                        # lambda_i  (per-sample)
                        exponent = float(vag_kappa) * (2.0 * sigma_i - 1.0) * s_i
                        lambda_i = guidance_scale * torch.exp(exponent)  # shape: (B,)

                        # Reshape to broadcast against (B, C, H, W)
                        lambda_i = lambda_i.to(v_uncond.dtype).view(-1, 1, 1, 1)
                        noise_pred = v_uncond + lambda_i * (v_cond - v_uncond)
                    else:
                        # Standard CFG
                        noise_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

                # Advance one ODE / scheduler step (Euler for flow-match)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    pooled_prompt_embeds = callback_outputs.pop(
                        "pooled_prompt_embeds", pooled_prompt_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 7. Decode
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)


# ---------------------------------------------------------------------------
# CLI / data plumbing (parallels the baseline)
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run SD3.5 inference with VAG-Gen adaptive CFG guidance."
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="Hugging Face token")

    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing .txt prompts")
    parser.add_argument("--split", type=str, default="test", help="train | val | validation | dev | test")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to LoRA or checkpoints (if contains 'checkpoint', load as LoRA)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output images")

    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Base guidance scale lambda (used as the nominal CFG scale).")
    parser.add_argument("--image_width", type=int, default=512, help="Output width")
    parser.add_argument("--image_height", type=int, default=512, help="Output height")

    parser.add_argument("--num_prompts_per_run", type=int, default=1,
                        help="Number of prompts to process at a time from each batch")
    parser.add_argument("--reverse_mode", action="store_true",
                        help="Reverse order of prompts (default: False)")

    default_sched = "flowmatch_euler" if "flowmatch_euler" in SCHEDULER_ALIASES else "dpmpp"
    parser.add_argument("--scheduler", type=str, default=default_sched,
                        help=("Scheduler to use. Available: " + ", ".join(sorted(SCHEDULER_ALIASES.keys()))))
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    # -- VAG-Gen options --
    parser.add_argument("--vag_kappa", type=float, default=0.5,
                        help=("Modulation strength kappa for VAG-Gen. "
                              "kappa=0 recovers standard constant CFG. "
                              "Ignored when --kappas is provided."))
    parser.add_argument("--disable_vag", action="store_true",
                        help="Disable VAG-Gen and use standard constant CFG.")

    # -- Multi-GPU parallel sweep --
    parser.add_argument("--kappas", type=float, nargs="+", default=None,
                        help=("List of kappa values to run in parallel, one per "
                              "GPU in --gpus. Each run writes to its own "
                              "subfolder '<output_dir>_k{kappa}'. If omitted, "
                              "a single run is performed with --vag_kappa."))
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help=("List of CUDA device indices, one per kappa in "
                              "--kappas. Must have the same length as --kappas."))

    args = parser.parse_args()
    return args


class PromptDataset(Dataset):
    def __init__(self, data_folder: str, reverse_mode: bool = False):
        super().__init__()
        self.data_folder = data_folder
        self.file_paths = [os.path.join(data_folder, f)
                           for f in os.listdir(data_folder) if f.endswith(".txt")]
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


def run_experiment(args):
    """Execute a single VAG-Gen inference run on whatever device `args.device`
    resolves to. This is the body of what used to be `main()`, extracted so
    parallel workers can call it after configuring their GPU.

    `args` must already have its `vag_kappa`, `output_dir`, and `device`
    finalised. This function does not re-parse CLI arguments.
    """
    import torch  # local import so workers pick up the re-initialised module

    # Infer split from prompt_dir path if present
    for split in ["train", "val", "validation", "dev", "test"]:
        if split in args.prompt_dir:
            args.split = split
            break

    set_seed(args.seed)
    start_time = time.time()

    dataset = PromptDataset(args.prompt_dir, reverse_mode=args.reverse_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=False)

    # Compose final output directory
    # args.output_dir = os.path.join(args.output_dir, 'coco17_sd35_vagsgen')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline as the VAG-Gen subclass
    pipe = VAGGenSD3Pipeline.from_pretrained(
        args.model_id,
        # token=args.huggingface_token,  # Uncomment if needed for private models
    )

    # Select scheduler from CLI
    if args.scheduler != "none":
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # Optionally load LoRA weights if path is a HF checkpoint
    # if "checkpoint" in args.model_path:
    #     pipe.load_lora_weights(args.model_path)
    if "checkpoint-" in args.model_path and (
        os.path.isfile(os.path.join(args.model_path, "pytorch_lora_weights.safetensors"))
        or os.path.isfile(os.path.join(args.model_path, "pytorch_lora_weights.bin"))
    ):
        pipe.load_lora_weights(args.model_path)

    pipe = pipe.to(args.device)

    vag_enabled = not args.disable_vag
    print(f"[VAG-Gen] pid={os.getpid()} device={args.device} "
          f"enabled={vag_enabled}  kappa={args.vag_kappa}  "
          f"base_guidance={args.guidance_scale}  out={args.output_dir}",
          flush=True)

    # Generate images
    for batch in dataloader:
        batch_prompts, batch_prompt_ids = batch
        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts, batch_prompt_ids):
                out_path = f"{args.output_dir}/{_id}.png"
                if os.path.exists(out_path):
                    print(f"Skipping {out_path}", flush=True)
                    continue
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                    vag_kappa=args.vag_kappa,
                    vag_enabled=vag_enabled,
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
                    print(f"Skipping {paths}", flush=True)
                    continue
                result = pipe(
                    prompt=list(prompts),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                    vag_kappa=args.vag_kappa,
                    vag_enabled=vag_enabled,
                )
                for image, _id in zip(result.images, prompt_ids):
                    image.save(f"{args.output_dir}/{_id}.png")
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}.")

    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600.0
    with open(os.path.join(args.output_dir, "runtime_inference.log"), "w") as f:
        f.write(f"Runtime duration: {duration_hours:.4f} hours\n")
        f.write(f"VAG-Gen: enabled={vag_enabled}  kappa={args.vag_kappa}  "
                f"base_guidance={args.guidance_scale}\n")
        f.write(f"Device (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'): "
                f"{args.device}\n")


def _worker_entry(args_dict, gpu_id):
    """Entry point for each spawned process.

    IMPORTANT: we pin the worker to a single GPU by setting
    ``CUDA_VISIBLE_DEVICES`` before any CUDA call is made. With the
    ``spawn`` start method the child is a fresh interpreter; although
    ``import torch`` has already run by the time this function executes,
    torch defers CUDA context creation until the first CUDA operation,
    so setting the env var here is still effective. Inside the worker
    the GPU is always addressed as ``cuda:0``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Rebuild argparse.Namespace from a plain dict to keep the payload
    # trivially picklable across the spawn boundary.
    args = argparse.Namespace(**args_dict)
    args.device = "cuda"  # resolves to cuda:0 inside this restricted view
    try:
        run_experiment(args)
    except Exception as e:
        # Surface a clear message in the parent's stdout; re-raise so
        # the process exits with non-zero status.
        print(f"[VAG-Gen] WORKER FAILED  gpu={gpu_id}  kappa={args.vag_kappa}  "
              f"err={type(e).__name__}: {e}", flush=True)
        raise


def _format_kappa_tag(k):
    """Produce a filesystem-safe tag for a kappa value."""
    # strip trailing zeros: 0.50 -> 0.5, 1.000 -> 1
    s = f"{k:.4f}".rstrip("0").rstrip(".")
    return s.replace("-", "neg").replace(".", "p")  # 0p5, neg0p25, 1


def main():
    args = parse_arguments()

    # ------------------------------------------------------------------
    # Single-run (legacy) path: no --kappas given.
    # ------------------------------------------------------------------
    if args.kappas is None:
        if args.gpus is not None:
            raise ValueError("--gpus is only meaningful when --kappas is provided.")
        run_experiment(args)
        return

    # ------------------------------------------------------------------
    # Parallel sweep: one worker per (kappa, gpu) pair.
    # ------------------------------------------------------------------
    if args.gpus is None:
        raise ValueError("--gpus must be provided when using --kappas.")
    if len(args.kappas) != len(args.gpus):
        raise ValueError(
            f"--kappas has {len(args.kappas)} entries but --gpus has "
            f"{len(args.gpus)}; they must match 1:1."
        )

    # import multiprocessing as mp
    # ctx = mp.get_context("spawn")

    # base_output_dir = os.path.join(args.output_dir, 'coco17_sd35')  # e.g. "outputs"
    # processes = []
    # sweep_start = time.time()

    # for kappa, gpu_id in zip(args.kappas, args.gpus):
    #     # Build a per-worker args payload as a plain dict.
    #     worker_args = vars(args).copy()
    #     worker_args["vag_kappa"] = float(kappa)
    #     worker_args["output_dir"] = f"{base_output_dir}/vagsgen_k{_format_kappa_tag(kappa)}"
    #     # These aren't used by the worker but keep the namespace complete.
    #     worker_args["kappas"] = None
    #     worker_args["gpus"] = None
    #     worker_args["device"] = "cuda"

    #     p = ctx.Process(
    #         target=_worker_entry,
    #         args=(worker_args, int(gpu_id)),
    #         name=f"vaggen-k{kappa}-gpu{gpu_id}",
    #     )
    #     p.start()
    #     print(f"[VAG-Gen] launched pid={p.pid}  kappa={kappa}  gpu={gpu_id}  "
    #           f"out_subdir={worker_args['output_dir']}", flush=True)
    #     processes.append((p, kappa, gpu_id))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    base_output_dir = os.path.join(args.output_dir)
    processes = []
    sweep_start = time.time()

    for kappa, gpu_id in zip(args.kappas, args.gpus):
        worker_args = vars(args).copy()
        worker_args["vag_kappa"] = float(kappa)
        worker_args["output_dir"] = f"{base_output_dir}/vagsgen_k{_format_kappa_tag(kappa)}"
        worker_args["kappas"] = None
        worker_args["gpus"] = None
        worker_args["device"] = "cuda"

        # Set env in parent BEFORE spawn so child inherits it for its
        # module-level imports (where CUDA may otherwise get initialised
        # with all devices visible, permanently).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        p = ctx.Process(
            target=_worker_entry,
            args=(worker_args, int(gpu_id)),
            name=f"vaggen-k{kappa}-gpu{gpu_id}",
        )
        p.start()
        print(f"[VAG-Gen] launched pid={p.pid}  kappa={kappa}  gpu={gpu_id}  "
              f"out_subdir={worker_args['output_dir']}", flush=True)
        processes.append((p, kappa, gpu_id))

    # Restore parent's env
    if parent_cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = parent_cvd

    # Wait for all workers; collect any failures but don't abort the others.
    failures = []
    for p, kappa, gpu_id in processes:
        p.join()
        if p.exitcode != 0:
            failures.append((kappa, gpu_id, p.exitcode))

    sweep_hours = (time.time() - sweep_start) / 3600.0
    print(f"[VAG-Gen] sweep finished in {sweep_hours:.4f} h  "
          f"failures={len(failures)}/{len(processes)}", flush=True)
    if failures:
        for kappa, gpu_id, code in failures:
            print(f"  - kappa={kappa}  gpu={gpu_id}  exitcode={code}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()