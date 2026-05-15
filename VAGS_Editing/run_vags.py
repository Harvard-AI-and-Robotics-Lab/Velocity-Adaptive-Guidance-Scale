"""
run_vags.py — Simple entry point for VAGS image editing (cosine variant).

Three usage modes:

  1. Single image pair
     python run_vags.py \
         --src_image path/to/source.jpg \
         --src_prompt "a cat on a red chair" \
         --tgt_prompt "a dog on a red chair" \
         --output edited.png

  2. PIE-Bench dataset
     python run_vags.py \
         --dataset pie_bench \
         --outdir outputs/pie_bench \
         --gpu 0

  3. Custom YAML file (same format as FlowEdit)
     python run_vags.py \
         --dataset yaml \
         --yaml_file Data/flowedit.yaml \
         --images_root Data/ \
         --outdir outputs/custom \
         --gpu 0
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent

# ── Hyperparameters (paper defaults) ─────────────────────────────────────────
SRC_GUIDANCE  = 3.5
TAR_GUIDANCE  = 13.5
KAPPA         = 0.9
T_STEPS       = 50
N_MAX         = 33      # active editing steps
N_MIN         = 0
N_AVG         = 1

PIE_MAPPING   = str(ROOT / "Data/PIE-Bench_v1/mapping_file.json")
PIE_IMAGES    = str(ROOT / "Data/PIE-Bench_v1/annotation_images")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_brackets(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]", r"\1", text).strip()


def _crop16(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w - w % 16, h - h % 16))


def _find(path: str) -> str:
    for ext in ("", ".png", ".jpg", ".jpeg"):
        p = Path(path + ext) if ext else Path(path)
        if p.exists():
            return str(p)
    return path


def load_pairs_pie(mapping_file: str, images_root: str):
    with open(mapping_file) as f:
        data = json.load(f)
    pairs = []
    for key, entry in data.items():
        rel  = entry["image_path"]
        img  = _find(str(Path(images_root) / rel))
        base = f"{Path(rel).parent.name}__{Path(rel).stem}"
        pairs.append({
            "image_path":    img,
            "base_name":     base,
            "source_prompt": _strip_brackets(str(entry["original_prompt"])),
            "target_prompt": _strip_brackets(str(entry["editing_prompt"])),
            "code":          key,
        })
    return pairs


def load_pairs_yaml(yaml_file: str, images_root: str):
    import yaml
    with open(yaml_file) as f:
        entries = list(yaml.safe_load_all(f))[0]
    pairs = []
    for entry in entries:
        rel  = entry["init_img"]
        img  = _find(str(Path(images_root) / rel))
        base = Path(rel).stem
        src  = str(entry["source_prompt"]).strip()
        for tp, code in zip(entry["target_prompts"], entry["target_codes"]):
            pairs.append({
                "image_path":    img,
                "base_name":     base,
                "source_prompt": src,
                "target_prompt": str(tp).strip(),
                "code":          str(code).strip(),
            })
    return pairs


# ── VAGS editing ──────────────────────────────────────────────────────────────

def edit_single(pipe, scheduler, src_img: Image.Image,
                src_prompt: str, tgt_prompt: str,
                device: str) -> Image.Image:
    """Run VAGS+Cosine on a single image and return the edited image."""
    from FlowEdit_utils import FlowEditSD3_ConflictAware_Cosine

    proc  = pipe.image_processor.preprocess(_crop16(src_img)).to(device, torch.float16)
    with torch.autocast("cuda"), torch.inference_mode():
        z = pipe.vae.encode(proc).latent_dist.mode()
    x_src = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    x_tar = FlowEditSD3_ConflictAware_Cosine(
        pipe, scheduler, x_src,
        src_prompt, tgt_prompt,
        negative_prompt="",
        T_steps=T_STEPS, n_avg=N_AVG,
        src_guidance_scale=SRC_GUIDANCE,
        tar_guidance_scale=TAR_GUIDANCE,
        kappa_tar=KAPPA,
        n_min=N_MIN, n_max=N_MAX,
    )

    x_out = x_tar / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        img_out = pipe.vae.decode(x_out, return_dict=False)[0]
    return pipe.image_processor.postprocess(img_out)[0]


def run_dataset(pairs, out_dir: Path, device: str):
    """Edit a list of pairs and save results to out_dir."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    ).to(device)
    scheduler = pipe.scheduler

    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in tqdm(pairs, desc="VAGS editing"):
        out_path = out_dir / f"{pair['base_name']}_{pair['code']}.png"
        if out_path.exists():
            continue
        try:
            src_img = Image.open(pair["image_path"]).convert("RGB")
            edited  = edit_single(pipe, scheduler, src_img,
                                  pair["source_prompt"], pair["target_prompt"],
                                  device)
            edited.save(out_path)
        except Exception as e:
            print(f"[skip] {pair['base_name']} {pair['code']}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Done. Results saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VAGS image editing (cosine variant, SD 3.5 Large)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--src_image", type=str,
                      help="Path to a single source image")
    mode.add_argument("--dataset", choices=["pie_bench", "yaml"],
                      help="Run on a dataset: 'pie_bench' or 'yaml'")

    # Single-image mode
    parser.add_argument("--src_prompt", type=str, default="",
                        help="Source prompt (single-image mode)")
    parser.add_argument("--tgt_prompt", type=str, default="",
                        help="Target prompt (single-image mode)")
    parser.add_argument("--output", type=str, default="edited.png",
                        help="Output path (single-image mode)")

    # Dataset mode
    parser.add_argument("--yaml_file",   type=str, default="Data/flowedit.yaml")
    parser.add_argument("--images_root", type=str, default="Data/")
    parser.add_argument("--outdir",      type=str, default="outputs/vags")

    # Shared
    parser.add_argument("--gpu",     type=int,   default=0)
    parser.add_argument("--kappa",   type=float, default=KAPPA,
                        help=f"Modulation strength κ (default: {KAPPA})")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    device = "cuda:0"

    global KAPPA
    KAPPA = args.kappa

    if args.src_image:
        # ── Single-image mode ────────────────────────────────────────────────
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16,
        ).to(device)

        src = Image.open(args.src_image).convert("RGB")
        edited = edit_single(pipe, pipe.scheduler, src,
                             args.src_prompt, args.tgt_prompt, device)
        edited.save(args.output)
        print(f"Saved: {args.output}")

    else:
        # ── Dataset mode ─────────────────────────────────────────────────────
        if args.dataset == "pie_bench":
            pairs = load_pairs_pie(PIE_MAPPING, PIE_IMAGES)
        else:
            pairs = load_pairs_yaml(args.yaml_file, args.images_root)

        print(f"Loaded {len(pairs)} pairs.")
        run_dataset(pairs, Path(args.outdir), device)


if __name__ == "__main__":
    main()
