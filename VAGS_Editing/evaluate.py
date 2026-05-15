"""
evaluate.py  –  PnP-Inv evaluation script adapted for the shared benchmark.

Keeps the exact same logic as the original PnP-Inv evaluate.py, with:
  1. Filename convention updated: edited images are stored as
         <out_dir>/<method>/<base_name>_<code>.png
     instead of the PnP-Inv annotation_images sub-folder layout.
  2. Three extra metrics added:
       • clip_i   – CLIP image-image similarity (CLIP-I)
       • fid      – Fréchet Inception Distance (FID, via clean-fid)
                    FID is a dataset-level metric: one value per method, written
                    as an extra summary row at the bottom of the same CSV.
     All original metrics are unchanged.

Usage example (PIE-Bench):
    python evaluate.py \
        --annotation_mapping_file Data/PIE-Bench_v1/mapping_file.json \
        --src_image_folder        Data/PIE-Bench_v1/annotation_images \
        --edited_images_dir       outputs/benchmark_pie_20250309_120000 \
        --tgt_methods             flowedit_sd35 splitflow_sd35 \
        --result_path             outputs/benchmark_pie_20250309_120000/eval.csv \
        --device                  cuda
    → produces: outputs/benchmark_pie_20250309_120000/results_flowedit_sd35.csv
                outputs/benchmark_pie_20250309_120000/results_splitflow_sd35.csv

Usage example (DIV2K / flowedit.yaml, no masks):
    python evaluate.py \
        --yaml_file          Data/flowedit.yaml \
        --src_image_folder   Data \
        --edited_images_dir  outputs/benchmark_20260226_160112 \
        --tgt_methods        flowedit_sd35 splitflow_sd35 \
        --result_path        outputs/benchmark_20260226_160112/eval.csv
    → produces: outputs/benchmark_20260226_160112/results_flowedit_sd35.csv
                outputs/benchmark_20260226_160112/results_splitflow_sd35.csv
"""

import json
import argparse
import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import csv
import yaml

# ---------------------------------------------------------------------------
# Locate MetricsCalculator from PnP-Inv (or fall back to uploaded copy)
# ---------------------------------------------------------------------------
_PNPINV_ROOT = Path(__file__).parent / "methods" / "PnPInversion"
if _PNPINV_ROOT.exists() and str(_PNPINV_ROOT) not in sys.path:
    sys.path.insert(0, str(_PNPINV_ROOT))
from evaluation.matrics_calculator import MetricsCalculator


# ---------------------------------------------------------------------------
# Patch MetricsCalculator with calculate_clip_image_similarity
# (original class uses CLIPScore text-image, not image-image)
# ---------------------------------------------------------------------------
def _calculate_clip_image_similarity(self, img1, img2):
    """CLIP-I: cosine similarity of CLIP image features, scale 0-100."""
    model     = self.clip_metric_calculator.model
    processor = self.clip_metric_calculator.processor
    inputs = processor(images=[img1, img2], return_tensors="pt",
                       padding=True).to(self.device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1]).item() * 100

MetricsCalculator.calculate_clip_image_similarity = _calculate_clip_image_similarity


# ---------------------------------------------------------------------------
# Helpers identical to original PnP-Inv evaluate.py
# ---------------------------------------------------------------------------

def mask_decode(encoded_mask, image_shape=[512, 512]):
    """Decode PIE-Bench RLE mask → (H, W, 3) float32 array with values in {0,1}."""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # Avoid annotation errors in boundary (official PnP-Inv convention)
    mask_array[0,  :] = 1
    mask_array[-1, :] = 1
    mask_array[:,  0] = 1
    mask_array[:, -1] = 1

    return mask_array[:, :, np.newaxis].repeat(3, axis=2)


def calculate_metric(metrics_calculator, metric,
                     src_image, tgt_image,
                     src_mask, tgt_mask,
                     src_prompt, tgt_prompt):
    """Compute a single metric.  Identical to the original PnP-Inv function."""
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    # ---- NEW metric: CLIP-I (image-image cosine similarity) ----
    if metric == "clip_i":
        return _calculate_clip_i(src_image, tgt_image, metrics_calculator)

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# NEW: CLIP-I helper (uses the CLIP model already loaded in MetricsCalculator)
# ---------------------------------------------------------------------------

def _calculate_clip_i(src_image: Image.Image, tgt_image: Image.Image,
                      metrics_calculator: MetricsCalculator) -> float:
    """
    CLIP image-image cosine similarity between src and tgt.
    Re-uses the CLIPScore model already initialised inside MetricsCalculator.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    # Lazy-load a shared CLIP model (cached on first call)
    if not hasattr(_calculate_clip_i, "_model"):
        device = metrics_calculator.device
        _calculate_clip_i._model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(device).eval()
        _calculate_clip_i._processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        _calculate_clip_i._device = device

    model     = _calculate_clip_i._model
    processor = _calculate_clip_i._processor
    device    = _calculate_clip_i._device

    with torch.no_grad():
        inp   = processor(images=[src_image, tgt_image],
                          return_tensors="pt").to(device)
        feats = model.get_image_features(**inp)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return (feats[0] @ feats[1]).item()


# ---------------------------------------------------------------------------
# NEW: FID helper using clean-fid
# ---------------------------------------------------------------------------

def compute_fid_for_method(method_folder: Path, src_images: list,
                           device: str = "cuda") -> float:
    """
    Compute FID between edited images in method_folder and source images.
    Uses clean-fid (mode='clean', images resized to 256×256).
    Returns float or "nan" if clean-fid is unavailable or < 2 images found.
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[FID] clean-fid not installed — skipping FID.")
        return "nan"

    with tempfile.TemporaryDirectory() as ref_tmp, \
         tempfile.TemporaryDirectory() as edit_tmp:

        ref_p  = Path(ref_tmp)
        edit_p = Path(edit_tmp)

        # --- prepare reference (source) images ---
        for p in src_images:
            p = Path(p)
            if not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
                img.save(ref_p / p.name)
            except Exception as e:
                print(f"[FID] could not load source image {p}: {e}")

        if len(list(ref_p.glob("*"))) < 2:
            print("[FID] not enough source images — skipping FID.")
            return "nan"

        # --- prepare edited images ---
        for f in method_folder.glob("*.png"):
            try:
                img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
                img.save(edit_p / f.name)
            except Exception as e:
                print(f"[FID] could not load edited image {f}: {e}")

        if len(list(edit_p.glob("*"))) < 2:
            print(f"[FID] not enough edited images for {method_folder.name} — skipping.")
            return "nan"

        try:
            score = cleanfid.compute_fid(str(edit_p), str(ref_p),
                                         mode="clean", verbose=False)
            return score
        except Exception as e:
            print(f"[FID] compute_fid failed for {method_folder.name}: {e}")
            return "nan"


# ---------------------------------------------------------------------------
# Filename resolution helper
# ---------------------------------------------------------------------------

def _find_edited_image(edited_dir: Path, method: str,
                       base_name: str, code: str) -> str | None:
    """
    Look for <edited_dir>/<method>/<base_name>_<code>.<ext>.
    Tries .png, .jpg, .jpeg in order.
    """
    folder = edited_dir / method
    stem   = f"{base_name}_{code}"
    for ext in (".png", ".jpg", ".jpeg"):
        p = folder / (stem + ext)
        if p.exists():
            return str(p)
    return None


def _find_source_image(src_folder: str, rel_path: str) -> str | None:
    """Find source image, trying multiple extensions."""
    p = Path(src_folder) / rel_path
    if p.exists():
        return str(p)
    for ext in (".png", ".jpg", ".jpeg"):
        c = p.with_suffix(ext)
        if c.exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate image-editing methods using PnP-Inv metrics + CLIP-I + FID"
    )
    parser.add_argument("--annotation_mapping_file", type=str,
                        default=None,
                        help="PIE-Bench mapping_file.json (mutually exclusive with --yaml_file)")
    parser.add_argument("--yaml_file", type=str,
                        default=None,
                        help="flowedit.yaml / DIV2K pairs file (mutually exclusive with "
                             "--annotation_mapping_file). Implies --no_pie_mask.")
    parser.add_argument("--src_image_folder", type=str,
                        default=None,
                        help="Root folder containing source images "
                             "(default: Data/PIE-Bench_v1/annotation_images for PIE-Bench, "
                             " Data/ for yaml mode)")
    parser.add_argument("--edited_images_dir", type=str, required=True,
                        help="Root output directory produced by benchmark_all_methods.py "
                             "(contains one sub-folder per method)")
    parser.add_argument("--tgt_methods", nargs="+", type=str,
                        default=["flowedit_sd35", "splitflow_sd35"],
                        help="Method names (= sub-folder names inside --edited_images_dir)")
    parser.add_argument("--metrics", nargs="+", type=str,
                        default=[
                            "structure_distance",
                            "psnr_unedit_part",
                            "lpips_unedit_part",
                            "mse_unedit_part",
                            "ssim_unedit_part",
                            "clip_similarity_source_image",
                            "clip_similarity_target_image",
                            "clip_similarity_target_image_edit_part",
                            "clip_i",   # NEW
                        ],
                        help="Metrics to compute per image pair")
    parser.add_argument("--result_path", type=str,
                        default="evaluation_result.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--edit_category_list", nargs="+", type=str,
                        default=["0","1","2","3","4","5","6","7","8","9"],
                        help="PIE-Bench editing category IDs to include")
    parser.add_argument("--no_pie_mask", action="store_true",
                        help="Set if not using PIE-Bench masks "
                             "(mask will be all-ones for every pair)")
    args = parser.parse_args()

    # ---- Validate / set defaults ----
    if args.yaml_file is None and args.annotation_mapping_file is None:
        args.annotation_mapping_file = "Data/PIE-Bench_v1/mapping_file.json"
    if args.yaml_file is not None and args.annotation_mapping_file is not None:
        parser.error("Specify either --yaml_file or --annotation_mapping_file, not both.")

    use_yaml = args.yaml_file is not None
    if use_yaml:
        args.no_pie_mask = True   # yaml mode → no masks

    if args.src_image_folder is None:
        args.src_image_folder = "Data" if use_yaml else "Data/PIE-Bench_v1/annotation_images"

    metrics          = args.metrics
    # In yaml / no-mask mode replace _unedit metrics with full-image equivalents
    if args.no_pie_mask:
        metrics = [m.replace("_unedit_part", "").replace("_edit_part", "")
                   if m.endswith("_unedit_part") or m.endswith("_edit_part") else m
                   for m in metrics]
        # deduplicate while preserving order
        seen = set(); metrics = [m for m in metrics if not (m in seen or seen.add(m))]

    src_image_folder = args.src_image_folder
    tgt_methods      = args.tgt_methods
    edit_category_list = args.edit_category_list
    edited_dir       = Path(args.edited_images_dir)

    # One output file per method, named results_{method}.csv
    # --result_path defines the output directory (its parent folder is used).
    #   --result_path outputs/eval/result.csv
    #   → outputs/eval/results_flowedit_sd35.csv
    #   → outputs/eval/results_splitflow_sd35.csv
    result_base  = Path(args.result_path)
    result_dir   = result_base.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    result_paths = {
        m: result_dir / f"results_{m}.csv"
        for m in tgt_methods
    }

    metrics_calculator = MetricsCalculator(args.device)

    # ---- Write CSV headers (one per method, simple column names without method prefix) ----
    # "fid" is added as a dedicated column; it's nan for every image row and
    # filled with the dataset-level FID score in the final FID_SUMMARY row.
    header = ["file_id"] + metrics + ["fid"]
    for method, rp in result_paths.items():
        with open(rp, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # ---- Build the list of pairs to evaluate ----
    # Each pair is a dict:
    #   file_id, base_image_path, original_prompt, editing_prompt,
    #   base_name, code, mask (np array or None → all-ones)
    pairs_to_eval = []

    if use_yaml:
        # ---- YAML / DIV2K mode ----
        with open(args.yaml_file) as f:
            yaml_data = yaml.safe_load(f)
        for entry in yaml_data:
            init_img      = entry["init_img"]                 # e.g. "flowedit_data/bear.png"
            src_prompt    = str(entry.get("source_prompt", "")).strip()
            tgt_prompts   = entry.get("target_prompts", [])
            tgt_codes     = entry.get("target_codes", [])
            stem          = Path(init_img).stem               # "bear"
            for tgt_prompt, code in zip(tgt_prompts, tgt_codes):
                pairs_to_eval.append(dict(
                    file_id         = f"{stem}_{code}",
                    base_image_path = init_img,
                    original_prompt = str(src_prompt).strip(),
                    editing_prompt  = str(tgt_prompt).strip(),
                    base_name       = stem,
                    code            = str(code),
                    mask            = None,   # → all-ones below
                ))
    else:
        # ---- PIE-Bench JSON mode ----
        with open(args.annotation_mapping_file, "r") as f:
            annotation_file = json.load(f)
        for key, item in annotation_file.items():
            if item["editing_type_id"] not in edit_category_list:
                continue
            base_image_path = item["image_path"]
            parts     = Path(base_image_path)
            base_name = f"{parts.parent.name}__{parts.stem}"
            pairs_to_eval.append(dict(
                file_id         = key,
                base_image_path = base_image_path,
                original_prompt = item["original_prompt"].replace("[", "").replace("]", ""),
                editing_prompt  = item["editing_prompt"].replace("[", "").replace("]", ""),
                base_name       = base_name,
                code            = key,
                mask            = None if args.no_pie_mask else mask_decode(item["mask"]),
            ))

    # Collect unique source image paths for FID
    source_image_paths: list = []

    # ---- Iterate over every pair ----
    for pair in pairs_to_eval:
        file_id         = pair["file_id"]
        base_image_path = pair["base_image_path"]
        original_prompt = pair["original_prompt"]
        editing_prompt  = pair["editing_prompt"]
        base_name       = pair["base_name"]
        code            = pair["code"]
        mask = pair["mask"] if pair["mask"] is not None \
               else np.ones((512, 512, 3), dtype=np.float32)

        print(f"Evaluating {file_id} …")

        # Load source image
        src_image_path = _find_source_image(src_image_folder, base_image_path)
        if src_image_path is None:
            print(f"  [SKIP] source image not found: {base_image_path}")
            continue
        src_image = Image.open(src_image_path).convert("RGB")

        if src_image_path not in source_image_paths:
            source_image_paths.append(src_image_path)

        for method in tgt_methods:
            tgt_image_path = _find_edited_image(edited_dir, method, base_name, code)
            print(f"  {method}  →  {tgt_image_path}")

            row = [file_id]

            if tgt_image_path is None:
                print(f"  [SKIP] edited image not found for {method} / {base_name}_{code}")
                row.extend(["nan"] * len(metrics))
            else:
                tgt_image = Image.open(tgt_image_path).convert("RGB")

                # Handle multi-panel outputs (original PnP-Inv convention):
                # if width != height, crop the last square panel (the edited result)
                if tgt_image.size[0] != tgt_image.size[1]:
                    side = min(tgt_image.size)
                    tgt_image = tgt_image.crop((
                        tgt_image.size[0] - side, tgt_image.size[1] - side,
                        tgt_image.size[0],         tgt_image.size[1],
                    ))

                # Align source image size to target (needed for non-PIE-Bench datasets)
                if src_image.size != tgt_image.size:
                    src_image_eval = src_image.resize(tgt_image.size, Image.LANCZOS)
                else:
                    src_image_eval = src_image
                # Also align mask to target size
                tgt_h, tgt_w = tgt_image.size[1], tgt_image.size[0]
                mask_used = mask
                if mask_used.shape[:2] != (tgt_h, tgt_w):
                    from PIL import Image as _PILImage
                    mask_pil = _PILImage.fromarray((mask_used[:,:,0] * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((tgt_w, tgt_h), _PILImage.NEAREST)
                    mask_used = np.array(mask_pil)[:, :, np.newaxis].repeat(3, axis=2) / 255.0

                for metric in metrics:
                    print(f"    {metric}")
                    value = calculate_metric(
                        metrics_calculator, metric,
                        src_image_eval, tgt_image,
                        mask_used, mask_used,
                        original_prompt, editing_prompt,
                    )
                    row.append(value)

            row.append("nan")  # fid column — filled in FID_SUMMARY row
            with open(result_paths[method], "a+", newline="") as f:
                csv.writer(f).writerow(row)

    print(f"\nPer-image results saved to:")
    for method, rp in result_paths.items():
        print(f"  {rp}")

    # ---- FID (one score per method, appended as a FID_SUMMARY row in each method CSV) ----
    fid_results = {}
    for method in tgt_methods:
        method_folder = edited_dir / method
        if not method_folder.exists():
            print(f"[FID] folder not found for method {method}, skipping.")
            fid_results[method] = "nan"
            continue
        print(f"Computing FID for {method} …")
        fid_results[method] = compute_fid_for_method(
            method_folder, source_image_paths, device=args.device)

    for method in tgt_methods:
        fid_value = fid_results.get(method, "nan")
        # FID_SUMMARY row: nan for all per-image metrics, fid value in the fid column
        fid_row = ["FID_SUMMARY"] + ["nan"] * len(metrics) + [fid_value]
        with open(result_paths[method], "a+", newline="") as f:
            csv.writer(f).writerow(fid_row)

    print("\n=== FID Summary ===")
    for method, score in fid_results.items():
        print(f"  {method:<40} FID = {score}")