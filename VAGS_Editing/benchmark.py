import os, sys, csv, json, re, yaml, traceback, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import convolve

ROOT         = Path(__file__).parent
METHODS_ROOT = ROOT / "methods"
LOG_FMT      = "[%(asctime)s %(levelname)s %(name)s] %(message)s"

METRICS_PIE = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
    "clip_i",
]
METRICS_YAML = [
    "structure_distance",
    "psnr",
    "lpips",
    "mse",
    "ssim",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_i",
]
CSV_COLS      = ["file_id"] + METRICS_PIE  + ["fid"]
CSV_COLS_PIE  = CSV_COLS
CSV_COLS_YAML = ["file_id"] + METRICS_YAML + ["fid"]
METRICS = METRICS_PIE

def check_and_install():
    """Install missing optional dependencies quietly before importing methods."""
    import importlib, subprocess
    _REQUIRED = {
        "jaxtyping":        "jaxtyping",
        "einops":           "einops",
        "imwatermark":      "invisible-watermark",
        "omegaconf":        "omegaconf",
        "pytorch_lightning": "pytorch-lightning",
        "cleanfid":         "clean-fid",
    }
    for mod, pkg in _REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[setup] Installing missing package: {pkg} …")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg,
                 "--break-system-packages", "--quiet"],
                check=False,
            )

check_and_install()

class MetricsEvaluator:
    """
    Unified evaluator matching evaluate.py metric names and logic exactly.

    PIE mode (mask_encoded provided): computes _unedit_part variants and
      clip_similarity_target_image_edit_part.
    YAML mode (no mask): computes full-image psnr/lpips/mse/ssim.
    """

    def __init__(self, device: str, pie_bench: bool = False):
        self.device    = device
        self.pie_bench = pie_bench
        self._ready    = False

    def _load(self):
        if self._ready:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._clip  = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device).eval()
        self._cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _pnp = str(Path(__file__).parent / "methods" / "PnPInversion")
        if _pnp not in sys.path:
            sys.path.insert(0, _pnp)
        from evaluation.matrics_calculator import MetricsCalculator
        self._mc = MetricsCalculator(self.device)
        self._ready = True

    @staticmethod
    def _mask_decode(encoded_mask, image_shape=(512, 512)):
        """Decode PIE-Bench RLE mask → (H, W, 3) float32 array matching evaluate.py."""
        length = image_shape[0] * image_shape[1]
        mask_array = np.zeros((length,))
        for i in range(0, len(encoded_mask), 2):
            splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i] + j] = 1
        mask_array = mask_array.reshape(image_shape[0], image_shape[1])
        mask_array[0,  :] = 1
        mask_array[-1, :] = 1
        mask_array[:,  0] = 1
        mask_array[:, -1] = 1
        return mask_array[:, :, np.newaxis].repeat(3, axis=2)

    def _clip_similarity(self, img: Image.Image, prompt: str,
                         mask: np.ndarray = None) -> float:
        """CLIP image-text similarity, optionally masked to a region."""
        self._load()
        with torch.no_grad():
            if mask is not None and mask.sum() > 0:
                ys, xs = np.where(mask[:, :, 0] > 0)
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())
                img_eval = img.convert("RGB").resize((512, 512)).crop(
                    (x1, y1, x2 + 1, y2 + 1))
            else:
                img_eval = img
            inp = self._cproc(text=[prompt], images=img_eval,
                              return_tensors="pt", padding=True).to(self.device)
            return self._clip(**inp).logits_per_image.item() / 100.0

    def _clip_i(self, src: Image.Image, tgt: Image.Image) -> float:
        """CLIP image-image cosine similarity."""
        self._load()
        with torch.no_grad():
            inp   = self._cproc(images=[src, tgt],
                                return_tensors="pt").to(self.device)
            feats = self._clip.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return (feats[0] @ feats[1]).item()

    def all_metrics(self, src: Image.Image, edited: Image.Image,
                    target_prompt: str, mask_encoded=None,
                    source_prompt: str = None) -> Dict[str, float]:
        if edited.size != src.size:
            edited = edited.resize(src.size, Image.LANCZOS)
        self._load()
        src512 = src.resize((512, 512))
        tgt512 = edited.resize((512, 512))

        _src_prompt = source_prompt if source_prompt is not None else target_prompt

        result = {
            "structure_distance":           float(self._mc.calculate_structure_distance(
                                                src512, tgt512, None, None)),
            "clip_similarity_source_image": self._clip_similarity(src512, _src_prompt),
            "clip_similarity_target_image": self._clip_similarity(tgt512, target_prompt),
            "clip_i":                       self._clip_i(src512, tgt512),
        }

        if mask_encoded is not None:
            mask = self._mask_decode(mask_encoded)
            inv_mask = 1 - mask
            if inv_mask.sum() == 0:
                result["psnr_unedit_part"]  = "nan"
                result["lpips_unedit_part"] = "nan"
                result["mse_unedit_part"]   = "nan"
                result["ssim_unedit_part"]  = "nan"
            else:
                result["psnr_unedit_part"]  = float(self._mc.calculate_psnr(
                    src512, tgt512, inv_mask, inv_mask))
                result["lpips_unedit_part"] = float(self._mc.calculate_lpips(
                    src512, tgt512, inv_mask, inv_mask))
                result["mse_unedit_part"]   = float(self._mc.calculate_mse(
                    src512, tgt512, inv_mask, inv_mask))
                result["ssim_unedit_part"]  = float(self._mc.calculate_ssim(
                    src512, tgt512, inv_mask, inv_mask))
            if mask.sum() == 0:
                result["clip_similarity_target_image_edit_part"] = "nan"
            else:
                result["clip_similarity_target_image_edit_part"] = self._clip_similarity(
                    tgt512, target_prompt, mask)
        else:
            result["psnr"]  = float(self._mc.calculate_psnr(src512, tgt512, None, None))
            result["lpips"] = float(self._mc.calculate_lpips(src512, tgt512, None, None))
            result["mse"]   = float(self._mc.calculate_mse(src512, tgt512, None, None))
            result["ssim"]  = float(self._mc.calculate_ssim(src512, tgt512, None, None))

        return result

def _strip_pie_brackets(text: str) -> str:
    """Remove PIE-Bench annotation brackets: '[rusty]' → 'rusty'."""
    import re as _re
    return _re.sub(r"\[([^\]]+)\]", r"\1", text).strip()

def load_pairs_pie(mapping_file: str, images_root: str,
                   max_pairs: Optional[int] = None) -> List[Dict]:
    """Return list of editing pairs from a PIE-Bench mapping_file.json."""
    with open(mapping_file) as f:
        data = json.load(f)
    pairs = []
    for key, entry in data.items():
        rel      = entry["image_path"]
        img_path = str(Path(images_root) / rel)
        resolved = _find_image(img_path)
        if resolved is None:
            continue
        parts     = Path(rel)
        base_name = f"{parts.parent.name}__{parts.stem}"
        pairs.append({
            "image_path":    resolved,
            "base_name":     base_name,
            "source_prompt": _strip_pie_brackets(str(entry["original_prompt"])),
            "target_prompt": _strip_pie_brackets(str(entry["editing_prompt"])),
            "code":          key,
            "mask_encoded":  entry.get("mask"),
        })
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs

def load_pairs(yaml_path: str, images_root: str,
               max_pairs: Optional[int] = None) -> List[Dict]:
    """Return list of editing pairs from the FlowEdit YAML."""
    with open(yaml_path) as f:
        entries = list(yaml.safe_load_all(f))[0]
    pairs = []
    for entry in entries:
        img_name  = Path(entry["init_img"]).name
        img_path  = str(Path(images_root) / img_name)
        resolved  = _find_image(img_path)
        if resolved:
            img_path = resolved
        base_name = Path(img_name).stem
        src_prompt = str(entry["source_prompt"]).strip()
        for tp, code in zip(entry["target_prompts"], entry["target_codes"]):
            pairs.append({
                "image_path":    img_path,
                "base_name":     base_name,
                "source_prompt": src_prompt,
                "target_prompt": str(tp).strip(),
                "code":          str(code).strip(),
            })
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs

def write_csv(out_dir: Path, method: str, rows: List[Dict],
              source_images: Optional[List[str]] = None):
    """Write per-method CSV matching evaluate.py format exactly.

    Format: header ["file_id"] + metrics + ["fid"]
    Image rows: metrics computed, fid = "nan"
    FID_SUMMARY row at end: metrics = "nan", fid = actual FID value
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_{method}.csv"

    if rows and "psnr_unedit_part" in rows[0]:
        cols = CSV_COLS
    else:
        cols = CSV_COLS_YAML

    metrics_cols = [c for c in cols if c not in ("file_id", "fid")]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="nan")
        w.writeheader()
        for row in rows:
            row_out = {**row, "fid": "nan"}
            w.writerow(row_out)

    fid_value = "nan"
    if source_images:
        try:
            import tempfile
            from cleanfid import fid as cleanfid
            from PIL import Image as _PIL
            method_folder = out_dir / method
            with tempfile.TemporaryDirectory() as ref_tmp, \
                 tempfile.TemporaryDirectory() as edit_tmp:
                ref_p  = Path(ref_tmp)
                edit_p = Path(edit_tmp)
                for p in source_images:
                    p = Path(p)
                    if p.exists():
                        try:
                            img = _PIL.open(p).convert("RGB").resize((256, 256), _PIL.LANCZOS)
                            img.save(ref_p / p.name)
                        except Exception:
                            pass
                for f in method_folder.glob("*.png"):
                    try:
                        img = _PIL.open(f).convert("RGB").resize((256, 256), _PIL.LANCZOS)
                        img.save(edit_p / f.name)
                    except Exception:
                        pass
                if len(list(ref_p.glob("*"))) >= 2 and len(list(edit_p.glob("*"))) >= 2:
                    fid_value = cleanfid.compute_fid(str(edit_p), str(ref_p),
                                                     mode="clean", verbose=False)
        except Exception as e:
            print(f"[FID] {method}: {e}")

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        fid_row = ["FID_SUMMARY"] + ["nan"] * len(metrics_cols) + [fid_value]
        w.writerow(fid_row)

    return path

def save_image(img: Image.Image, out_dir: Path, method: str,
               base: str, code: str) -> Path:
    folder = out_dir / method
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{base}_{code}.png"
    img.save(p)
    return p

def _make_row(pair: Dict, method: str, metrics: Dict, file_path: str) -> Dict:
    return {
        "file_id": f"{pair['base_name']}_{pair['code']}",
        **metrics,
    }

def _load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _crop16(img: Image.Image) -> Image.Image:
    """Crop so both dimensions are divisible by 16."""
    w, h = img.size
    return img.crop((0, 0, w - w % 16, h - h % 16))

def _setup_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FMT))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log

def _find_image(path: str) -> Optional[str]:
    """Return an existing path for the image, trying .png / .jpg / .jpeg in order."""
    p = Path(path)
    if p.exists():
        return str(p)
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    return None

def _safe_load_pil(path: str, log=None) -> Optional[Image.Image]:
    """Load an image as PIL RGB, trying multiple extensions. Returns None and warns if not found."""
    resolved = _find_image(path)
    if resolved is None:
        msg = f"Image not found (skipping): {path}"
        if log:
            log.warning(msg)
        else:
            print(f"[WARNING] {msg}")
        return None
    return Image.open(resolved).convert("RGB")

def _patch_flux_rope():
    """
    Monkey-patch apply_rotary_emb to tolerate RoPE dimension mismatches in diffusers > 0.31.0.
    On ≤ 0.31.0 this is a no-op. On newer versions the patch retries with trimmed freqs_cis
    if the original call raises a shape error.
    """
    import functools
    try:
        import diffusers
        from packaging.version import Version
        if Version(diffusers.__version__) <= Version("0.31.0"):
            return
    except Exception:
        pass

    try:
        import diffusers.models.embeddings as _emb
        _orig = getattr(_emb, "apply_rotary_emb", None)
        if _orig is None:
            return

        @functools.wraps(_orig)
        def _safe_apply_rotary_emb(x, freqs_cis, **kwargs):
            try:
                return _orig(x, freqs_cis, **kwargs)
            except (RuntimeError, ValueError):
                seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[0]
                if isinstance(freqs_cis, tuple):
                    freqs_cis = tuple(
                        f[:seq_len] if hasattr(f, "shape") and f.shape[0] > seq_len else f
                        for f in freqs_cis
                    )
                elif hasattr(freqs_cis, "shape") and freqs_cis.shape[0] > seq_len:
                    freqs_cis = freqs_cis[:seq_len]
                return _orig(x, freqs_cis, **kwargs)

        _emb.apply_rotary_emb = _safe_apply_rotary_emb

        try:
            import diffusers.models.attention_processor as _ap
            if hasattr(_ap, "apply_rotary_emb"):
                _ap.apply_rotary_emb = _safe_apply_rotary_emb
        except Exception:
            pass
    except Exception:
        pass

def run_flowedit_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("flowedit_sd35")
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "flowedit_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "flowedit_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "flowedit_sd35", rows,
              source_images=[p["image_path"] for p in pairs])
    log.info("FlowEdit (SD 3.5) done.")

def run_rf_inversion(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="rf_inversion", strategy="reflow",
        num_steps=25, guidance=2.0, inject=2,
        start_layer=0, end_layer=37,
    )

def run_rf_solver(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="rf_solver", strategy="rf_solver",
        num_steps=25, guidance=2.0, inject=2,
        start_layer=0, end_layer=37,
    )

def run_fireflow(pairs, gpu_id, out_dir):
    _run_flux_fireflow_method(
        pairs, gpu_id, out_dir,
        method_name="fireflow", strategy="fireflow",
        num_steps=8, guidance=2.0, inject=1,
        start_layer=0, end_layer=37,
    )

def run_new_ddim_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DDIM+P2P and DDIM+PnP on CompVis/stable-diffusion-v1-4."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("ddim_sd14")
    log.info(f"Starting DDIM+P2P + DDIM+PnP (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    MODEL_KEY = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["ddim+p2p"], device, num_ddim_steps=50)
    rows_p2p: List[Dict] = []
    for pair in tqdm(pairs, desc="DDIM+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "ddim+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "ddim_p2p",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], source_prompt=pair.get("source_prompt"))
            rows_p2p.append(_make_row(pair, "ddim_p2p", m, str(fp)))
            log.info(f"  [ddim+p2p] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"ddim+p2p failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "ddim_p2p", rows_p2p, source_images=[p["image_path"] for p in pairs])
    del editor
    torch.cuda.empty_cache()

    preproc   = _make_pnp_preprocess(device, MODEL_KEY)
    pnp_model = _make_pnp_model(device, MODEL_KEY)
    rows_pnp: List[Dict] = []
    for pair in tqdm(pairs, desc="DDIM+PnP (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _, recon = preproc.extract_latents(_res,
                                               src_prompt=pair["source_prompt"])
            out_t  = pnp_model.run_pnp(recon, pair["target_prompt"],
                                        guidance_scale=7.5)
            edited = _pnp_tensor_to_pil(out_t)
            fp = save_image(edited, out_dir, "ddim_pnp",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], source_prompt=pair.get("source_prompt"))
            rows_pnp.append(_make_row(pair, "ddim_pnp", m, str(fp)))
            log.info(f"  [ddim+pnp] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"ddim+pnp failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "ddim_pnp", rows_pnp, source_images=[p["image_path"] for p in pairs])
    log.info("DDIM+P2P + DDIM+PnP (SD 1.4) done.")

def run_nulltext_sd21(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """Null-text inversion + P2P on CompVis/stable-diffusion-v1-4 (SD 1.4)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("nulltext_sd14")
    log.info(f"Starting Null-text+P2P (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["null-text-inversion+p2p"], device, num_ddim_steps=50)
    _swap_p2p_editor_to_model(editor, SD14_MODEL, device)

    evaluator = MetricsEvaluator(str(device))
    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="Null-text+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "null-text-inversion+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "null_text_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "null_text_sd14", m, str(fp)))
            log.info(f"  [null-text+p2p SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"null-text+p2p SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "null_text_sd14", rows, source_images=[p["image_path"] for p in pairs])
    log.info("Null-text+P2P (SD 1.4) done.")

def run_pnpinv_p2p_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DirectInversion+P2P on CompVis/stable-diffusion-v1-4 (GPU 2)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("pnpinv_p2p_sd14")
    log.info(f"Starting DirectInv+P2P (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    from models.p2p_editor import P2PEditor
    editor = P2PEditor(["directinversion+p2p"], device, num_ddim_steps=50)
    _swap_p2p_editor_to_model(editor, SD14_MODEL, device)

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="DirectInv+P2P (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            _combined = editor(
                "directinversion+p2p", image_path=_res,
                prompt_src=pair["source_prompt"],
                prompt_tar=pair["target_prompt"],
                guidance_scale=7.5,
                cross_replace_steps=0.4, self_replace_steps=0.6,
            )
            _pw = _combined.width // 4
            edited = _combined.crop((3 * _pw, 0, 4 * _pw, _combined.height))
            fp = save_image(edited, out_dir, "pnpinv_p2p_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "pnpinv_p2p_sd14", m, str(fp)))
            log.info(f"  [pnpinv+p2p SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"pnpinv+p2p SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "pnpinv_p2p_sd14", rows, source_images=[p["image_path"] for p in pairs])
    log.info("DirectInv+P2P (SD 1.4) done.")

def run_pnpinv_pnp_sd14(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """DirectInversion+PnP on CompVis/stable-diffusion-v1-4 (GPU 5)."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    log = _setup_logger("pnpinv_pnp_sd14")
    log.info(f"Starting DirectInv+PnP (SD 1.4) on GPU {gpu_id}")

    pnp_root = METHODS_ROOT / "PnPInversion"
    _setup_pnp_repo(pnp_root)

    SD14_MODEL = "CompVis/stable-diffusion-v1-4"
    evaluator = MetricsEvaluator(str(device))

    preproc   = _make_pnp_preprocess(device, SD14_MODEL)
    pnp_model = _make_pnp_model(device, SD14_MODEL)
    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="DirectInv+PnP (SD1.4)"):
        src_img = _safe_load_pil(pair["image_path"], log)
        if src_img is None:
            continue
        _res = _find_image(pair["image_path"]) or pair["image_path"]
        try:
            torch.cuda.empty_cache()
            inverted_x, _ = preproc.extract_latents(
                _res, src_prompt=pair["source_prompt"])
            out_t  = pnp_model.run_pnp(inverted_x, pair["target_prompt"],
                                        guidance_scale=7.5)
            edited = _pnp_tensor_to_pil(out_t)
            fp = save_image(edited, out_dir, "pnpinv_pnp_sd14",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "pnpinv_pnp_sd14", m, str(fp)))
            log.info(f"  [pnpinv+pnp SD1.4] {pair['base_name']} {pair['code']}"
                     f"  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"pnpinv+pnp SD1.4 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())
    write_csv(out_dir, "pnpinv_pnp_sd14", rows, source_images=[p["image_path"] for p in pairs])
    log.info("DirectInv+PnP (SD 1.4) done.")

def run_splitflow_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("splitflow_sd35")
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, edit_guidance_scale=13.5,
                n_min=0, n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, "splitflow_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "splitflow_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows, source_images=[p["image_path"] for p in pairs])
    log.info("SplitFlow (SD 3.5) done.")

def run_flowedit_sd35_conflictaware_cosine(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                     kappa_tar: float = 0.9, expr_name: str = "flowedit_sd35_conflictaware_cosine"):
    """FlowEdit Conflict-Aware on stabilityai/stable-diffusion-3.5-large.
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting FlowEdit (SD 3.5) on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_ConflictAware_Cosine
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc="FlowEdit (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img  = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_ConflictAware_Cosine(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                kappa_tar=kappa_tar,
                n_min=0, n_max=33,
            )
            x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"FlowEdit SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows, source_images=[p["image_path"] for p in pairs])
    log.info(f"Conflict-Aware FlowEdit (SD 3.5) done.")

def run_ftedit_only(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FTEdit + AdaLN on SD 3.5-Large (no SplitFlow).
    num_steps=30, inv_cfg=1.0, recov_cfg=2.0, skip_steps=0,
    ly_ratio=1.0, attn_ratio=0.15, num_fixpoint_steps=3, average_step_ranges=(0,5).
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("ftedit_only")
    log.info(f"Starting FTEdit (SD 3.5) on GPU {gpu_id}")

    evaluator = MetricsEvaluator(device)
    _run_ftedit(pairs, device, out_dir, log, evaluator)
    log.info("FTEdit (SD 3.5) done.")

def run_irfds_sd35(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """iRFDS score distillation (SD3-medium, matching iRFDS_sd3.py hardcode).
    max_iters=1400, lr=2e-3, guidance_scale=2.0, num_inference_steps=15.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger("irfds_sd35")
    log.info(f"Starting iRFDS (SD 3.5) on GPU {gpu_id}")

    irfds_root = str(METHODS_ROOT / "rectified_flow_prior")
    if irfds_root not in sys.path:
        sys.path.insert(0, irfds_root)
    os.chdir(irfds_root)

    import types as _types, builtins as _bi
    _C_EXT_STUBS = {"tinycudann", "igl", "envlight", "nvdiffrast", "nerfacc",
                    "open3d", "pymeshlab", "xatlas", "wandb", "cv2"}
    _real_import = _bi.__import__

    def _permissive_import(name, globs=None, locs=None, fromlist=(), level=0):
        try:
            return _real_import(name, globs, locs, fromlist, level)
        except (ImportError, ModuleNotFoundError):
            root = name.split(".")[0]
            if root in _C_EXT_STUBS:
                for i, part in enumerate(name.split(".")):
                    full = ".".join(name.split(".")[:i + 1])
                    if full not in sys.modules:
                        sys.modules[full] = _types.ModuleType(full)
                return sys.modules[name.split(".")[0]]
            raise

    _bi.__import__ = _permissive_import
    import importlib.machinery as _imach
    for _s in _C_EXT_STUBS:
        if _s not in sys.modules:
            _m = _types.ModuleType(_s)
            _m.__spec__ = _imach.ModuleSpec(_s, None)
            sys.modules[_s] = _m
    sys.modules["tinycudann"].free_temporary_memory = lambda: None
    sys.modules["igl"].fast_winding_number_for_meshes = lambda *a, **kw: None
    sys.modules["igl"].point_mesh_squared_distance    = lambda *a, **kw: (None,) * 3
    sys.modules["igl"].read_obj                       = lambda *a, **kw: (None,) * 6
    _nvdr_torch = _types.ModuleType("nvdiffrast.torch")
    for _cls in ("RasterizeGLContext", "RasterizeCudaContext", "DepthPeelContext"):
        setattr(_nvdr_torch, _cls,
                type(_cls, (), {"__init__": lambda *a, **kw: None}))
    for _fn in ("rasterize", "interpolate", "antialias", "texture", "antialias_func"):
        setattr(_nvdr_torch, _fn, lambda *a, **kw: (None, None))
    sys.modules["nvdiffrast.torch"] = _nvdr_torch
    sys.modules["nvdiffrast"].torch = _nvdr_torch
    try:
        import threestudio
    finally:
        _bi.__import__ = _real_import

    from diffusers import StableDiffusion3Pipeline
    import torch.nn.functional as F
    import torchvision.transforms as T

    SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL, torch_dtype=torch.float16)
    pipe_sd3.enable_model_cpu_offload(gpu_id=0)
    pipe_sd3.set_progress_bar_config(disable=True)

    guidance_cfg = {
        "half_precision_weights":        True,
        "view_dependent_prompting":      False,
        "guidance_scale":                1.0,
        "pretrained_model_name_or_path": SD3_MODEL,
        "min_step_percent":              0.02,
        "max_step_percent":              0.98,
    }
    pp_cfg = {"pretrained_model_name_or_path": SD3_MODEL, "spawn": False}
    to_tensor = T.Compose([T.ToTensor()])
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="iRFDS (SD3.5)"):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            img_t   = to_tensor(src_img).unsqueeze(0).to(device)
            img_512 = F.interpolate(img_t, (512, 512), mode="bilinear",
                                    align_corners=False)

            pp_copy = dict(pp_cfg, prompt=pair["source_prompt"])
            guidance = threestudio.find("iRFDS-sd3")(guidance_cfg).to(device)
            guidance.camera_embedding = guidance.camera_embedding.to(device)
            prompt_processor = threestudio.find("sd3-prompt-processor")(pp_copy)

            with torch.no_grad():
                target_latent = guidance.encode_images(img_512)

            target    = target_latent.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([target], lr=2e-3, weight_decay=0)

            max_iters      = 1400
            n_accumulation = 2
            prompt_utils   = prompt_processor()
            dummy_cam  = torch.zeros([1, 4, 4], device=device)
            dummy_elev = torch.zeros([1], device=device)
            dummy_azim = torch.zeros([1], device=device)
            dummy_dist = torch.zeros([1], device=device)

            for step in range(max_iters * n_accumulation + 1):
                loss_dict = guidance(
                    noise_to_optimize=target,
                    rgb=target_latent.permute(0, 2, 3, 1),
                    prompt_utils=prompt_utils,
                    mvp_mtx=dummy_cam, elevation=dummy_elev,
                    azimuth=dummy_azim, camera_distances=dummy_dist,
                    c2w=dummy_cam.clone(), rgb_as_latents=True,
                )
                loss = (loss_dict["loss_iRFDS"]
                        + loss_dict["loss_regularize"]) / n_accumulation
                loss.backward()
                if (step + 1) % n_accumulation == 0:
                    actual_step = (step + 1) // n_accumulation
                    guidance.update_step(epoch=0, global_step=actual_step)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                out = pipe_sd3(
                    prompt=pair["target_prompt"],
                    latents=target.detach(),
                    num_inference_steps=15,
                    guidance_scale=2.0,
                    output_type="pil",
                )
            edited = out.images[0]
            del guidance, prompt_processor
            torch.cuda.empty_cache()

            fp = save_image(edited, out_dir, "irfds_sd35",
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, "irfds_sd35", m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"iRFDS SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "irfds_sd35", rows, source_images=[p["image_path"] for p in pairs])
    log.info("iRFDS (SD 3.5) done.")

def run_splitflow_sd35_conflictaware_cosine(pairs: List[Dict], gpu_id: int, out_dir: Path,
                                            kappa_tar: float = 0.9, expr_name: str = "splitflow_sd35_conflictaware_cosine"):
    """SplitFlow on stabilityai/stable-diffusion-3.5-large.
    LLM: mistralai/Mistral-7B-Instruct-v0.3
    T_steps=50, n_avg=1, src_g=3.5, tar_g=13.5, n_min=0, n_max=33.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting SplitFlow (SD 3.5) on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3_ConflictAware_Cosine
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc="SplitFlow (SD3.5)"):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3_ConflictAware_Cosine(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale_base=3.5, tar_guidance_scale_base=13.5,
                kappa_tar=kappa_tar,
                n_max=33,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name,
                            pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"SplitFlow SD3.5 failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, "splitflow_sd35", rows, source_images=[p["image_path"] for p in pairs])
    log.info("SplitFlow (SD 3.5) done.")

def _calc_v_sd3_cfg_zero(pipe, model_input, prompt_embeds, pooled_embeds,
                         guidance_scale, t, step_index, zero_init_steps=2):
    """
    SD3.5 transformer call with CFG-Zero* guidance [Fan et al., arXiv:2503.18886].

    model_input is a 2-item batch [uncond_latent, cond_latent].

    step_index < zero_init_steps:
        Returns torch.zeros_like(model_input[:half_batch]) — exactly as in the
        reference implementation — without calling the transformer. This avoids
        the large first-step guidance error when starting from pure noise t≈1.0.
    step_index >= zero_init_steps:
        Runs the transformer, splits [v_uncond, v_cond], applies star correction
        (s*) + CFG guidance via star_correction() from schedulers.py.
    """
    half_batch = model_input.shape[0] // 2
    if step_index < zero_init_steps:
        return torch.zeros_like(model_input[:half_batch])

    timestep = t.expand(model_input.shape[0])
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    v_uncond, v_cond = noise_pred.chunk(2)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import star_correction
    return star_correction(v_uncond, v_cond, guidance_scale)

def _run_flowedit_sd35_cfg_zero(pairs, gpu_id, out_dir, expr_name,
                                          zero_init_steps=2):
    """
    FlowEdit SD3.5 with true CFG-Zero* zero-init, starting from pure noise
    (n_max = T_steps = 50) [Fan et al., arXiv:2503.18886].

    Implements the reference FlowEditSD3_CFGZero logic directly:
    - Source and target are processed in SEPARATE 2-item batches [uncond, cond],
      matching the reference script (not the joint 4-item batch of FlowEditSD3).
    - For step_index < zero_init_steps: both vt_src and vt_tar are zero tensors
      (no transformer call) so V_delta = 0 and zt_edit is unchanged.
    - For remaining steps: star correction (s*) + CFG guidance is applied.
    """
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    evaluator = MetricsEvaluator(
        device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    T_STEPS = 50
    SRC_CFG = 3.5
    TAR_CFG = 13.5
    N_AVG   = 1

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc=expr_name):
        try:
            fp_check = out_dir / expr_name / f"{pair['base_name']}_{pair['code']}.png"
            if fp_check.exists():
                log.info(f"  [skip] {pair['base_name']} {pair['code']} already exists")
                continue

            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            pipe.scheduler.set_timesteps(T_STEPS, device=device)
            timesteps = pipe.scheduler.timesteps

            pipe._guidance_scale = SRC_CFG
            src_pe, src_npe, src_ppe, src_nppe = pipe.encode_prompt(
                prompt=pair["source_prompt"], prompt_2=None, prompt_3=None,
                negative_prompt="", do_classifier_free_guidance=True,
                device=device)
            pipe._guidance_scale = TAR_CFG
            tar_pe, tar_npe, tar_ppe, tar_nppe = pipe.encode_prompt(
                prompt=pair["target_prompt"], prompt_2=None, prompt_3=None,
                negative_prompt="", do_classifier_free_guidance=True,
                device=device)

            src_embeds = torch.cat([src_npe, src_pe])
            src_pooled = torch.cat([src_nppe, src_ppe])
            tar_embeds = torch.cat([tar_npe, tar_pe])
            tar_pooled = torch.cat([tar_nppe, tar_ppe])
            del src_pe, src_npe, src_ppe, src_nppe
            del tar_pe, tar_npe, tar_ppe, tar_nppe
            torch.cuda.empty_cache()

            zt_edit = x_src.clone()

            for step_i, t in enumerate(timesteps):
                torch.cuda.empty_cache()

                t_curr = (t / 1000.0).float()
                if step_i + 1 < len(timesteps):
                    t_prev = (timesteps[step_i + 1] / 1000.0).float()
                else:
                    t_prev = torch.zeros_like(t_curr)
                dt = t_prev - t_curr

                v_delta_avg = torch.zeros_like(x_src)
                for _ in range(N_AVG):
                    noise  = torch.randn_like(x_src)
                    zt_src = (1 - t_curr) * x_src + t_curr * noise
                    zt_tar = zt_edit + zt_src - x_src
                    del noise

                    model_in_src = torch.cat([zt_src, zt_src])
                    del zt_src
                    vt_src = _calc_v_sd3_cfg_zero(
                        pipe, model_in_src, src_embeds, src_pooled,
                        SRC_CFG, t, step_i, zero_init_steps)
                    del model_in_src

                    model_in_tar = torch.cat([zt_tar, zt_tar])
                    del zt_tar
                    vt_tar = _calc_v_sd3_cfg_zero(
                        pipe, model_in_tar, tar_embeds, tar_pooled,
                        TAR_CFG, t, step_i, zero_init_steps)
                    del model_in_tar

                    v_delta_avg += (vt_tar - vt_src) / N_AVG
                    del vt_src, vt_tar

                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + v_delta_avg * dt
                zt_edit = zt_edit.to(v_delta_avg.dtype)
                del v_delta_avg

            x_tar_denorm = (zt_edit / pipe.vae.config.scaling_factor
                            + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"],
                                       pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows, source_images=[p["image_path"] for p in pairs])
    log.info(f"{expr_name} done.")

def _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: FlowEdit SD3.5 with a cfg_scheduler function."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from FlowEdit_utils import FlowEditSD3_Scheduler
    from diffusers import StableDiffusion3Pipeline

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))

    rows: List[Dict] = []
    for pair in tqdm(pairs, desc=expr_name):
        try:
            fp_check = out_dir / expr_name / f"{pair['base_name']}_{pair['code']}.png"
            if fp_check.exists():
                log.info(f"  [skip] {pair['base_name']} {pair['code']} already exists")
                continue
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)
            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x_src_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x_src = ((x_src_denorm - pipe.vae.config.shift_factor)
                     * pipe.vae.config.scaling_factor).to(device)

            x_tar = FlowEditSD3_Scheduler(
                pipe, scheduler, x_src,
                pair["source_prompt"], pair["target_prompt"],
                negative_prompt="",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, tar_guidance_scale=13.5,
                n_min=0, n_max=33,
                cfg_scheduler=cfg_scheduler,
            )
            x_tar_denorm = x_tar / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows, source_images=[p["image_path"] for p in pairs])
    log.info(f"{expr_name} done.")

def _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: SplitFlow SD3.5 with a cfg_scheduler function."""
    import random, re as _re
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    sf_root = str(METHODS_ROOT / "SplitFlow")
    if sf_root not in sys.path:
        sys.path.insert(0, sf_root)

    from SplitFlow_utils import SplitFlowSD3_Scheduler
    from diffusers import StableDiffusion3Pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL, torch_dtype=torch.float16).to(device)
    scheduler = pipe.scheduler

    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16)

    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc=expr_name):
        try:
            _raw = _safe_load_pil(pair["image_path"], log)
            if _raw is None:
                continue
            src_img = _crop16(_raw)

            llm_prompt = (
                f"Given the source sentence:\n\"{pair['source_prompt']}\"\n"
                f"and the target sentence:\n\"{pair['target_prompt']}\"\n\n"
                "Split the target sentence into three concise sentences "
                "based on step-by-step changes.\n"
                "List each as a numbered item.\n"
                "Do not include any explanation or reasoning.\n"
            )
            inputs = tokenizer_llm(llm_prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**inputs, max_new_tokens=200)
            decoded     = tokenizer_llm.decode(out_ids[0], skip_special_tokens=True)
            intermed    = _re.findall(r"\d+\.\s*(.*)", decoded)
            tar_prompts = intermed + [pair["target_prompt"]]

            img_proc = pipe.image_processor.preprocess(src_img).to(device, torch.float16)
            with torch.autocast("cuda"), torch.inference_mode():
                x0_denorm = pipe.vae.encode(img_proc).latent_dist.mode()
            x0_src = ((x0_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor).to(device)

            x0_tar = SplitFlowSD3_Scheduler(
                pipe, scheduler, x0_src,
                pair["source_prompt"], tar_prompts, "",
                T_steps=50, n_avg=1,
                src_guidance_scale=3.5, edit_guidance_scale=13.5,
                n_min=0, n_max=33,
                cfg_scheduler=cfg_scheduler,
            )
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor
                             + pipe.vae.config.shift_factor)
            with torch.autocast("cuda"), torch.inference_mode():
                img_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            edited = pipe.image_processor.postprocess(img_out)[0]

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows, source_images=[p["image_path"] for p in pairs])
    log.info(f"{expr_name} done.")

def _run_ftedit_with_scheduler(pairs, gpu_id, out_dir, expr_name, cfg_scheduler):
    """Internal: FTEdit SD3.5 with a cfg_scheduler function."""
    import random
    torch.manual_seed(42)
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    log = _setup_logger(expr_name)
    log.info(f"Starting {expr_name} on GPU {gpu_id}")

    ftedit_root = str(METHODS_ROOT / "FTEdit")
    if ftedit_root not in sys.path:
        sys.path.insert(0, ftedit_root)

    from mmdit.sd35_pipeline import StableDiffusion3Pipeline as SD35Pipeline
    from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
    from controller import attn_norm_ctrl_sd35

    SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
    pipe = SD35Pipeline.from_pretrained(SD35_MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()

    saved_path = str(out_dir / expr_name)
    Path(saved_path).mkdir(parents=True, exist_ok=True)

    invf = Inversed_flow_fixpoint_residual(
        pipe, steps=30, device=device,
        inv_cfg=1.0, recov_cfg=1.0, skip_steps=7,
        saved_path=saved_path,
    )
    evaluator = MetricsEvaluator(device, pie_bench=bool(pairs and pairs[0].get('mask_encoded')))
    rows: List[Dict] = []

    for pair in tqdm(pairs, desc=expr_name):
        try:
            src_img = _safe_load_pil(pair["image_path"], log)
            if src_img is None:
                continue
            _img_path = _find_image(pair["image_path"]) or pair["image_path"]
            prompts = [pair["source_prompt"], pair["target_prompt"]]

            attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, None, None)
            all_latents = invf.euler_flow_inversion(
                prompt=pair["source_prompt"],
                image=_img_path,
                num_fixpoint_steps=3,
                average_step_ranges=(0, 5),
            )

            controller_ada  = attn_norm_ctrl_sd35.Adalayernorm_replace(
                prompts, 30, 1.0,
                pipe.tokenizer, pipe.tokenizer_3, device=device,
            )
            controller_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts, 30, 1.0)
            attn_norm_ctrl_sd35.register_attention_control_sd35(
                pipe, controller_attn, controller_ada
            )

            _image1, image2 = invf.edit_img_with_residual(
                prompts, all_latents, controller_ada,
                cfg_scheduler=cfg_scheduler,
            )

            if isinstance(image2, np.ndarray):
                arr = np.squeeze(image2)
                if arr.dtype != np.uint8:
                    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                edited = Image.fromarray(arr)
            else:
                edited = image2

            fp = save_image(edited, out_dir, expr_name, pair["base_name"], pair["code"])
            m  = evaluator.all_metrics(src_img, edited, pair["target_prompt"], pair.get("mask_encoded"), source_prompt=pair.get("source_prompt"))
            rows.append(_make_row(pair, expr_name, m, str(fp)))
            log.info(f"  [{expr_name}] {pair['base_name']} {pair['code']}  CLIP={m.get('clip_similarity_target_image', 0):.3f}")
        except Exception:
            log.error(f"{expr_name} failed {pair['base_name']} {pair['code']}:\n"
                      + traceback.format_exc())

    write_csv(out_dir, expr_name, rows, source_images=[p["image_path"] for p in pairs])
    log.info(f"{expr_name} done.")

def run_flowedit_sd35_interval(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit SD3.5 + Interval CFG scheduler [Kynkäänniemi et al., NeurIPS 2024]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_interval
    _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                      "flowedit_sd35_interval", scheduler_interval)

def run_flowedit_sd35_monotone(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """FlowEdit SD3.5 + Monotone cosine-decay CFG scheduler [Wang et al., 2404.13040]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_monotone
    _run_flowedit_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                      "flowedit_sd35_monotone", scheduler_monotone)

def run_flowedit_sd35_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """
    FlowEdit SD3.5 + CFG-Zero* with true zero-init [Fan et al., arXiv:2503.18886].
    Implements the reference loop directly (separate 2-item batches, torch.zeros_like
    skip for first 2 steps). See _run_flowedit_sd35_cfg_zero.
    """
    _run_flowedit_sd35_cfg_zero(pairs, gpu_id, out_dir,
                                "flowedit_sd35_zeroinit", zero_init_steps=2)

def run_splitflow_sd35_interval(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + Interval CFG scheduler [Kynkäänniemi et al., NeurIPS 2024]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_interval
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_interval", scheduler_interval)

def run_splitflow_sd35_monotone(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + Monotone cosine-decay CFG scheduler [Wang et al., 2404.13040]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_monotone
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_monotone", scheduler_monotone)

def run_splitflow_sd35_zeroinit(pairs: List[Dict], gpu_id: int, out_dir: Path):
    """SplitFlow SD3.5 + CFG-Zero* (zero-init + star correction) [Fan et al., 2503.18886]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from schedulers import scheduler_zero_init
    _run_splitflow_sd35_with_scheduler(pairs, gpu_id, out_dir,
                                       "splitflow_sd35_zeroinit", scheduler_zero_init)

def merge_csvs(out_dir: Path) -> Path:
    all_rows = []
    for f in sorted(f for f in out_dir.glob("results_*.csv") if f.name != "results_merged.csv"):
        with open(f) as fh:
            all_rows.extend(list(csv.DictReader(fh)))
    merged = out_dir / "results_merged.csv"
    if all_rows:
        with open(merged, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_rows[0].keys(),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
    return merged

def print_summary(merged_csv: Path, fid_scores: Dict[str, float] = None):
    """Print per-method averages from per-method CSVs (new format: file_id + metrics + fid)."""
    if fid_scores is None:
        fid_scores = {}

    out_dir = merged_csv.parent
    from collections import defaultdict
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnts: Dict[str, int]              = defaultdict(int)
    fid_by_method: Dict[str, float]   = {}
    active_cols: list                 = []

    for csv_path in sorted(f for f in out_dir.glob("results_*.csv")
                           if f.name != "results_merged.csv"):
        method = csv_path.stem[len("results_"):]
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("file_id") == "FID_SUMMARY":
                        try:
                            fid_by_method[method] = float(r.get("fid", "nan"))
                        except ValueError:
                            fid_by_method[method] = float("nan")
                        continue
                    if not active_cols and r:
                        active_cols = [c for c in r.keys()
                                       if c not in ("file_id", "fid")]
                    cnts[method] += 1
                    for col in active_cols:
                        try:
                            sums[method][col] += float(r[col])
                        except (ValueError, KeyError):
                            pass
        except FileNotFoundError:
            pass

    if not cnts:
        print("[Summary] no results found.")
        return

    methods = sorted(cnts.keys())
    avgs = {m: {c: sums[m][c] / cnts[m] for c in active_cols} for m in methods}

    show_cols = active_cols
    col_w = 12
    hdr   = f"{'Method':<30}" + "".join(f"{c:>{col_w}}" for c in show_cols) + f"{'FID':>{col_w}}"
    sep   = "─" * len(hdr)
    print("\n" + sep)
    print("  BENCHMARK SUMMARY")
    print(sep)
    print(hdr)
    print(sep)
    for m in methods:
        row_str = f"{m:<30}"
        for c in show_cols:
            v = avgs[m].get(c, float("nan"))
            row_str += f"{v:>{col_w}.3f}"
        fid = fid_by_method.get(m, fid_scores.get(m, float("nan")))
        try:
            row_str += f"{float(fid):>{col_w}.2f}"
        except (ValueError, TypeError):
            row_str += f"{'nan':>{col_w}}"
        print(row_str)
    print(sep + "\n")

_RUNNERS = [
    (run_new_ddim_sd14,                       0),
    (run_nulltext_sd21,                       1),
    (run_pnpinv_p2p_sd14,                    2),
    (run_pnpinv_pnp_sd14,                    3),
    (run_rf_inversion,                        4),
    (run_rf_solver,                           5),
    (run_fireflow,                            6),
    (run_irfds_sd35,                          7),
    (run_ftedit_only,                         0),
    (run_flowedit_sd35,                       1),
    (run_flowedit_sd35_interval,              2),
    (run_flowedit_sd35_monotone,              3),
    (run_flowedit_sd35_zeroinit,              4),
    (run_flowedit_sd35_conflictaware_cosine,  5),
    (run_splitflow_sd35,                      6),
    (run_splitflow_sd35_interval,             7),
    (run_splitflow_sd35_monotone,             0),
    (run_splitflow_sd35_zeroinit,             1),
    (run_splitflow_sd35_conflictaware_cosine, 2),
]

_METHOD_NAMES = [
    ["ddim_p2p", "ddim_pnp"],
    ["null_text_sd14"],
    ["pnpinv_p2p_sd14"],
    ["pnpinv_pnp_sd14"],
    ["rf_inversion"],
    ["rf_solver"],
    ["fireflow"],
    ["irfds_sd35"],
    ["ftedit"],
    ["flowedit_sd35"],
    ["flowedit_sd35_interval"],
    ["flowedit_sd35_monotone"],
    ["flowedit_sd35_zeroinit"],
    ["flowedit_sd35_conflictaware_cosine"],
    ["splitflow_sd35"],
    ["splitflow_sd35_interval"],
    ["splitflow_sd35_monotone"],
    ["splitflow_sd35_zeroinit"],
    ["splitflow_sd35_conflictaware_cosine"],
]

def _worker(fn, pairs, gpu_id, out_dir_str):
    """Top-level function for multiprocessing.spawn."""
    out_dir = Path(out_dir_str)
    try:
        fn(pairs, gpu_id, out_dir)
    except Exception:
        logging.getLogger(fn.__name__).error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Benchmark image-editing methods")
    parser.add_argument("--yaml",      default="Data/flowedit.yaml")
    parser.add_argument("--images",    default="Data/flowedit_data/")
    parser.add_argument("--outdir",    default=None)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs (e.g. 2 for testing)")
    parser.add_argument("--methods",   nargs="*", default=None,
                        help="Run only specific runners (0-6). "
                             "e.g. --methods 0 3 6 to run DDIM + FlowEdit + SplitFlow")
    parser.add_argument("--pie_bench",   action="store_true",
                        help="Use PIE-Bench dataset instead of FlowEdit YAML")
    parser.add_argument("--pie_mapping", default="Data/PIE-Bench_v1/mapping_file.json",
                        help="Path to PIE-Bench mapping_file.json")
    parser.add_argument("--pie_images",  default="Data/PIE-Bench_v1/annotation_images",
                        help="Path to PIE-Bench annotation_images directory")
    parser.add_argument("--gpu_map", nargs="*", type=int, default=None,
                        help="Override GPU IDs for selected runners (one per --methods entry). "
                             "e.g. --gpu_map 0 0 4 0 5")
    parser.add_argument("--monotonic_alpha", type=float, default=1.0,
                        help="Limit number of pairs (e.g. 2 for testing)")
    parser.add_argument("--monotonic_beta", type=float, default=1.0,
                        help="Limit number of pairs (e.g. 2 for testing)")

    args = parser.parse_args()

    if args.pie_bench:
        mapping_file = str(ROOT / args.pie_mapping)
        images_root  = str(ROOT / args.pie_images)
        pairs = load_pairs_pie(mapping_file, images_root, args.max_pairs)
        print(f"Output directory : {args.outdir or 'auto'}")
        print(f"PIE-Bench mapping: {mapping_file}")
        print(f"Images root      : {images_root}")
    else:
        yaml_path   = str(ROOT / args.yaml)
        images_root = str(ROOT / args.images)
        pairs = load_pairs(yaml_path, images_root, args.max_pairs)
        print(f"Output directory : {args.outdir or 'auto'}")
        print(f"YAML             : {yaml_path}")
        print(f"Images root      : {images_root}")

    if args.outdir is None:
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag     = "pie" if args.pie_bench else "flowedit"
        out_dir = ROOT / "outputs" / f"benchmark_{tag}_{stamp}"
    else:
        out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_dir}")
    print(f"Pairs            : {len(pairs)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump({"pie_bench": args.pie_bench, "images": images_root,
                   "max_pairs": args.max_pairs, "num_pairs": len(pairs)}, f, indent=2)

    runner_indices = list(range(len(_RUNNERS)))
    if args.methods is not None:
        runner_indices = [int(x) for x in args.methods]

    gpu_ids = [_RUNNERS[idx][1] for idx in runner_indices]
    if args.gpu_map is not None:
        if len(args.gpu_map) != len(runner_indices):
            raise ValueError(f"--gpu_map has {len(args.gpu_map)} entries but "
                             f"{len(runner_indices)} runners selected")
        gpu_ids = args.gpu_map

    ctx = mp.get_context("spawn")
    procs = []
    for idx, gpu_id in zip(runner_indices, gpu_ids):
        fn, _ = _RUNNERS[idx]
        p = ctx.Process(
            target=_worker,
            args=(fn, pairs, gpu_id, str(out_dir)),
            name=fn.__name__,
        )
        p.start()
        procs.append(p)
        print(f"  Launched {fn.__name__} → GPU {gpu_id}  (pid={p.pid})")

    for p in procs:
        p.join()
        status = "OK" if p.exitcode == 0 else f"EXIT={p.exitcode}"
        print(f"  {p.name} finished [{status}]")

    merged = merge_csvs(out_dir)
    print(f"Merged CSV       : {merged}")

    print_summary(merged)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
