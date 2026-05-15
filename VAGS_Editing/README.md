# VAGS: Velocity Adaptive Guidance Scale Enhancing Generative Models

Official implementation for **NeurIPS 2026**.

VAGS is a **training-free** adaptive guidance scheduling framework for diffusion-based image editing. At each denoising step, it modulates the CFG scale based on the cosine similarity between source and target velocity fields — suppressing guidance in early noisy steps (where velocities are misaligned) and amplifying it in late semantic steps (where they agree).

---

## Setup

```bash
conda create -n vags python=3.10 -y && conda activate vags
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate sentencepiece
pip install pillow numpy pandas scikit-image lpips open-clip-torch
```

Tested on NVIDIA H100/A100 80 GB with SD 3.5 Large (`stabilityai/stable-diffusion-3.5-large`).

---

## Quick Start

### Edit a single image

```bash
python run_vags.py \
    --src_image inputs/cat.jpg \
    --src_prompt "a cat sitting on a red chair" \
    --tgt_prompt "a dog sitting on a red chair" \
    --output outputs/dog.png
```

### Run on PIE-Bench

```bash
python run_vags.py \
    --dataset pie_bench \
    --outdir outputs/pie_bench \
    --gpu 0
```

### Run on a custom YAML file

```bash
python run_vags.py \
    --dataset yaml \
    --yaml_file Data/flowedit.yaml \
    --images_root Data/ \
    --outdir outputs/custom \
    --gpu 0
```

**Key option:** `--kappa 0.9` (default) controls the modulation strength.

---

## Quick Demo (no dataset needed)

Three example images from the paper are included in `data/`. To run immediately:

```bash
python run_vags.py \
    --dataset yaml \
    --yaml_file data/edits.yaml \
    --images_root . \
    --outdir outputs/demo \
    --gpu 0
```

This edits the three pairs used in the trajectory figure of the paper (cat→tiger, tiger→crochet tiger, raspberries→strawberries).

---

## Data Preparation

**PIE-Bench** — Download from [DirectInversion](https://github.com/cure-lab/DirectInversion) and place at:
```
Data/
└── PIE-Bench_v1/
    ├── annotation_images/
    └── mapping_file.json
```

**Custom pairs** — Define source/target prompt pairs in a YAML file (see `data/edits.yaml` for the format).

---

## Evaluation

After editing, compute metrics with:

```bash
# PIE-Bench (uses foreground masks)
python evaluate.py \
    --annotation_mapping_file Data/PIE-Bench_v1/mapping_file.json \
    --edited_images_dir outputs/pie_bench \
    --tgt_methods flowedit_sd35_conflictaware_cosine \
    --result_path outputs/pie_bench/results_vags.csv

# Custom YAML (full image, no mask)
python evaluate.py \
    --yaml_file Data/flowedit.yaml \
    --src_image_folder Data/ \
    --edited_images_dir outputs/custom \
    --tgt_methods flowedit_sd35_conflictaware_cosine \
    --no_pie_mask \
    --result_path outputs/custom/results_vags.csv
```

Then aggregate:
```bash
python compute_all_metrics.py --result_csv outputs/pie_bench/results_vags.csv
```

---

## Reproducing Paper Results

To reproduce the full comparison table, use `benchmark.py` which runs all baselines and VAGS in parallel across GPUs:

```bash
# PIE-Bench — all methods
python benchmark.py \
    --pie_bench \
    --outdir outputs/pie_bench_full \
    --methods 9 10 11 12 13 \
    --gpu_map 0 1 2 3 4
```

| `--methods` index | Method |
|:-----------------:|--------|
| 9  | FlowEdit (SD 3.5) |
| 10 | FlowEdit + Interval |
| 11 | FlowEdit + Monotone |
| 12 | FlowEdit + Zero-Init |
| **13** | **FlowEdit + VAGS** ← ours |

---

## Method Details

The core function is `FlowEditSD3_ConflictAware_Cosine` in `FlowEdit_utils.py`.

**Adaptive target scale at active step $k$:**

$$\lambda_k = \lambda_{\text{tar}} \cdot \exp\!\left(\kappa\,(2\sigma_k-1)\,s_k\right)$$

where $\lambda_{\text{tar}} = 13.5$ (constant base), $\sigma_k = 1-t_k$, and $s_k = \cos(V_\text{src}^{(k)}, V_\text{tar}^{(k)})$.

At early noisy steps ($\sigma_k < 0.5$, so $2\sigma_k - 1 < 0$), the scale is attenuated when velocities are aligned ($s_k > 0$). At late clean steps ($\sigma_k > 0.5$), it is amplified.

Default hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kappa_tar` | **0.9** | Modulation strength κ |
| `src_guidance_scale` | 3.5 | λ_src (fixed) |
| `tar_guidance_scale` | 13.5 | λ_tar base value |
| `T_steps` | 50 | Total denoising steps |
| `n_max` | 33 | Active editing window |

---

## Results

### PIE-Bench (700 pairs)

| Method | Dist↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP-I↑ |
|--------|------:|------:|-------:|-----:|------:|--------:|
| FlowEdit | 30.31 | 21.87 | 117.06 | 92.70 | 82.54 | 83.69 |
| + Interval | 29.82 | 21.93 | 114.14 | 91.72 | 82.88 | 84.03 |
| + Monotone | 15.91 | 25.43 | 69.87 | 43.47 | 87.67 | 84.73 |
| + Zero-Init | 194.67 | 11.69 | 354.57 | 867.59 | 57.64 | 75.28 |
| **+ VAGS (ours)** | **13.84** | **26.38** | 70.38 | **34.86** | **87.68** | **87.44** |

### DIV2K (272 pairs)

| Method | Dist↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ |
|--------|------:|------:|-------:|-----:|------:|
| FlowEdit | 18.27 | 20.27 | 145.55 | 103.37 | 82.15 |
| + Monotone | 10.89 | 22.47 | 102.91 | 63.91 | 83.64 |
| **+ VAGS (ours)** | **8.62** | **23.61** | **98.43** | **50.22** | **84.41** |

---

## Baseline Methods

Included in `methods/`:

| Method | Reference |
|--------|-----------|
| FlowEdit | [kulikov2024flowedit](https://github.com/fallenshock/FlowEdit) |
| SplitFlow | [yoon2025splitflow](https://github.com/SplitFlow) |
| RF-Inversion | [rout2024rfinversion](https://github.com/LituRout/RFInversion) |
| RF-Solver | [wang2024rfsolver](https://github.com/wangjiangshan0725/RF-Solver-Edit) |
| FireFlow | [deng2024fireflow](https://github.com/HolmesShuan/FireFlow) |
| FTEdit | [xu2025ftedit](https://github.com/FTEdit) |
| DDIM+P2P / DDIM+PnP | [hertz2022prompt2prompt](https://github.com/google/prompt-to-prompt) |
| Null-text Inversion | [mokady2023nulltext](https://github.com/google/null-text-inversion) |
| iRFDS | [yang2025irfds](https://github.com/iRFDS) |

---

## Computational Cost

Measured on a single NVIDIA RTX 6000 Ada (48GB) with SD 3.5 Large, $N=50$ timesteps, $N_{\max}=33$ active steps (averaged over 10 runs):

| | Time per image |
|--|--|
| FlowEdit (baseline) | **~13s** (±0.5s) |
| FlowEdit + VAGS | **~13s** (±0.5s) |

VAGS adds no overhead over standard FlowEdit: both perform the same number of model forward passes. The adaptive guidance reuses the raw transformer outputs from the single forward pass — no extra model call.

---

## Citation

```bibtex
@inproceedings{vags2026,
  title     = {VAGS: Velocity Adaptive Guidance Scale Enhancing Generative Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
