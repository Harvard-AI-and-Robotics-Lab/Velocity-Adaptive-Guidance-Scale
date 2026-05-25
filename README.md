# VAGS: Velocity-Adaptive Guidance Scale

Official implementation of **Velocity-Adaptive Guidance Scale (VAGS)**, a training-free, plug-and-play modulation of classifier-free guidance for flow-based diffusion models.

VAGS replaces the constant CFG scale with a per-step adaptive scale that depends on the diffusion timestep and the cosine similarity between two velocity fields:

```
$\lambda_i = \lambda \cdot \exp\left(\kappa \cdot (2\sigma_i - 1) \cdot s_i\right)$
$λ_i = λ · exp( κ · (2σ_i − 1) · s_i )$
```

where `s_i` is the velocity-alignment signal, `σ_i ∈ [0, 1]` is the signal level (1 = clean), and `κ ≥ 0` controls modulation strength. Setting `κ = 0` recovers vanilla CFG.

---

## Repository Layout

| Directory | Task | Backbone |
|-----------|------|----------|
| [`VAGS_Generation/`](./VAGS_Generation) | Text-to-image generation (COCO17, CUB-200, Flickr30K) | SD 3.5 |
| [`VAGS_Editing/`](./VAGS_Editing) | Image editing (PIE-Bench, DIV2K) | SD 3.5 + FlowEdit |

Each subdirectory is self-contained with its own environment, scripts, and README.

---

## Quick Start

### Generation
```bash
cd VAGS_Generation
conda env create -f environment.yml && conda activate vags_gen
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_coco17.sh
```

### Editing
```bash
cd VAGS_Editing
conda create -n vags python=3.10 -y && conda activate vags
pip install torch torchvision diffusers transformers accelerate
python run_vags.py --dataset yaml --yaml_file data/edits.yaml --images_root . --outdir outputs/demo --gpu 0
```

See the per-subproject READMEs for full setup, dataset preparation, evaluation, and reproduction details.

---

<!-- ## Citation

```bibtex
@inproceedings{vags2026,
  title     = {VAGS: Velocity Adaptive Guidance Scale Enhancing Generative Models},
  booktitle = {arXiv },
  year      = {2026}
}
``` -->
