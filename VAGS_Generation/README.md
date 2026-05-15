# VAGS: Velocity-Adaptive Guidance Scale for Image Generation

Reference implementation for **Velocity-Adaptive Guidance Scale (VAGS)**, a
training-free, plug-and-play modulation of classifier-free guidance for
flow-based text-to-image generation.

VAGS replaces the constant CFG scale with a per-step adaptive scale that
depends on (i) the diffusion timestep and (ii) the cosine similarity between
the conditional and unconditional velocity fields:

```
λ_i = λ · exp( κ · (2σ_i − 1) · s_i )
```

where `s_i = cos(v_∅, v_c)` is the velocity-alignment signal, `σ_i ∈ [0, 1]`
is the signal level (1 = clean), and `κ ≥ 0` controls the modulation strength.
Setting `κ = 0` recovers vanilla CFG with constant `λ`.

This codebase covers the **image-generation** experiments (SDv3.5 backbone)
on COCO17, CUB-200, and Flickr30K. The image-editing codebase lives in a
separate repository.

This project is built upon [SimpleTuner](https://github.com/bghira/SimpleTuner).

## Environment

```bash
conda env create -f environment.yml
conda activate vags_gen
```

Or manually:
```bash
conda create --name vags_gen python=3.11
conda activate vags_gen
pip install -r requirements.txt
```

## Datasets

Download the three benchmarks used in the paper:
```bash
python scripts/download_coco17.py
python scripts/download_flickr30k.py
# CUB-200 follows the standard release; see paper §4 for details.
```

Each script writes to `<DATA_ROOT>/datasets/<dataset>/...`. Update the path
in the script (look for `/path/to/datasets/...`) before running.

For COCO17 and Flickr30K, we use the CLIP model released by OpenAI to evaluate CLIPScores for each image-caption pair. Then, we use the caption with the best CLIPScore for inference and evaluation purposes, i.e.,

```bash
python scripts/measure_clipscore.py
python scripts/extract_one_caption_to_txt_best_clipscore.py
```

## Hugging Face login

Required for the SDv3.5 weights:
```bash
huggingface-cli login
```

## Sampling — vanilla SDv3.5 baselines

```bash
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_cub200.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_flickr30k.sh
```

## Sampling — VAGS-Gen

```bash
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_cub200.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_flickr30k.sh
```

The κ value (modulation strength) is set inside each script. Defaults match
the paper; sweep `κ ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0}` for the
ablation in Tab. 5.

## Trajectory analysis (per-step cos / λ)

The `*_analysis.sh` variants additionally log the per-step cosine similarity
and adaptive guidance scale used to produce Figs. 11–13 of the paper:
```bash
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_coco17_analysis.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_cub200_analysis.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_vagsgen_flickr30k_analysis.sh
```

## Evaluation

### Image-quality + image–text alignment (FID, IS, CLIPScore)
```bash
bash evaluate_on_image_generation.sh
```

### Caption-based metrics (BLEU-4, METEOR, ROUGE-L)
```bash
# Step 1: generate BLIP captions for the generated images
bash evaluate_with_blip_caption.sh
# Step 2: score captions against references
bash evaluate_generated_images.sh
```

### CLAIR (GPT-3.5-turbo caption similarity)
```bash
export OPENAI_API_KEY=sk-...     # required
bash evaluate_CLAIR_with_generated_captions_coco17.sh
bash evaluate_CLAIR_with_generated_captions_cub200.sh
bash evaluate_CLAIR_with_generated_captions_flickr30k.sh
```

### Other metrics
```bash
bash evaluate_pickscore.sh        # PickScore
bash evaluate_image_reward.sh     # ImageReward
```

## Reproducing paper figures

The `analysis/` directory contains plotting and trajectory-aggregation
scripts (`plot_cos_sim_step.py`, `plot_cos_sim_prompt.py`,
`plot_adaptive_cfg_step.py`, `plot_adaptive_cfg_prompt.py`) used to render
Figs. 11–13 from the per-step CSVs produced by the `*_analysis.sh` runs.

## Citation

```bibtex
% TODO: add citation block once paper is on arXiv
```
