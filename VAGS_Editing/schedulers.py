"""
CFG Schedulers for flow-matching image editing — benchmark_schedulers.py.

Three standalone functions that modulate tar_guidance_scale dynamically
during the denoising loop.

Parameters (all functions)
--------------------------
step_idx    : index of the current active denoising step (0 = noisiest).
total_steps : total number of active denoising steps for this run.
tar_cfg     : the base (maximum) target guidance scale.

References
----------
Interval  : Kynkäänniemi et al., NeurIPS 2024.
            "Applying guidance in a limited interval improves sample and
            distribution quality in diffusion models."
Monotone  : Wang et al., arXiv:2404.13040.
            "Analysis of classifier-free guidance weight schedulers."
Zero-Init : Fan et al., arXiv:2503.18886.
            "CFG-Zero*: Improved classifier-free guidance for flow matching
            models."
"""

import math
import torch


def star_correction(v_uncond: "torch.Tensor", v_cond: "torch.Tensor",
                    tar_guidance_scale: float,
                    eps: float = 1e-8) -> "torch.Tensor":
    """
    CFG-Zero* star correction [Fan et al., arXiv:2503.18886, Eq. 11 + Eq. 6].

    s* = (v_cond · v_uncond) / (‖v_uncond‖² + ε)
    ṽ  = s* · v_uncond + w · (v_cond − s* · v_uncond)

    s* rescales the unconditional velocity to remove its systematic bias
    before computing the guided velocity. This replaces the standard formula
    v_uncond + w*(v_cond - v_uncond) with the corrected version above.

    Implementation note: computed per-batch in float32 (flatten [B,C,H,W]→[B,-1],
    sum over dim=1). The previous global .sum() collapsed all batch elements to a
    single scalar, which is wrong for B>1 and also overflows float16 for 65k+
    elements (SD3.5 latent: 16×64×64 = 65 536 values, max float16 ≈ 65 504).
    """
    # Flatten spatial dims: [B, C, H, W] → [B, N]  (float32 for numerical safety)
    v_c = v_cond.float().view(v_cond.shape[0], -1)
    v_u = v_uncond.float().view(v_uncond.shape[0], -1)

    # Per-batch dot product and squared norm, keepdim so we can divide
    dot   = (v_c * v_u).sum(dim=1, keepdim=True)           # [B, 1]
    norm2 = v_u.pow(2).sum(dim=1, keepdim=True) + eps      # [B, 1]

    # Reshape to [B, 1, 1, 1] for broadcasting with [B, C, H, W]
    s_star = (dot / norm2).view(v_cond.shape[0], *([1] * (v_cond.dim() - 1)))

    return s_star * v_uncond + tar_guidance_scale * (v_cond - s_star * v_uncond)


def scheduler_interval(step_idx: int, total_steps: int, tar_cfg: float,
                       t_low: float = 0.0, t_high: float = 0.9) -> float:
    """
    Interval scheduler [Kynkäänniemi et al., NeurIPS 2024].

    τ = step_idx / total_steps   (0 → noisiest, ~1 → cleanest)
    λ = tar_cfg   if  t_low ≤ τ ≤ t_high
        1.0       otherwise  (conditional only, no guidance correction)

    The paper found optimal guidance in steps 17–22 out of 32 (EDM sampler)
    for diffusion models starting from pure noise. For flow-matching (SD3.5)
    with FlowEdit/SplitFlow/FTEdit, the noisiest steps are already excluded
    by n_max, so the active steps span the medium-to-clean denoising range.
    Defaults t_low=0.0, t_high=0.9 apply guidance from the start of the active
    window and only drop off in the final 10 % (very clean steps), which
    preserves the interval spirit while ensuring complete edits.
    """
    tau = step_idx / max(total_steps, 1)
    return tar_cfg if t_low <= tau <= t_high else 1.0


def scheduler_monotone(step_idx: int, total_steps: int, tar_cfg: float) -> float:
    """
    Monotone cosine-increase scheduler [Wang et al., arXiv:2404.13040].

    τ = step_idx / total_steps   (0 → noisiest, ~1 → cleanest)
    λ = tar_cfg * (0.5 + 0.5 * (1 − cos(π * τ))),  clamped to [tar_cfg/2, tar_cfg]

    Guidance INCREASES monotonically as denoising progresses. The formula is
    shifted to start at tar_cfg/2 (instead of near 0) so that even the noisiest
    active steps receive meaningful edit guidance. This is appropriate for
    FlowEdit/SplitFlow where n_max already skips the pure-noise steps, so the
    active range begins at moderate noise (t ≈ 0.66) rather than pure noise.
    """
    tau = step_idx / max(total_steps, 1)
    # Shifted cosine: ranges from tar_cfg/2 (at τ=0) to tar_cfg (at τ=1)
    lam = tar_cfg * (0.5 + 0.25 * (1.0 - math.cos(math.pi * tau)))
    return max(lam, tar_cfg * 0.5)


def scheduler_zero_init(step_idx: int, total_steps: int,
                        tar_cfg: float,
                        zero_init_steps: int = 2):
    """
    CFG-Zero* with full zero-init [Fan et al., arXiv:2503.18886].

    Reference implementation: skip the transformer entirely for the first K
    steps (V_delta = 0, equivalent to torch.zeros_like before the forward
    pass). For subsequent steps, apply star correction (s*) + full guidance.

        step_idx < zero_init_steps → (0.0,    False, True)   skip_step=True
        step_idx ≥ zero_init_steps → (tar_cfg, True,  False) star correction

    Returns (lambda: float, apply_star_correction: bool, skip_step: bool).
    skip_step=True means the caller must zero out V_delta without calling
    the transformer (true zero-init, not lam=0 with star correction).
    """
    if step_idx < zero_init_steps:
        return 0.0, False, True
    return tar_cfg, True, False
