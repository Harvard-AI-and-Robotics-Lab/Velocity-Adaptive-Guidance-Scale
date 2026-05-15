from typing import Optional, Union
import torch
from tqdm import tqdm
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import math

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def calc_v_sd3(
        pipe, src_tar_latent_model_input,
        src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
        src_guidance_scale, tar_guidance_scale, t,
        apply_star_correction: bool = False):
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            if apply_star_correction:
                from schedulers import star_correction
                noise_pred_tar = star_correction(tar_noise_pred_uncond, tar_noise_pred_text, tar_guidance_scale)
            else:
                noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar

def calc_v_flux(
        pipe, latents, prompt_embeds,
        pooled_prompt_embeds, guidance,
        text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred

@torch.no_grad()
def FlowEditSD3(pipe,
                scheduler,
                x_src,
                src_prompt,
                tar_prompt,
                negative_prompt,
                T_steps: int = 50,
                n_avg: int = 1,
                src_guidance_scale: float = 3.5,
                tar_guidance_scale: float = 13.5,
                n_min: int = 0,
                n_max: int = 15,):

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale

    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t/1000
        if i+1 < len(timesteps):
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat(
                    [zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_model_input, src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds, src_guidance_scale,
                    tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:

            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat(
                [xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_Scheduler(pipe,
                scheduler,
                x_src,
                src_prompt,
                tar_prompt,
                negative_prompt,
                T_steps: int = 50,
                n_avg: int = 1,
                src_guidance_scale: float = 3.5,
                tar_guidance_scale: float = 13.5,
                n_min: int = 0,
                n_max: int = 15,
                cfg_scheduler=None,):

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale

    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    zt_edit = x_src.clone()

    n_active = max(n_max - n_min, 1)
    active_step = 0

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t/1000
        if i+1 < len(timesteps):
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if cfg_scheduler is None:
            lam, apply_star = tar_guidance_scale, False
        else:
            result = cfg_scheduler(active_step, n_active, tar_guidance_scale)
            lam, apply_star = result if isinstance(result, tuple) else (result, False)
        active_step += 1

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat(
                    [zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_model_input, src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds, src_guidance_scale,
                    lam, t, apply_star_correction=apply_star)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:

            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat(
                [xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, lam, t, apply_star_correction=apply_star)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.7,
    softness_m: float = 1.0,
):
    """
    Adaptive Guidance FlowEdit for SD3.

    In addition to the standard FlowEdit parameters, this variant introduces:
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant that controls the saturation of the
                      conflict signal  s_i = Δ_i / (Δ_i + m)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot).item()

                sigma_i = 1.0 - t_i.item() if isinstance(t_i, torch.Tensor) else 1.0 - t_i
                s_i = delta_i / (delta_i + softness_m)

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (2.0 * sigma_i - 1.0) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (2.0 * sigma_i - 1.0) * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware_Normalization(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.7,
    softness_m: float = 1.0,
    normalization_type: int = 1,
):
    """
    Adaptive Guidance FlowEdit for SD3.

    In addition to the standard FlowEdit parameters, this variant introduces:
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant that controls the saturation of the
                      conflict signal  s_i = Δ_i / (Δ_i + m)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot).item()

                sigma_i = 1.0 - t_i.item() if isinstance(t_i, torch.Tensor) else 1.0 - t_i

                if normalization_type == 1:
                    s_i = delta_i / (torch.norm(Vt_tar_pilot).item() + softness_m)
                elif normalization_type == 2:
                    s_i = delta_i / (torch.norm(Vt_src_pilot).item() + softness_m)
                elif normalization_type == 3:
                    s_i = delta_i / (max(torch.norm(Vt_tar_pilot).item(), torch.norm(Vt_src_pilot).item()) + softness_m)
                elif normalization_type == 4:
                    s_i = torch.nn.functional.cosine_similarity(
                        Vt_src_pilot.flatten(), Vt_tar_pilot.flatten(), dim=0
                    ).item()

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (2.0 * sigma_i - 1.0) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (2.0 * sigma_i - 1.0) * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware_Cosine(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.9,
):
    """
    VAGS (Velocity-Adaptive Guidance Scale) FlowEdit for SD3 — cosine variant.

    Replaces the fixed target CFG scale with a step-dependent adaptive scale
    modulated by the cosine similarity between source and target guided velocities.
    kappa_src / kappa_tar control the modulation strength for each scale.
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                # ONE forward pass: get raw uncond/cond for both src and tar
                latent_batch = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                timestep_exp = t.expand(latent_batch.shape[0])
                with torch.no_grad():
                    raw = pipe.transformer(
                        hidden_states=latent_batch,
                        timestep=timestep_exp,
                        encoder_hidden_states=src_tar_prompt_embeds,
                        pooled_projections=src_tar_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                u_src, p_src, u_tar, p_tar = raw.chunk(4)

                sigma_i = 1.0 - t_i.item() if isinstance(t_i, torch.Tensor) else 1.0 - t_i

                # Pilot velocities — source: constant λ_src; target: temporal-only λ^{tar,base} (paper Eq. pilot)
                lambda_tar_base_i = tar_guidance_scale * math.exp(kappa_tar * (2.0 * sigma_i - 1.0))
                Vt_src_pilot = u_src + src_guidance_scale * (p_src - u_src)
                Vt_tar_pilot = u_tar + lambda_tar_base_i * (p_tar - u_tar)

                s_i = torch.nn.functional.cosine_similarity(
                    Vt_src_pilot.flatten(), Vt_tar_pilot.flatten(), dim=0
                ).item()

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (2.0 * sigma_i - 1.0) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (2.0 * sigma_i - 1.0) * s_i
                )

                # Final velocities — reuse raw outputs, no second model call
                Vt_src = u_src + lambda_src_i * (p_src - u_src)
                Vt_tar = u_tar + lambda_tar_i * (p_tar - u_tar)

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware_SigmaType(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.9,
    sigma_type: int = 1,
    softness_m: float = 1.0,
):
    """
    Adaptive Guidance FlowEdit for SD3.

    In addition to the standard FlowEdit parameters, this variant introduces:
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant that controls the saturation of the
                      conflict signal  s_i = Δ_i / (Δ_i + m)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot).item()

                if sigma_type == 1:
                    sigma_i = 0
                    noise_signal = (2.0 * sigma_i - 1.0)
                elif sigma_type == 2:
                    sigma_i = 1
                    noise_signal = (2.0 * sigma_i - 1.0)
                elif sigma_type == 3:
                    sigma_i = 1.0 - t_i.item() if isinstance(t_i, torch.Tensor) else 1.0 - t_i
                    noise_signal = -1*(2.0 * sigma_i - 1.0)

                s_i = torch.nn.functional.cosine_similarity(
                        Vt_src_pilot.flatten(), Vt_tar_pilot.flatten(), dim=0
                    ).item()

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * noise_signal * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * noise_signal * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware_Sigma(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.7,
    sigma: float = 0,
    softness_m: float = 1.0,
):
    """
    Adaptive Guidance FlowEdit for SD3.

    In addition to the standard FlowEdit parameters, this variant introduces:
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant that controls the saturation of the
                      conflict signal  s_i = Δ_i / (Δ_i + m)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot).item()

                sigma_i = sigma
                s_i = delta_i / (delta_i + softness_m)

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (2.0 * sigma_i - 1.0) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (2.0 * sigma_i - 1.0) * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_ConflictAware_Sigma_Reverse(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.7,
    sigma: float = 0,
    softness_m: float = 1.0,
):
    """
    Adaptive Guidance FlowEdit for SD3.

    In addition to the standard FlowEdit parameters, this variant introduces:
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant that controls the saturation of the
                      conflict signal  s_i = Δ_i / (Δ_i + m)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot).item()

                sigma_i = sigma
                s_i = delta_i / (delta_i + softness_m)

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (1.0 - 2.0 * sigma_i) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (1.0 - 2.0 * sigma_i) * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

def optimized_scale(v_cond, v_uncond):
    """
    Calculates the optimized scale s* based on the projection of v_cond onto v_uncond.
    Formula: s* = (v_cond^T * v_uncond) / ||v_uncond||^2

    This ensures the unconditional velocity is properly scaled before
    computing the guidance direction, preventing misaligned baselines.
    """
    v_cond_flat   = v_cond.view(v_cond.shape[0], -1)
    v_uncond_flat = v_uncond.view(v_uncond.shape[0], -1)

    dot_product  = torch.sum(v_cond_flat * v_uncond_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(v_uncond_flat ** 2, dim=1, keepdim=True) + 1e-8

    s_star = dot_product / squared_norm

    return s_star.view(v_cond.shape[0], 1, 1, 1)

def calc_v_sd3_cfg_zero(
    pipe,
    model_input,
    prompt_embeds,
    pooled_prompt_embeds,
    guidance_scale,
    t,
    step_index,
    zero_init_steps=1,
    use_optimized_scale=True
):
    """
    Run the SD3 transformer and apply CFG-Zero* guidance.

    CFG-Zero* improves standard classifier-free guidance by:
      1. Zero-Init: Returning zero velocity for the first K steps to avoid
         early denoising divergence caused by a misaligned unconditional prediction.
      2. Optimized Scale: Projecting v_uncond onto v_cond to find the best
         rescaling factor s*, preventing over-subtraction of the baseline.

    Args:
        pipe:                  The loaded StableDiffusion3Pipeline.
        model_input:           Concatenated latents [uncond_input, cond_input].
        prompt_embeds:         Concatenated text embeddings [uncond_embeds, cond_embeds].
        pooled_prompt_embeds:  Concatenated pooled embeddings [uncond_pooled, cond_pooled].
        guidance_scale:        CFG guidance scale (w in the paper).
        t:                     Current timestep tensor.
        step_index:            Current integer step index (0-based).
        zero_init_steps:       Number of steps to apply zero-init (K).
        use_optimized_scale:   Whether to use s* rescaling.

    Returns:
        v_pred: Guided velocity prediction with the same shape as a single
                (non-batched-pair) latent.
    """
    if step_index < zero_init_steps:
        half_batch = model_input.shape[0] // 2
        return torch.zeros_like(model_input[:half_batch])

    timestep = t.expand(model_input.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    v_uncond, v_cond = noise_pred.chunk(2)
    del noise_pred

    if use_optimized_scale:
        s_star = optimized_scale(v_cond, v_uncond)

        v_pred = v_uncond * s_star + guidance_scale * (v_cond - v_uncond * s_star)
        del s_star
    else:
        v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

    del v_uncond, v_cond
    return v_pred

@torch.no_grad()
def FlowEditSD3_CFGZero(
    pipe,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 33,
    zero_init_steps: int = 2,
    use_optimized_scale: bool = True
):
    """
    Flow-based image editing for Stable Diffusion 3.5 with CFG-Zero* guidance.

    The editing follows the FlowEdit paradigm:
      - The source trajectory is reconstructed at each step via linear interpolation
        in flow-matching space: z_t = (1 - t) * x0 + t * noise
      - The target trajectory is offset from the source trajectory by the
        difference between the edited latent and the original source latent.
      - The net velocity delta (v_tar - v_src) is integrated via an Euler step.

    Args:
        pipe:               Loaded StableDiffusion3Pipeline.
        x_src:              Encoded source image latent, shape [1, C, H, W].
        src_prompt:         Text prompt describing the source image.
        tar_prompt:         Text prompt describing the desired edited image.
        negative_prompt:    Negative prompt for CFG guidance.
        T_steps:            Total number of denoising steps.
        n_avg:              Number of noise samples to average over per step.
        src_guidance_scale: CFG scale for the source branch.
        tar_guidance_scale: CFG scale for the target branch.
        n_min:              Minimum step index (from end) to apply editing.
        n_max:              Maximum step index (from end) to apply editing.
        zero_init_steps:    Number of initial steps using CFG-Zero* zero-init.
        use_optimized_scale: Whether to use the s* optimized scale in CFG-Zero*.

    Returns:
        zt_edit: Edited latent tensor, same shape as x_src.
    """
    device = x_src.device

    pipe.scheduler.set_timesteps(T_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    src_embeds_pair = torch.cat([src_negative_prompt_embeds, src_prompt_embeds])
    src_pooled_pair = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds])
    tar_embeds_pair = torch.cat([tar_negative_prompt_embeds, tar_prompt_embeds])
    tar_pooled_pair = torch.cat([tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds])

    del src_prompt_embeds, src_negative_prompt_embeds
    del src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds
    del tar_prompt_embeds, tar_negative_prompt_embeds
    del tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds
    torch.cuda.empty_cache()

    src_embeds_pair = src_embeds_pair.cpu()
    src_pooled_pair = src_pooled_pair.cpu()
    tar_embeds_pair = tar_embeds_pair.cpu()
    tar_pooled_pair = tar_pooled_pair.cpu()
    torch.cuda.empty_cache()

    zt_edit = x_src.clone()

    active_step = 0

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit"):
        step_from_end = T_steps - i
        if step_from_end > n_max or step_from_end < n_min:
            continue

        t_curr = (t / 1000.0).float()

        if i + 1 < len(timesteps):
            t_prev = (timesteps[i + 1] / 1000.0).float()
        else:
            t_prev = torch.tensor(0.0, device=device)

        dt = t_prev - t_curr

        src_embeds_step = src_embeds_pair.to(device)
        src_pooled_step = src_pooled_pair.to(device)
        tar_embeds_step = tar_embeds_pair.to(device)
        tar_pooled_step = tar_pooled_pair.to(device)

        v_delta_avg = torch.zeros_like(x_src)

        for _ in range(n_avg):
            noise = torch.randn_like(x_src)

            zt_src = (1 - t_curr) * x_src + t_curr * noise

            zt_tar = zt_edit + zt_src - x_src

            del noise

            model_input_src = torch.cat([zt_src, zt_src])
            del zt_src
            vt_src = calc_v_sd3_cfg_zero(
                pipe, model_input_src, src_embeds_step, src_pooled_step,
                src_guidance_scale, t, step_index=active_step,
                zero_init_steps=zero_init_steps,
                use_optimized_scale=use_optimized_scale
            )
            del model_input_src

            model_input_tar = torch.cat([zt_tar, zt_tar])
            del zt_tar
            vt_tar = calc_v_sd3_cfg_zero(
                pipe, model_input_tar, tar_embeds_step, tar_pooled_step,
                tar_guidance_scale, t, step_index=active_step,
                zero_init_steps=zero_init_steps,
                use_optimized_scale=use_optimized_scale
            )
            del model_input_tar

            v_delta_avg += (vt_tar - vt_src) / n_avg
            del vt_src, vt_tar

        del src_embeds_step, src_pooled_step, tar_embeds_step, tar_pooled_step
        torch.cuda.empty_cache()

        zt_edit = zt_edit + v_delta_avg * dt
        del v_delta_avg

        active_step += 1

    return zt_edit

@torch.no_grad()
def FlowEditSD3_DirectionalConflict(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    kappa_src: float = 1.0,
    kappa_tar: float = 1.0,
    softness_m: float = 1.0,
    epsilon: float = 1e-8,
):
    """
    Directional Conflict FlowEdit for SD3.

    Extends FlowEditSD3 with timestep-adaptive guidance scales that are
    modulated by both the *magnitude* and *directional disagreement* between
    the source and target velocity predictions.

    Additional parameters (beyond standard FlowEditSD3):
        kappa_src   – modulation strength for the source guidance scale
        kappa_tar   – modulation strength for the target guidance scale
        softness_m  – softness constant controlling saturation of the
                      magnitude gate  g_i = Δ_i / (Δ_i + m)
        epsilon     – small constant for numerical stability in the
                      cosine-similarity denominator
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_pilot = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src_pilot, Vt_tar_pilot = calc_v_sd3(
                    pipe, src_tar_latent_pilot,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                )

                sigma_i = (1.0 - t_i.item()) if isinstance(t_i, torch.Tensor) else (1.0 - t_i)

                delta_i = torch.norm(Vt_tar_pilot - Vt_src_pilot)
                g_i = delta_i / (delta_i + softness_m)

                cos_sim = (
                    torch.sum(Vt_tar_pilot * Vt_src_pilot)
                    / (torch.norm(Vt_tar_pilot) * torch.norm(Vt_src_pilot) + epsilon)
                )
                c_i = 0.5 * (1.0 - cos_sim)

                s_i = (c_i * g_i).item()

                lambda_src_i = src_guidance_scale * math.exp(
                    kappa_src * (2.0 * sigma_i - 1.0) * s_i
                )
                lambda_tar_i = tar_guidance_scale * math.exp(
                    kappa_tar * (2.0 * sigma_i - 1.0) * s_i
                )

                src_tar_latent_adaptive = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_adaptive,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    lambda_src_i, lambda_tar_i, t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditSD3_Monotonic(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale_lo: float = 1.0,
    src_guidance_scale_hi: float = 3.5,
    tar_guidance_scale_hi: float = 13.5,
    tar_guidance_scale_lo: float = 1.5,
    alpha: float = 0.0,
    beta: float = 0.0,
    n_min: int = 0,
    n_max: int = 15,
):
    """
    FlowEdit with Monotonic Schedule-Based Adaptive Guidance (SD3 variant).

    Instead of fixed src/tar guidance scales, this method computes per-step
    adaptive scales:
        tau_i       = t_i / t_{n_max}          (normalized timestep)
        omega_tar^i = omega_tar_lo + (omega_tar_hi - omega_tar_lo) * tau_i^alpha   (decreasing)
        omega_src^i = omega_src_lo + (omega_src_hi - omega_src_lo) * (1 - tau_i)^beta  (increasing)
    """

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None
    )
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale_hi
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale_hi
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )

    idx_nmax = T_steps - n_max
    t_nmax = timesteps[idx_nmax].float() / 1000.0

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t.float() / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1].float() / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            tau_i = t_i / t_nmax
            omega_tar_i = (
                tar_guidance_scale_lo
                + (tar_guidance_scale_hi - tar_guidance_scale_lo)
                * (tau_i ** alpha)
            )
            omega_src_i = (
                src_guidance_scale_lo
                + (src_guidance_scale_hi - src_guidance_scale_lo)
                * ((1.0 - tau_i) ** beta)
            )

            pipe._guidance_scale = max(float(omega_tar_i), float(omega_src_i))

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tar)
                )

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe,
                    src_tar_latent_model_input,
                    src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds,
                    float(omega_src_i),
                    float(omega_tar_i),
                    t,
                )

                V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = (
                torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
                if pipe.do_classifier_free_guidance
                else (xt_src, xt_tar)
            )

            _, Vt_tar = calc_v_sd3(
                pipe,
                src_tar_latent_model_input,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                tar_guidance_scale_hi,
                tar_guidance_scale_hi,
                t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

def calc_v_sd3_raw(
        pipe, src_tar_latent_model_input,
        src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, t):
    """
    Run a single batched forward pass and return the four raw velocity
    components instead of the already-guided combinations.

    Returns:
        V_uncond_src, V_cond_src, V_uncond_tar, V_cond_tar
    """
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        V_uncond_src, V_cond_src, V_uncond_tar, V_cond_tar = noise_pred_src_tar.chunk(4)

    return V_uncond_src, V_cond_src, V_uncond_tar, V_cond_tar

@torch.no_grad()
def FlowEditSD3_Magnitude(
        pipe,
        scheduler,
        x_src,
        src_prompt,
        tar_prompt,
        negative_prompt,
        T_steps: int = 50,
        n_avg: int = 1,
        src_guidance_scale_base: float = 3.5,
        tar_guidance_scale_base: float = 13.5,
        n_min: int = 0,
        n_max: int = 15,
        omega_tar_max: float = 15.0,
        omega_tar_min: float = 1.5,
        omega_src_max: float = 5.0,
        omega_src_min: float = 1.0,
        gamma: float = 10.0,
        r_0: float = 1.0,
        epsilon: float = 1e-6,):
    """
    FlowEdit with Velocity-Magnitude-Driven Adaptive Guidance for SD3.

    Instead of using fixed guidance scales throughout, this method computes
    a velocity-magnitude ratio between the target and source guided
    velocities (at base scales) at each timestep, then adaptively adjusts
    both source and target guidance via a sigmoid modulation.

    New parameters (compared to FlowEditSD3):
        src_guidance_scale_base / tar_guidance_scale_base:
            Base guidance scales used to compute the velocity ratio.
        omega_tar_max, omega_tar_min:
            Upper/lower bounds for the adaptive target guidance scale.
        omega_src_max, omega_src_min:
            Upper/lower bounds for the adaptive source guidance scale.
        gamma:  Sigmoid steepness controlling how sharply scales transition.
        r_0:    Ratio midpoint around which the sigmoid is centred.
        epsilon: Small constant for numerical stability in ratio computation.
    """

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale_base
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale_base
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = (timesteps[i + 1]) / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat(
                    [zt_src, zt_src, zt_tar, zt_tar]
                ) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)

                V_uncond_src, V_cond_src, V_uncond_tar, V_cond_tar = \
                    calc_v_sd3_raw(
                        pipe, src_tar_latent_model_input,
                        src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, t)

                Vt_src_base = V_uncond_src + src_guidance_scale_base * (V_cond_src - V_uncond_src)
                Vt_tar_base = V_uncond_tar + tar_guidance_scale_base * (V_cond_tar - V_uncond_tar)

                r_i = (torch.norm(Vt_tar_base) /
                       (torch.norm(Vt_src_base) + epsilon))

                omega_tar_i = (omega_tar_min
                               + (omega_tar_max - omega_tar_min)
                               * torch.sigmoid(gamma * (r_i - r_0)))
                omega_src_i = (omega_src_min
                               + (omega_src_max - omega_src_min)
                               * torch.sigmoid(gamma * (r_0 - r_i)))

                Vt_src_adaptive = V_uncond_src + omega_src_i * (V_cond_src - V_uncond_src)
                Vt_tar_adaptive = V_uncond_tar + omega_tar_i * (V_cond_tar - V_uncond_tar)

                V_delta = Vt_tar_adaptive - Vt_src_adaptive
                V_delta_avg += (1 / n_avg) * V_delta

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat(
                [xt_tar, xt_tar, xt_tar, xt_tar]
            ) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                tar_guidance_scale_base, tar_guidance_scale_base, t)

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditFLUX(pipe,
                 scheduler,
                 x_src,
                 src_prompt,
                 tar_prompt,
                 negative_prompt,
                 T_steps: int = 28,
                 n_avg: int = 1,
                 src_guidance_scale: float = 1.5,
                 tar_guidance_scale: float = 5.5,
                 n_min: int = 0,
                 n_max: int = 24,):

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size=x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None, latents=x_src)
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents,
        x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )

    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):

                fwd_noise = torch.randn_like(
                    x_src_packed).to(x_src_packed.device)

                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = calc_v_flux(pipe,
                                     latents=zt_src,
                                     prompt_embeds=src_prompt_embeds,
                                     pooled_prompt_embeds=src_pooled_prompt_embeds,
                                     guidance=src_guidance,
                                     text_ids=src_text_ids,
                                     latent_image_ids=latent_src_image_ids,
                                     t=t)

                Vt_tar = calc_v_flux(pipe,
                                     latents=zt_tar,
                                     prompt_embeds=tar_prompt_embeds,
                                     pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                     guidance=tar_guidance,
                                     text_ids=tar_text_ids,
                                     latent_image_ids=latent_tar_image_ids,
                                     t=t)

                V_delta_avg += (1/n_avg)*(Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:

            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(
                    x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(
                    scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux(pipe,
                                 latents=xt_tar,
                                 prompt_embeds=tar_prompt_embeds,
                                 pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                 guidance=tar_guidance,
                                 text_ids=tar_text_ids,
                                 latent_image_ids=latent_tar_image_ids,
                                 t=t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    torch.cuda.empty_cache()
    gc.collect()
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(
        out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out

@torch.no_grad()
def FlowEditSD3_VAGS_Cosine_Monotone(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 33,
    kappa_src: float = 0.0,
    kappa_tar: float = 0.9,
    softness_m: float = 1e-8,
):
    """
    VAGS-Cosine x Monotone: combines a cosine-similarity conflict signal
    with a monotone-increasing guidance envelope.

    The base target scale at active step k (out of total_active steps) is:
        tau = k / max(total_active - 1, 1)       (0=noisiest, 1=cleanest)
        mono_scale = tar_guidance_scale * (0.5 + 0.25 * (1 - cos(pi*tau)))
    then modulated by the cosine-similarity conflict signal s_i:
        lambda_tar = mono_scale * exp(kappa_tar * (2*sigma_i - 1) * s_i)
    """
    import math as _math

    device = x_src.device
    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (src_prompt_embeds, src_negative_prompt_embeds,
     src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)

    pipe._guidance_scale = tar_guidance_scale
    (tar_prompt_embeds, tar_negative_prompt_embeds,
     tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    total_active = sum(1 for i in range(T_steps) if (T_steps - i) <= n_max and (T_steps - i) > n_min)
    active_step = 0

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        torch.cuda.empty_cache()
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        t_im1 = timesteps[i + 1] / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            tau = active_step / max(total_active - 1, 1)
            mono_scale = tar_guidance_scale * (0.5 + 0.25 * (1.0 - _math.cos(_math.pi * tau)))
            active_step += 1

            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                # ONE forward pass: get raw uncond/cond for both src and tar
                latent_batch = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                timestep_exp = t.expand(latent_batch.shape[0])
                with torch.no_grad():
                    raw = pipe.transformer(
                        hidden_states=latent_batch,
                        timestep=timestep_exp,
                        encoder_hidden_states=src_tar_prompt_embeds,
                        pooled_projections=src_tar_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                u_src, p_src, u_tar, p_tar = raw.chunk(4)

                sigma_i = 1.0 - t_i.item() if isinstance(t_i, torch.Tensor) else 1.0 - t_i

                # Pilot — source: constant λ_src; target: mono_scale (temporal-only base)
                Vt_src_pilot = u_src + src_guidance_scale * (p_src - u_src)
                Vt_tar_pilot = u_tar + mono_scale * (p_tar - u_tar)

                s_i = torch.nn.functional.cosine_similarity(
                    Vt_src_pilot.flatten(), Vt_tar_pilot.flatten(), dim=0).item()

                lambda_src_i = src_guidance_scale * _math.exp(kappa_src * (2.0 * sigma_i - 1.0) * s_i)
                lambda_tar_i = mono_scale * _math.exp(kappa_tar * (2.0 * sigma_i - 1.0) * s_i)

                # Final velocities — reuse raw outputs, no second model call
                Vt_src = u_src + lambda_src_i * (p_src - u_src)
                Vt_tar = u_tar + lambda_tar_i * (p_tar - u_tar)

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)
        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)
            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds,
                                   src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)
            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            xt_tar = prev_sample.to(Vt_tar.dtype)

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

