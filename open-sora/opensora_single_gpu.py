import os
import time
import torch
from mmengine.runner import set_random_seed
from tqdm import tqdm
import argparse
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    extract_json_from_prompts,
    extract_prompts_loop,
    merge_prompt,
    prepare_multi_resolution_info,
    split_prompt,
)
from opensora.utils.misc import to_torch_dtype
from opensora.schedulers.rf.rectified_flow import timestep_transform

class ETC():
    def __init__(self, model, p=6, threshold=0.75, alpha=0.5):
        self.model = model
        self.p = p #model pre-inference step, in paper we use n
        self.threshold = threshold
        self.alpha = alpha
        self.k = 0 #approximation step
        self.gradient = None
        self.pre_noise = None

    @torch.no_grad()
    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.model.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.model.num_sampling_steps) * self.model.num_timesteps for i in range(self.model.num_sampling_steps)]
        if self.model.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.model.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.model.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        current_step = 0
        trend = None
        with tqdm(total=len(timesteps)) as progress_bar:
            while current_step<len(timesteps):
                if current_step < self.p or current_step==(len(timesteps)-1):
                    t = timesteps[current_step]
                    # mask for adding noise
                    if mask is not None:
                        mask_t = mask * self.model.num_timesteps
                        x0 = z.clone()
                        x_noise = self.model.scheduler.add_noise(x0, torch.randn_like(x0), t)

                        mask_t_upper = mask_t >= t.unsqueeze(1)
                        model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                        mask_add_noise = mask_t_upper & ~noise_added

                        z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                        noise_added = mask_t_upper

                    # classifier-free guidance
                    z_in = torch.cat([z, z], 0)
                    t = torch.cat([t, t], 0)
                    pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                    pred_cond, pred_uncond = pred.chunk(2, dim=0)
                    v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                    # update z
                    dt = timesteps[current_step] - timesteps[current_step + 1] if current_step < len(timesteps) - 1 else timesteps[current_step]
                    dt = dt / self.model.num_timesteps
                    z = z + v_pred * dt[:, None, None, None, None]

                    if mask is not None:
                        z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

                    #accelerate module
                    if trend is None:
                        if current_step >0:
                            trend = v_pred-self.pre_noise
                    else:
                        trend = (1-self.alpha)*trend + self.alpha*(v_pred-self.pre_noise)

                    self.pre_noise = v_pred
                    
                    current_step+=1
                    progress_bar.update()
                else: #using approximation noise and model adjustment
                    for i in range(self.k+1):
                        if current_step >= len(timesteps)-1:
                            break
                        t = timesteps[current_step]
                        # mask for adding noise
                        if mask is not None:
                            mask_t = mask * self.model.num_timesteps
                            x0 = z.clone()
                            x_noise = self.model.scheduler.add_noise(x0, torch.randn_like(x0), t)

                            mask_t_upper = mask_t >= t.unsqueeze(1)
                            model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                            mask_add_noise = mask_t_upper & ~noise_added

                            z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                            noise_added = mask_t_upper
                        if i<self.k:
                            v_pred = self.pre_noise + self.gradient
                        else:
                            # classifier-free guidance
                            z_in = torch.cat([z, z], 0)
                            t = torch.cat([t, t], 0)
                            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                            pred_cond, pred_uncond = pred.chunk(2, dim=0)
                            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                            #accelerate module
                            trend = (1-self.alpha)*trend + self.alpha*(v_pred-self.pre_noise)
                            if self.k!=0:
                                self.gradient = (1/self.k)*trend
                            else:
                                self.gradient = trend

                            #upadte k
                            if (v_pred - self.pre_noise - trend).abs().mean().item() < self.threshold:
                                self.k+=1
                            else:
                                if self.k>0:
                                    self.k-=1

                        self.pre_noise = v_pred

                        # update z
                        dt = timesteps[current_step] - timesteps[current_step + 1] if current_step < len(timesteps) - 1 else timesteps[current_step]
                        dt = dt / self.model.num_timesteps
                        z = z + v_pred * dt[:, None, None, None, None]

                        if mask is not None:
                            z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

                        current_step+=1
                        progress_bar.update()

        # Offload all models
        self.k=0
        self.gradient = None
        self.pre_noise = None

        return z

def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    #cfg = parse_configs(training=False)
    cfg = argparse.Namespace(
            resolution = "720p",
            aspect_ratio = "9:16",
            num_frames = '4s',
            fps = 24,
            frame_interval = 1,
            save_fps = 24,
            seed = 42,
            batch_size = 1,
            multi_resolution = "STDiT2",
            dtype = "bf16",
            condition_frame_length = 5,
            align = 5,
            model = dict(
                type="STDiT3-XL/2",
                from_pretrained="./ckpt/Open-Sora-v1.2/OpenSora-STDiT-v3",
                qk_norm=True,
                enable_flash_attn=True,
                enable_layernorm_kernel=True,
                ),
            vae = dict(
                type="OpenSoraVAE_V1_2",
                from_pretrained="./ckpt/Open-Sora-v1.2/OpenSora-VAE-v1.2",
                micro_frame_size=17,
                micro_batch_size=4,
                ),
            text_encoder = dict(
                type="t5",
                from_pretrained="./ckpt/Open-Sora-v1.2/t5-v1_1-xxl",
                model_max_length=300,
                ),
            scheduler = dict(
                type="rflow",
                use_timestep_transform=True,
                num_sampling_steps=30,
                cfg_scale=7.0,
                ),
            aes = 6.5,
            flow = None
        )
    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.dtype #cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.dtype) #cfg.get("dtype", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    set_random_seed(seed=cfg.seed) #cfg.get("seed", 1024)

    # ======================================================
    # build model & load weights
    # ======================================================
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = None #cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.resolution #cfg.get("resolution", None)
        aspect_ratio = cfg.aspect_ratio #cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=False,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS) #rf.RFLOW

    #kstep
    pipe = ETC(model=scheduler,p=4,threshold=0.16840968)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = ["a beautiful waterfall"]
    
    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.save_fps #cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.multi_resolution #cfg.get("multi_resolution", None)
    batch_size = cfg.batch_size #cfg.get("batch_size", 1)
    num_sample = 1 #cfg.get("num_sample", 1)
    loop = 1 #cfg.get("loop", 1)
    condition_frame_length = cfg.condition_frame_length #cfg.get("condition_frame_length", 5)
    condition_frame_edit = 0.0 #cfg.get("condition_frame_edit", 0.0)
    align = cfg.align #cfg.get("align", None)

    model_args = prepare_multi_resolution_info(
        multi_resolution, len(prompts), image_size, num_frames, fps, device, dtype
    )

    # == sampling ==
    z = torch.randn(len(prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
    '''samples = scheduler.sample(
        model,
        text_encoder,
        z=z,
        prompts=prompts,
        device=device,
        additional_args=model_args
    )'''
    start = time.time()
    samples = pipe.sample(
        model,
        text_encoder,
        z=z,
        prompts=prompts,
        device=device,
        additional_args=model_args
    )

    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
    end = time.time()
    print('use time: ',end-start)
    # == save samples ==
    save_path = save_sample(
        samples[0],
        fps=save_fps,
        save_path=f'ETC-{end-start}',
    )


if __name__ == "__main__":

    main()

