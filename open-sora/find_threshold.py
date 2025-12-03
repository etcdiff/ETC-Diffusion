import os
import time
import torch
from mmengine.runner import set_random_seed
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
import sys
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
from opensora.schedulers.rf.rectified_flow import RFlowScheduler, timestep_transform
import numpy as np
import ruptures as rpt

def get_video_ssim(origin,bias):
    ssim_list = []
    for i in range(len(origin_video)):
        ssim_list.append(ssim(origin_video[i].numpy(), video_bias[i].numpy(), channel_axis=2))
    return sum(ssim_list)/len(ssim_list)

@torch.no_grad()
def return_noise(
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
        guidance_scale = self.cfg_scale

    n = len(prompts)
    # text encoding
    model_args = text_encoder.encode(prompts)
    y_null = text_encoder.null(n)
    model_args["y"] = torch.cat([model_args["y"], y_null], 0)
    if additional_args is not None:
        model_args.update(additional_args)

    # prepare timesteps
    timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
    if self.use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    if self.use_timestep_transform:
        timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

    if mask is not None:
        noise_added = torch.zeros_like(mask, dtype=torch.bool)
        noise_added = noise_added | (mask == 1)
    
    noise_list = []
    for i, t in tqdm(enumerate(timesteps)):
        # mask for adding noise
        if mask is not None:
            mask_t = mask * self.num_timesteps
            x0 = z.clone()
            x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

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
        
        noise_list.append(v_pred)

        # update z
        dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
        dt = dt / self.num_timesteps
        z = z + v_pred * dt[:, None, None, None, None]

        if mask is not None:
            z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

    return noise_list,z

#get different length prompt
prompts_set = []
data_path = './t2v/data/T2V-CompBench-prompts'
for tx in os.listdir(data_path):
    with open(os.path.join(data_path,tx), 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    caption_list = [prompt.strip() for prompt in prompts]
    prompt_lengths = [len(prompt) for prompt in caption_list]
    plt.cla()
    histplot = sns.histplot(prompt_lengths, kde=True, bins=3, color='blue', edgecolor='black')
    bin_edges = [int(patch.get_x()) for patch in histplot.patches]
    intervals = list(zip(bin_edges[:-1], bin_edges[1:]))
    for start, end in intervals:
        selected_prompts = [caption_list[i] for i in range(len(caption_list)) if start <= prompt_lengths[i] < end]
        if len(selected_prompts)>0:
            prompts_set.append(selected_prompts[len(selected_prompts)//2])


#load model
torch.set_grad_enabled(False)
cfg = parse_configs(training=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_dtype = cfg.get("dtype", "fp32")
assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
set_random_seed(seed=cfg.get("seed", 1024))
verbose = cfg.get("verbose", 1)
image_size = cfg.get("image_size", None)
if image_size is None:
    resolution = cfg.get("resolution", None)
    aspect_ratio = cfg.get("aspect_ratio", None)
    assert (
        resolution is not None and aspect_ratio is not None
    ), "resolution and aspect_ratio must be provided if image_size is not provided"
    image_size = get_image_size(resolution, aspect_ratio)
num_frames = get_num_frames(cfg.num_frames)

text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
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
scheduler = build_module(cfg.scheduler, SCHEDULERS)

fps = cfg.fps
save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
multi_resolution = cfg.get("multi_resolution", None)
batch_size = cfg.get("batch_size", 1)
num_sample = cfg.get("num_sample", 1)
loop = cfg.get("loop", 1)
condition_frame_length = cfg.get("condition_frame_length", 5)
condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
align = cfg.get("align", None)

#start to find threshold
chazhi = None
sim = None
for prompt in prompts_set:
    prompts = [prompt]

    model_args = prepare_multi_resolution_info(
        multi_resolution, len(prompts), image_size, num_frames, fps, device, dtype
    )

    # == sampling ==
    z = torch.randn(len(prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
    noise_list,latents = return_noise(
        self=scheduler,
        model = model,
        text_encoder = text_encoder,
        z=z,
        prompts=prompts,
        device=device,
        additional_args=model_args
    )
    origin_video = vae.decode(latents.to(dtype), num_frames=num_frames)[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8) #[102, 720, 1280, 3]
    
    cha = [(noise_list[i+1].detach().to(torch.float) - noise_list[i].detach().to(torch.float)).abs().mean().cpu().numpy() for i in range(len(noise_list)-1)]
    if chazhi is None:
        chazhi = cha
    else:
        for i in range(len(cha)):
            chazhi[i]+=cha[i]
    ssim_list = []
    for c in cha:
        bias = torch.randn_like(latents)
        bias = bias / bias.abs().mean() * c.item()
        latents_bias = latents + bias
        #decode video
        video_bias = vae.decode(latents_bias.to(dtype), num_frames=num_frames)[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8) #[102, 720, 1280, 3]
        
        score = get_video_ssim(origin_video, video_bias)
        ssim_list.append(score)
    if sim is None:
        sim = ssim_list
    else:
        for i in range(len(ssim_list)):
            sim[i]+=ssim_list[i]
chazhi = np.array(chazhi) / len(prompts_set)
sim = np.array(sim) / len(prompts_set)

'''chazhi = [0.8506895  0.43648037 0.23910406 0.21014808 0.16840968 0.12949552
 0.12958191 0.13334268 0.11480681 0.10974248 0.10369124 0.09566643
 0.09029839 0.09106562 0.0870579  0.0863518  0.0851834  0.08764504
 0.08657116 0.08659511 0.0867734  0.08864087 0.09225243 0.0953825
 0.10332327 0.11308245 0.13556004 0.18203184 0.31204957]
sim = [0.44650451 0.62349985 0.7750346  0.80889098 0.84670824 0.88891779
 0.88586834 0.88177121 0.90320517 0.90747488 0.91299006 0.92096215
 0.92651537 0.92623516 0.92970501 0.92994658 0.93132949 0.92956769
 0.93047215 0.93020834 0.92964974 0.92718594 0.92278635 0.91954144
 0.91044299 0.90013892 0.87673798 0.8313456  0.71094791]
'''
algo = rpt.BottomUp(model="l1").fit(sim)
result = algo.predict(n_bkps=1)[0]

print('Final Threshold: ',chazhi[result])
