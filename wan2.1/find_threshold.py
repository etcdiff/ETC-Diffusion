import torch
import inspect
import time
import numpy as np
import os
import sys
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool
import argparse
import random
import math
import gc
from contextlib import contextmanager
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torchvision
import ruptures as rpt

def get_video_ssim(origin_video,video_bias):
    ssim_list = []
    for i in range(len(origin_video)):
        ssim_list.append(ssim(origin_video[i], video_bias[i], channel_axis=2))
    return sum(ssim_list)/len(ssim_list)

def to_video(origin_video):
    origin_video = origin_video[None].clamp(-1, 1)
    origin_video = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=1, normalize=True, value_range=(-1, 1))
        for u in origin_video.unbind(2)
    ],dim=1).permute(1, 2, 3, 0)
    origin_video = (origin_video * 255).type(torch.uint8).cpu()
    return origin_video

def _validate_args(args):
    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args,_ = parser.parse_known_args() #.parse_args()

    _validate_args(args)

    return args

@torch.no_grad()
def return_noise(
    self,
    input_prompt,
    size=(1280, 720),
    frame_num=81,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True
):
    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}
        
        noise_list = []
        with tqdm(total=len(timesteps)) as progress_bar:
            for i,t in enumerate(timesteps):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                noise_list.append(noise_pred)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                progress_bar.update()
        last_latent = latents 

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0],noise_list,last_latent if self.rank == 0 else None


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

#get each prompt noise
args = _parse_args()
args.task = 't2v-14B'
args.size = "832*480"
args.base_seed = 42
args.ckpt_dir = './ckpt/Wan2.1-T2V-14B'
cfg = WAN_CONFIGS[args.task]
if "t2v" in args.task:
    #loading model
    pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0
    )
elif "i2v" in args.task:
    #loading model
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0
    )

#start to find threshold
chazhi = None
sim = None
for prompt in prompts_set:
    origin_video,noise_list,latents = return_noise(
        self=pipe,
        input_prompt=prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model
    )
    origin_video = to_video(origin_video)
    cha = [(noise_list[i+1].detach().to(torch.float) - noise_list[i].detach().to(torch.float)).abs().mean().cpu().numpy() for i in range(len(noise_list)-1)]
    if chazhi is None:
        chazhi = cha
    else:
        for i in range(len(cha)):
            chazhi[i]+=cha[i]
    ssim_list = []
    for c in cha:
        bias = torch.randn_like(latents[0])
        bias = bias / bias.abs().mean() * c.item()
        latents_bias = [latents[0] + bias]
        #decode video
        video_bias = pipe.vae.decode(latents_bias)[0]
        video_bias = to_video(video_bias)
        score = get_video_ssim(origin_video.numpy(), video_bias.numpy())
        ssim_list.append(score)
    if sim is None:
        sim = ssim_list
    else:
        for i in range(len(ssim_list)):
            sim[i]+=ssim_list[i]
chazhi = np.array(chazhi) / len(prompts_set)
sim = np.array(sim) / len(prompts_set)

'''
[0.48879606 0.4973958  0.4225485  0.2762299  0.21026638 0.15230505
 0.11810397 0.10747004 0.08690636 0.0761179  0.06785388 0.06412537
 0.05850638 0.05871553 0.05447624 0.05234103 0.05166012 0.04992264
 0.04872459 0.04689917 0.0459352  0.04502122 0.04394585 0.04398945
 0.04258331 0.04170249 0.04093243 0.04111382 0.04030636 0.04093097
 0.04075148 0.04050629 0.04030353 0.0409973  0.04134507 0.04176274
 0.04300997 0.04321859 0.04409026 0.04487447 0.04716801 0.04862734
 0.05102556 0.0540616  0.05861389 0.06440091 0.07458542 0.09589475
 0.16395251]
[0.43777105 0.43500277 0.47631479 0.61722994 0.68898177 0.76483378
 0.82581934 0.84244701 0.87486163 0.89481261 0.91059045 0.9166236
 0.92735611 0.92663733 0.9341355  0.93689584 0.9379211  0.94158247
 0.9444389  0.94744326 0.94872581 0.95056181 0.95201707 0.95228082
 0.95446496 0.9561622  0.95720964 0.95709532 0.95826066 0.95719826
 0.95748345 0.95757949 0.95803029 0.95688193 0.95640387 0.95544633
 0.95364072 0.95349714 0.95202457 0.95092659 0.94726456 0.94489986
 0.94084392 0.93607803 0.92883599 0.91926935 0.90240046 0.86551519
 0.74595061]
'''
result = rpt.KernelCPD(kernel="rbf").fit(sim).predict(n_bkps=4)[:-1]
print('Final Threshold: ',chazhi[result].max()) 