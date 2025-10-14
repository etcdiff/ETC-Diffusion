import gc
import math
import random
import sys
from contextlib import contextmanager
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import argparse
from PIL import Image
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import time

class ETC():
    def __init__(self, model, p=6, threshold=0.75, alpha=0.5):
        self.model = model
        self.p = p #model pre-inference step, in paper we use n
        self.threshold = threshold
        self.alpha=alpha
        self.k = 0 #approximation step
        self.gradient = None
        self.pre_noise = None

    @torch.no_grad()
    def generate(
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
        target_shape = (self.model.vae.model.z_dim, (F - 1) // self.model.vae_stride[0] + 1,
                        size[1] // self.model.vae_stride[1],
                        size[0] // self.model.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.model.patch_size[1] * self.model.patch_size[2]) *
                            target_shape[1] / self.model.sp_size) * self.model.sp_size

        if n_prompt == "":
            n_prompt = self.model.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.model.device)
        seed_g.manual_seed(seed)

        if not self.model.t5_cpu:
            self.model.text_encoder.model.to(self.model.device)
            context = self.model.text_encoder([input_prompt], self.model.device)
            context_null = self.model.text_encoder([n_prompt], self.model.device)
            if offload_model:
                self.model.text_encoder.model.cpu()
        else:
            context = self.model.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.model.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.model.device) for t in context]
            context_null = [t.to(self.model.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.model.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.model.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.model.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.model.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.model.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.model.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            
            current_step = 0
            trend = None           
            with tqdm(total=len(timesteps)) as progress_bar:
                while current_step<len(timesteps):
                    if current_step < self.p or current_step==(len(timesteps)-1):
                        t = timesteps[current_step]
                        latent_model_input = latents
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        self.model.model.to(self.model.device)
                        noise_pred_cond = self.model.model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = self.model.model(
                            latent_model_input, t=timestep, **arg_null)[0]

                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                        #accelerate module
                        if trend is None:
                            if current_step >0:
                                trend = noise_pred-self.pre_noise
                        else:
                            trend = (1-self.alpha)*trend + self.alpha*(noise_pred-self.pre_noise)

                        self.pre_noise = noise_pred
                        
                        current_step+=1
                        progress_bar.update()

                    else: #using approximation noise and model adjustment
                        for i in range(self.k+1):
                            if current_step >= len(timesteps)-1:
                                break
                            t = timesteps[current_step]
                            if i<self.k:
                                noise_pred = self.pre_noise + self.gradient
                            else:
                                latent_model_input = latents
                                timestep = [t]

                                timestep = torch.stack(timestep)

                                self.model.model.to(self.model.device)
                                noise_pred_cond = self.model.model(
                                    latent_model_input, t=timestep, **arg_c)[0]
                                noise_pred_uncond = self.model.model(
                                    latent_model_input, t=timestep, **arg_null)[0]

                                noise_pred = noise_pred_uncond + guide_scale * (
                                    noise_pred_cond - noise_pred_uncond)
                                
                                #accelerate module
                                trend = (1-self.alpha)*trend + self.alpha*(noise_pred-self.pre_noise)
                                #upadte k
                                if (noise_pred - self.pre_noise - trend).abs().mean().item() < self.threshold:
                                    self.k+=1
                                else:
                                    if self.k>0:
                                        self.k-=1
                                if self.k!=0:
                                    self.gradient = trend/self.k
                            
                            self.pre_noise = noise_pred

                            temp_x0 = sample_scheduler.step(
                                noise_pred.unsqueeze(0),
                                t,
                                latents[0].unsqueeze(0),
                                return_dict=False,
                                generator=seed_g)[0]
                            latents = [temp_x0.squeeze(0)]

                            current_step+=1
                            progress_bar.update()

            x0 = latents
            if offload_model:
                self.model.model.cpu()
                torch.cuda.empty_cache()
            if self.model.rank == 0:
                videos = self.model.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        # Offload all models
        self.k=0
        self.gradient = None
        self.pre_noise = None

        return videos[0] if self.model.rank == 0 else None

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

    args,_ = parser.parse_known_args() #parse_args()

    _validate_args(args)

    return args


def get_model(task,size,base_seed,ckpt_dir,device,threshold=0.1181):
    args = _parse_args()
    args.task = task
    args.size = size
    args.base_seed = base_seed
    args.ckpt_dir = ckpt_dir

    cfg = WAN_CONFIGS[args.task]
    
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device
    )

    model = ETC(model=wan_t2v,p=6,threshold=threshold)
    
    return model,args,cfg


model,args,cfg = get_model(task='t2v-14B',size="832*480",base_seed=42,ckpt_dir='./ckpt/Wan2.1-T2V-14B',device=torch.cuda.current_device(),threshold=0.1181)

prompt = 'a yellow bicycle'

start = time.time()

video = model.generate(
    prompt,
    size=SIZE_CONFIGS[args.size],
    frame_num=args.frame_num,
    shift=args.sample_shift,
    sample_solver=args.sample_solver,
    sampling_steps=args.sample_steps,
    guide_scale=args.sample_guide_scale,
    seed=args.base_seed,
    offload_model=args.offload_model
)
end = time.time()
cache_video(
    tensor=video[None],
    save_file=f'ETC-{end-start}.mp4',
    fps=cfg.sample_fps,
    nrow=1,
    normalize=True,
    value_range=(-1, 1)) 


print(end-start)

