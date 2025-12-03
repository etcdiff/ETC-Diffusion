import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import time
import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import USE_PEFT_BACKEND,is_torch_xla_available,logging,scale_lora_layers,unscale_lora_layers
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class ETC():
    def __init__(self, model, p=8, threshold=0.75, alpha=0.5):
        self.model = model
        self.p = p #model pre-inference step, in paper we use n
        self.threshold = threshold
        self.alpha = alpha
        self.k = 0 #approximation step
        self.gradient = None
        self.pre_noise = None
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.model.default_sample_size * self.model.vae_scale_factor
        width = width or self.model.default_sample_size * self.model.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.model.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self.model._guidance_scale = guidance_scale
        self.model._joint_attention_kwargs = joint_attention_kwargs
        self.model._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device

        lora_scale = (
            self.model.joint_attention_kwargs.get("scale", None) if self.model.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.model.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.model.transformer.config.in_channels // 4
        latents, latent_image_ids = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.model.scheduler.config.base_image_seq_len,
            self.model.scheduler.config.max_image_seq_len,
            self.model.scheduler.config.base_shift,
            self.model.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        self.model._num_timesteps = len(timesteps)

        # handle guidance
        if self.model.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.model.joint_attention_kwargs is None:
            self.model._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.model.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.model.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        current_step = 0
        trend = None
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            if image_embeds is not None:
                self.model._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
            while current_step<len(timesteps):
                if current_step < self.p or current_step==(len(timesteps)-1):
                    t = timesteps[current_step]
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.model.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.model.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self.model._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                        neg_noise_pred = self.model.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.model.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

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
                            timestep = t.expand(latents.shape[0]).to(latents.dtype)

                            noise_pred = self.model.transformer(
                                hidden_states=latents,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self.model.joint_attention_kwargs,
                                return_dict=False,
                            )[0]

                            if do_true_cfg:
                                if negative_image_embeds is not None:
                                    self.model._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                                neg_noise_pred = self.model.transformer(
                                    hidden_states=latents,
                                    timestep=timestep / 1000,
                                    guidance=guidance,
                                    pooled_projections=negative_pooled_prompt_embeds,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    joint_attention_kwargs=self.model.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                            
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
                        
                        # compute the previous noisy sample x_t -> x_t-1
                        latents_dtype = latents.dtype
                        latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        if latents.dtype != latents_dtype:
                            if torch.backends.mps.is_available():
                                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                latents = latents.to(latents_dtype)
                        
                        current_step+=1
                        progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = self.model._unpack_latents(latents, height, width, self.model.vae_scale_factor)
            latents = (latents / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
            image = self.model.vae.decode(latents, return_dict=False)[0]
            image = self.model.image_processor.postprocess(image, output_type=output_type)
        # Offload all models
        self.model.maybe_free_model_hooks()
        self.k=0
        self.gradient = None
        self.pre_noise = None

        if not return_dict:
            return (image,)


        return FluxPipelineOutput(images=image)

if __name__ == "__main__":
    model = FluxPipeline.from_pretrained("./ckpt/FLUX.1-dev", torch_dtype=torch.bfloat16)
    model.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    pipe = ETC(model=model, p=6, threshold=0.1269)
    
    start = time.time()
    prompt = "A cat holding a sign that says hello world"
    num_inference_steps = 50
    image = pipe(prompt,height=1024,width=1024,guidance_scale=3.5,num_inference_steps=num_inference_steps, 
                max_sequence_length=512,generator=generator).images[0]
    end= time.time()
    image.save(f"ETC-{end-start}.png")



