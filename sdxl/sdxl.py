import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


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
    def __init__(self, model, p=6, threshold=0.017):
        self.model = model
        self.p = p #model pre-inference step, in paper we use n
        self.threshold = threshold
        self.k = 0 #approximation step
        self.gradient = None
        self.pre_noise = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.model.default_sample_size * self.model.vae_scale_factor
        width = width or self.model.default_sample_size * self.model.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.model.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds
        )

        self.model._guidance_scale = guidance_scale
        self.model._guidance_rescale = guidance_rescale
        self.model._clip_skip = clip_skip
        self.model._cross_attention_kwargs = cross_attention_kwargs
        self.model._denoising_end = denoising_end
        self.model._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.model.cross_attention_kwargs.get("scale", None) if self.model.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.model.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.model.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.model.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim

        add_time_ids = self.model._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.model._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.model.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.model.do_classifier_free_guidance,
            )

        # 8.1 Apply denoising_end
        if (
            self.model.denoising_end is not None
            and isinstance(self.model.denoising_end, float)
            and self.model.denoising_end > 0
            and self.model.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.model.scheduler.config.num_train_timesteps
                    - (self.model.denoising_end * self.model.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.model.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.model.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.model.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.model._num_timesteps = len(timesteps)
        #for accelerate
        current_step = 0
        ema = None
        #start denoising
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            while current_step<len(timesteps):
                if current_step < self.p or current_step==(len(timesteps)-1):
                    t = timesteps[current_step]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds
                    noise_pred = self.model.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.model.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.model.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.model.do_classifier_free_guidance and self.model.guidance_rescale > 0.0:
                        # Based on 3.4. in https://huggingface.co/papers/2305.08891
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.model.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    #accelerate module
                    if ema is None:
                        if current_step >0:
                            ema = noise_pred-self.pre_noise
                    else:
                        ema = (ema + noise_pred-self.pre_noise)/2

                    self.pre_noise = noise_pred
                    
                    current_step+=1
                    progress_bar.update()
                else:
                    for i in range(self.k+1):
                        if current_step >= len(timesteps)-1:
                            break
                        t = timesteps[current_step]
                        if i<self.k:
                            noise_pred = self.pre_noise + self.gradient
                        else:
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                            # predict the noise residual
                            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                                added_cond_kwargs["image_embeds"] = image_embeds
                            noise_pred = self.model.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.model.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]

                            # perform guidance
                            if self.model.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                            if self.model.do_classifier_free_guidance and self.model.guidance_rescale > 0.0:
                                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.model.guidance_rescale)

                            #accelerate module
                            ema = (ema + noise_pred-self.pre_noise)/2
                            if self.k!=0:
                                self.gradient = (1/self.k)*ema
                            else:
                                self.gradient = ema

                            #upadte k
                            if (noise_pred - self.pre_noise - ema).abs().mean().item() < self.threshold:
                                if self.k<5:
                                    self.k+=1
                            else:
                                if self.k>0:
                                    self.k-=1

                        self.pre_noise = noise_pred

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_dtype = latents.dtype
                        latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                        if latents.dtype != latents_dtype:
                            if torch.backends.mps.is_available():
                                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                latents = latents.to(latents_dtype)

                        current_step+=1
                        progress_bar.update()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.model.vae.dtype == torch.float16 and self.model.vae.config.force_upcast

            if needs_upcasting:
                self.model.upcast_vae()
                latents = latents.to(next(iter(self.model.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.model.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.model.vae = self.model.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.model.vae.config, "latents_mean") and self.model.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.model.vae.config, "latents_std") and self.model.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.model.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.model.vae.config.scaling_factor

            image = self.model.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.model.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            image = self.model.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.model.maybe_free_model_hooks()
        self.k=0
        self.gradient = None
        self.pre_noise = None

        if not return_dict:
            return (image,)


        return StableDiffusionXLPipelineOutput(images=image)

model = StableDiffusionXLPipeline.from_pretrained("./ckpt/stable-diffusion-xl-base", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe = ETC(model=model, p=6, threshold=0.0171)
start = time.time()
image = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator(device='cuda').manual_seed(42)
).images[0]
end = time.time()
image.save(f"ETC.png")
