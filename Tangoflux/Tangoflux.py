import torchaudio
from tangoflux import TangoFluxInference
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List
import inspect
import sys
import time

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):

    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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
    def __init__(self, model, p=8, threshold=0.75):
        self.model = model
        self.p = p #model pre-inference step, in paper we use n
        self.threshold = threshold
        self.k = 0 #approximation step
        self.gradient = None
        self.pre_noise = None
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=4.5,
        duration=10,
        seed=0,
        disable_progress=False,
        num_samples_per_prompt=1
    ):
        """Only tested for single inference. Haven't test for batch inference"""
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        bsz = num_samples_per_prompt
        device = self.model.model.transformer.device
        scheduler = self.model.model.noise_scheduler

        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor([duration], device=device)
        classifier_free_guidance = guidance_scale > 1.0
        duration_hidden_states = self.model.model.encode_duration(duration)
        if classifier_free_guidance:
            bsz = 2 * num_samples_per_prompt

            encoder_hidden_states, boolean_encoder_mask = (
                self.model.model.encode_text_classifier_free(
                    prompt, num_samples_per_prompt=num_samples_per_prompt
                )
            )
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

        else:

            encoder_hidden_states, boolean_encoder_mask = self.model.model.encode_text(
                prompt, num_samples_per_prompt=num_samples_per_prompt
            )

        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(
            encoder_hidden_states
        )
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"))
        )

        pooled = torch.nanmean(masked_data, dim=1)
        pooled_projection = self.model.model.fc(pooled)

        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, duration_hidden_states], dim=1
        )  ## (bs,seq_len,dim)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas
        )

        latents = torch.randn(num_samples_per_prompt, self.model.model.audio_seq_len, 64)
        weight_dtype = latents.dtype

        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(self.model.model.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )

        timesteps = timesteps.to(device)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)
        # 6. Denoising loop
        current_step = 0
        ema = None
        while current_step<len(timesteps):
            if current_step < self.p or current_step==(len(timesteps)-1):
                t = timesteps[current_step]
                latents_input = (
                    torch.cat([latents] * 2) if classifier_free_guidance else latents
                )

                noise_pred = self.model.model.transformer(
                    hidden_states=latents_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=torch.tensor([t / 1000], device=device),
                    guidance=None,
                    pooled_projections=pooled_projection,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=audio_ids,
                    return_dict=False,
                )[0]

                if classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = scheduler.step(noise_pred, t, latents).prev_sample

                #accelerate module
                if ema is None:
                    if current_step >0:
                        ema = noise_pred-self.pre_noise
                else:
                    ema = (ema + noise_pred-self.pre_noise)/2

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
                        latents_input = (
                            torch.cat([latents] * 2) if classifier_free_guidance else latents
                        )

                        noise_pred = self.model.model.transformer(
                            hidden_states=latents_input,
                            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                            timestep=torch.tensor([t / 1000], device=device),
                            guidance=None,
                            pooled_projections=pooled_projection,
                            encoder_hidden_states=encoder_hidden_states,
                            txt_ids=txt_ids,
                            img_ids=audio_ids,
                            return_dict=False,
                        )[0]

                        if classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        #accelerate module
                        ema = (ema + noise_pred-self.pre_noise)/2
                        if self.k!=0:
                            self.gradient = (1/self.k)*ema
                        else:
                            self.gradient = ema

                        #upadte k
                        if (noise_pred - self.pre_noise - ema).abs().mean().item() < self.threshold:
                            self.k+=1
                        else:
                            if self.k>0:
                                self.k-=1
                    self.pre_noise = noise_pred

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                    current_step+=1
                    progress_bar.update()


        wave = self.model.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * self.model.vae.config.sampling_rate)
        wave = wave[:, :waveform_end]

        # Offload all models
        self.k=0
        self.gradient = None
        self.pre_noise = None
    
        return wave


model = TangoFluxInference(name='./ckpt/TangoFlux')
pipe = ETC(model=model,p=6,threshold=0.0675)
start = time.time()
audio = pipe('Hammer slowly hitting the wooden table', num_inference_steps=50, duration=10)
end = time.time()
print('use time: ',end-start)

torchaudio.save('kstep.wav', audio, 44100)

