import torchaudio
from tangoflux import TangoFluxInference
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Optional, Union, List
import inspect
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ruptures as rpt
import os
import torch.nn.functional as F
import ruptures as rpt

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

@torch.no_grad()
def return_noise(
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
    device = self.model.transformer.device
    scheduler = self.model.noise_scheduler

    if not isinstance(prompt, list):
        prompt = [prompt]
    if not isinstance(duration, torch.Tensor):
        duration = torch.tensor([duration], device=device)
    classifier_free_guidance = guidance_scale > 1.0
    duration_hidden_states = self.model.encode_duration(duration)
    if classifier_free_guidance:
        bsz = 2 * num_samples_per_prompt

        encoder_hidden_states, boolean_encoder_mask = (
            self.model.encode_text_classifier_free(
                prompt, num_samples_per_prompt=num_samples_per_prompt
            )
        )
        duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

    else:

        encoder_hidden_states, boolean_encoder_mask = self.model.encode_text(
            prompt, num_samples_per_prompt=num_samples_per_prompt
        )

    mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(
        encoder_hidden_states
    )
    masked_data = torch.where(
        mask_expanded, encoder_hidden_states, torch.tensor(float("nan"))
    )

    pooled = torch.nanmean(masked_data, dim=1)
    pooled_projection = self.model.fc(pooled)

    encoder_hidden_states = torch.cat(
        [encoder_hidden_states, duration_hidden_states], dim=1
    )  ## (bs,seq_len,dim)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps, sigmas
    )

    latents = torch.randn(num_samples_per_prompt, self.model.audio_seq_len, 64)
    weight_dtype = latents.dtype

    progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)

    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
    audio_ids = (
        torch.arange(self.model.audio_seq_len)
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(bsz, 1, 3)
        .to(device)
    )

    timesteps = timesteps.to(device)
    latents = latents.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)

    noise_list = []
    for i, t in enumerate(timesteps):
        latents_input = (
            torch.cat([latents] * 2) if classifier_free_guidance else latents
        )
        noise_pred = self.model.transformer(
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

        noise_list.append(noise_pred)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        
        progress_bar.update()

    last_latent = latents

    wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
    waveform_end = int(duration * self.vae.config.sampling_rate)
    wave = wave[:, :waveform_end]

    return wave,noise_list,last_latent,duration


#get different length prompt
data_path = './t2a/data/audiocaps/test.csv'
data = pd.read_csv(data_path)
caption_list = [anno.strip() for anno in data['caption']]
prompt_lengths = [len(prompt) for prompt in caption_list]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
histplot = sns.histplot(prompt_lengths, kde=True, bins=10, color='blue', edgecolor='black')
bin_edges = [int(patch.get_x()) for patch in histplot.patches]
intervals = list(zip(bin_edges[:-1], bin_edges[1:]))
prompts_set = []
for start, end in intervals:
    selected_prompts = [caption_list[i] for i in range(len(caption_list)) if start <= prompt_lengths[i] < end]
    prompts_set.append(selected_prompts[len(selected_prompts)//2])

#get each prompt noise
pipe = TangoFluxInference(name='./ckpt/TangoFlux')
pipe.vae.requires_grad_(False)

chazhi = None
sim = None
for prompt in tqdm(prompts_set):
    origin_audio,noise_list,latents,duration = return_noise(self=pipe,prompt=prompt, num_inference_steps=50, duration=10)
    #torchaudio.save('origin.wav', origin_audio, 44100)

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
        #decode audio
        audio_bias = pipe.vae.decode(latents_bias.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * pipe.vae.config.sampling_rate)
        audio_bias = audio_bias[:, :waveform_end]
        #torchaudio.save('bias.wav', audio_bias, 44100)
        #get mel spectgram and compute ssim
        score = F.l1_loss(origin_audio,audio_bias).item()
        ssim_list.append(score)
    if sim is None:
        sim = ssim_list
    else:
        for i in range(len(ssim_list)):
            sim[i]+=ssim_list[i]
chazhi = np.array(chazhi) / len(prompts_set)
sim = np.array(sim) / len(prompts_set)

'''
[0.13857575 0.06628755 0.06716062 0.05027343 0.04448322 0.06623545
 0.06648006 0.06745452 0.08350985 0.07223634 0.06972806 0.06982774
 0.06926528 0.07200786 0.06253377 0.06751337 0.06764431 0.06099252
 0.05437095 0.05627336 0.05615753 0.0518349  0.04914302 0.05134092
 0.05227728 0.04743505 0.04757364 0.04507352 0.05254839 0.04735048
 0.0413162  0.04304382 0.04269561 0.04389434 0.03655219 0.03697587
 0.03811277 0.04215851 0.03039545 0.03601727 0.03764512 0.03203111
 0.03093321 0.03205958 0.03665511 0.03339781 0.03498854 0.03496639
 0.03863942]
[0.02940917 0.01714098 0.01672548 0.01305426 0.01134586 0.01589471
 0.01711668 0.0177264  0.02294267 0.01739602 0.01620563 0.01627035
 0.0160213  0.01729746 0.01450036 0.01539754 0.01540672 0.01395509
 0.01264362 0.01296445 0.01224655 0.0116395  0.01087956 0.01157924
 0.01162933 0.01021482 0.01042763 0.0101289  0.01146388 0.01052602
 0.00933046 0.00941868 0.00947362 0.00948049 0.00801505 0.00813269
 0.00837401 0.00935395 0.00648809 0.00785363 0.0082808  0.00718994
 0.00667626 0.00699744 0.00829665 0.00741837 0.00798465 0.00788176
 0.0087312 ]
'''
result = rpt.KernelCPD(kernel="rbf").fit(sim).predict(n_bkps=4)[:-1]
print('Final Threshold: ',chazhi[result].max()) 
