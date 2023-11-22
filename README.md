# One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls

One More Step (OMS) module was proposed in [One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls](https://mhh0318.github.io)
by *Minghui Hu, Jianbin Zheng, Chuanxia Zheng, Tat-Jen Cham et al.*


By **adding one small step** on the top the sampling process, we can address the issues caused by the current schedule flaws of diffusion models **without changing the original model parameters**. This also allows for some control over low-frequency information, such as color. 

Our model is **versatile** and can be integrated into almost all widely-used Stable Diffusion frameworks. It's compatible with community favorites such as **LoRA, ControlNet, Adapter, and foundational models**.


## Usage

OMS now is supported diffusers with a customized pipeline [github](https://mhh0318.github.io).  To run the model, first install the latest version of the Diffusers library as well as `accelerate` and `transformers`.

```bash
pip install --upgrade pip
pip install --upgrade diffusers transformers accelerate
```

And then we clone the repo
```bash
git clone 
cd 
```


### SDXL

The OMS module can be loaded with SDXL base model `stabilityai/stable-diffusion-xl-base-1.0`. 
And all the SDXL based model and its LoRA can **share the same OMS** `h1t/oms_b_openclip_xl`.

Here is an example for SDXL with LCM-LoRA.

```python
from diffusers_patch import OMSPipeline

import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_id = "latent-consistency/lcm-lora-sdxl"

sd_pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
sd_scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
pipe.load_lora_weights(adapter_id)
sd_pipe.to("cuda")

pipe = OMSPipeline.from_pretrained('h1t/oms_b_openclip_xl', sd_pipeline = sd_pipe, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True, sd_scheduler=sd_scheduler)
pipe.to('cuda')

generator = torch.Generator(device=pipe.device).manual_seed(1337)

# pass oms_prompt for diverse prompt control
prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
image = pipe(prompt, oms_prompt='a blue night' , guidance_scale=1, num_inference_steps=4, oms_guidance_scale=2., generator=generator)

```
