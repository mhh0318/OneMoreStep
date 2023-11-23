# One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls

One More Step (OMS) module was proposed in [One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls](https://github.com/mhh0318/OneMoreStep)
by *Minghui Hu, Jianbin Zheng, Chuanxia Zheng, Tat-Jen Cham et al.*


By **adding one small step** on the top the sampling process, we can address the issues caused by the current schedule flaws of diffusion models **without changing the original model parameters**. This also allows for some control over low-frequency information, such as color. 

Our model is **versatile** and can be integrated into almost all widely-used Stable Diffusion frameworks. It's compatible with community favorites such as **LoRA, ControlNet, Adapter, and foundational models**.


## Usage

OMS now is supported diffusers with a customized pipeline [github](https://github.com/mhh0318/OneMoreStep).  To run the model, first install the latest version of the Diffusers library as well as `accelerate` and `transformers`.

```bash
pip install --upgrade pip
pip install --upgrade diffusers transformers accelerate
```

And then we clone the repo
```bash
git clone https://github.com/mhh0318/OneMoreStep.git
cd OneMoreStep
```


### SDXL

The OMS module can be loaded with SDXL base model `stabilityai/stable-diffusion-xl-base-1.0`. 
And all the SDXL based model and its LoRA can **share the same OMS** `h1t/oms_b_openclip_xl`.

Here is an example for SDXL with LCM-LoRA.
Firstly import the related packages and choose SDXL based backbone and LoRA:

```python
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler

sd_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", add_watermarker=False).to('cuda')

sd_scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe.load_lora_weights(l'latent-consistency/lcm-lora-sdxl'h, variant="fp16")
```

Following import the customized OMS pipeline to wrap the backbone and add OMS for sampling. We have uploaded the `.safetensors` to [HuggingFace Hub](https://huggingface.co/h1t/). There are 2 choices for SDXL backbone currently, one is base OMS module with OpenCLIP text encoder [h1t/oms_b_openclip_xl)](https://huggingface.co/h1t/oms_b_openclip_xl) and the other is large OMS module with two text encoder followed by SDXL architecture [h1t/oms_l_mixclip_xl)](https://huggingface.co/h1t/oms_b_mixclip_xl).
```python
from diffusers_patch import OMSPipeline

pipe = OMSPipeline.from_pretrained('h1t/oms_b_openclip_xl', sd_pipeline = sd_pipe, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True, sd_scheduler=sd_scheduler)
pipe.to('cuda')
```

After setting a random seed, we can easily generate images with the OMS module.
```python
prompt = 'close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux'
generator = torch.Generator(device=pipe.device).manual_seed(1024)

image = pipe(prompt, guidance_scale=1, num_inference_steps=4, generator=generator)
image['images'][0]
```
![oms_xl](/asset/sdxl_oms.png)

Or we can offload the OMS module and generate a image only using backbone
```python
image = pipe(prompt, guidance_scale=1, num_inference_steps=4, generator=generator, oms_flag=False)
image['images'][0]
```
![oms_xl](/asset/sdxl_wo_oms.png)

For more functions like diverse prompt, please refer to `demo_sdxl_lcm_lora.py`. 

### SD15 and SD21

Due to differences in the *VAE latent space* between SD1.5/SD2.1 and SDXL, the OMS module for SD1.5/SD2.1 cannot be shared with SDXL, **however, SD1.5/SD2.1 can share the same OMS module as well as with models like LCM that are based on SD1.5 or SD2.1.** For more details, please refer to our paper.

We have uploaded one OMS module for SD15/21 series at [h1t/oms_b_openclip_15_21](https://huggingface.co/h1t/oms_b_openclip_15_21), which has a base architecture, an OpenCLIP text encoder. 

We simply put a demo here:

```python
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler

sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, variant="fp16", safety_checker=None).to('cuda')

pipe = OMSPipeline.from_pretrained('h1t/oms_b_openclip_15_21', sd_pipeline = sd_pipe, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True)
pipe.to('cuda')

generator = torch.Generator(device=pipe.device).manual_seed(100)

prompt = "a starry night"

image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20, oms_guidance_scale=2., generator=generator)

image['images'][0]
```
![oms_15](/asset/sd15_oms.png)

and without OMS:

```python
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20, oms_guidance_scale=2., generator=generator, oms_flag=False)

image['images'][0]
```
![oms_15](/asset/sd15_wo_oms.png)

We found that the quality of the generative model has been greatly improved.