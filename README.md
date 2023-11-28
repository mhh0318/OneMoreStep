# One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls

<a href="http://arxiv.org/abs/2311.15744"><img src="https://img.shields.io/badge/arXiv-2211.15744-b31b1b.svg" height=22.5></a>
<a href="https://jabir-zheng.github.io/OneMoreStep/"><img src="https://img.shields.io/badge/Web-Project Page-brightgreen.svg" height=22.5></a>
<a href="https://huggingface.co/spaces/h1t/oms_sdxl_lcm"><img src="https://img.shields.io/badge/HuggingFace-Space-purple.svg" height=22.5></a> 


One More Step (OMS) module was proposed in [One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls](http://arxiv.org/abs/2311.15744)
by *Minghui Hu, Jianbin Zheng, Chuanxia Zheng, Tat-Jen Cham et al.*

By incorporating **one minor, additional step** atop the existing sampling process, it can address inherent limitations in the diffusion schedule of current diffusion models.  Crucially, this augmentation does not necessitate alterations to the original parameters of the model. Furthermore, the OMS module enhances control over low-frequency elements, such as color, within the generated images.

Our model is **versatile** and allowing for **seamless integration** with a broad spectrum of prevalent Stable Diffusion frameworks.  It demonstrates compatibility with community-favored tools and techniques, including LoRA, ControlNet, Adapter, and other foundational models, underscoring its utility and adaptability in diverse applications.

## Usage

OMS now is supported diffusers with a customized pipeline, as detailed in [github](https://github.com/mhh0318/OneMoreStep). To run the model, first install the latest version of `diffusers` (especially for `LCM` feature) as well as `accelerate` and `transformers`.

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

The OMS module is seamlessly compatible with the SDXL base model, specifically `stabilityai/stable-diffusion-xl-base-1.0`. It offers the distinct advantage of being **universally applicable** across all SDXL-based models and their respective LoRA configurations, exemplified by the shared OMS module `h1t/oms_b_openclip_xl`.

To illustrate its application with the SDXL model enhanced by LCM-LoRA, one begins by importing the necessary packages. Then involves selecting the appropriate SDXL-based backbone along with the LoRA configuration:
```python
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler

sd_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", add_watermarker=False).to('cuda')

sd_scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe.load_lora_weights(l'latent-consistency/lcm-lora-sdxl'h, variant="fp16")
```

Following import the customized OMS pipeline to wrap the backbone and add OMS for sampling. We The required `.safetensor` files have been made accessible on the [h1t'sHuggingFace Hub](https://huggingface.co/h1t/). Currently, there are 2 choices for SDXL backbone, one is base OMS module with OpenCLIP text encoder [h1t/oms-b-openclip-xl](https://huggingface.co/h1t/oms_b_openclip_xl) and the other is large OMS module with two text encoder followed by SDXL architecture [h1t/oms_l-mixclip-xl](https://huggingface.co/h1t/oms_b_mixclip_xl).
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

It is important to note the distinctions in the Variational Autoencoder (VAE) latent spaces among different versions of Stable Diffusion models. Specifically, due to these differences, the OMS module designed for SD1.5 and SD2.1 models is not interchangeable with the SDXL model. However, models based on SD1.5 or SD2.1, including those like LCM and various LoRAs, can **universally utilize the same OMS module**. This interoperability between SD1.5/SD2.1 and their derivative models is a key consideration. For a comprehensive understanding of these nuances, we invite readers to consult our detailed paper.

There is a OMS module for SD15/21 series accessible at [h1t/oms-b-openclip-15-21](https://huggingface.co/h1t/oms_b_openclip_15_21), which has a base architecture and an OpenCLIP text encoder. 

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
