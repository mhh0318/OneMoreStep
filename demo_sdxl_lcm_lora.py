from diffusers_patch import OMSPipeline

import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler


sd_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", add_watermarker=False).to('cuda')
print('successfully load pipe')
sd_scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
lora_path = 'latent-consistency/lcm-lora-sdxl'
sd_pipe.load_lora_weights(lora_path, variant="fp16")


pipe = OMSPipeline.from_pretrained('h1t/oms_b_openclip_xl', sd_pipeline = sd_pipe, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True, sd_scheduler=sd_scheduler)
pipe.to('cuda')

generator = torch.Generator(device=pipe.device).manual_seed(1024)

# pass oms_prompt for diverse prompt control
prompt = "a car"
image = pipe(prompt, oms_prompt="red car", guidance_scale=1, num_inference_steps=4, oms_guidance_scale=1.5, generator=generator)

image['images'][0].save('sdxl_oms_diverse.png')