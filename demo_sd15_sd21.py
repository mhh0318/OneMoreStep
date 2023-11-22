from diffusers_patch import OMSPipeline

import torch
from diffusers import StableDiffusionPipeline


sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16",safety_checker=None).to('cuda')
print('successfully load pipe')

pipe = OMSPipeline.from_pretrained('h1t/oms_b_openclip_15_21', sd_pipeline = sd_pipe, torch_dtype=torch.float16, variant="fp16", trust_remote_code=True)
pipe.to('cuda')


generator = torch.Generator(device=pipe.device).manual_seed(100)

prompt = "a starry night"

image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20, oms_guidance_scale=2., generator=generator)

image['images'][0].save('sd15_wo_oms.png')

# pass "oms_flag = False" to unload OMS module

image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20, oms_guidance_scale=1., generator=generator, oms_flag=False)

image['images'][0].save('sd15_wo_oms.png')
