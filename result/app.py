import mediapy as media
import random
import sys
import torch
import argparse
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Pick:
# -    2, 4 or 8 steps for lora,
# - 1, 2, 4 or 8 steps for unet.
num_inference_steps = 4

# Prefer "unet" over "lora" for quality.
use_lora = False
model_type = "lora" if use_lora else "unet"

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = f"sdxl_lightning_{num_inference_steps}step_{model_type}.safetensors"
device = "cuda"

unet = UNet2DConditionModel.from_config(
    base,
    subfolder="unet",
    ).to(device, torch.float16)

unet.load_state_dict(
    load_file(
        hf_hub_download(
            repo,
            ckpt,
            ),
        device=device,
        ),
    )

pipe = StableDiffusionXLPipeline.from_pretrained(
    base,
    unet=unet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    ).to(device)

if use_lora:
  pipe.load_lora_weights(hf_hub_download(repo, ckpt))
  pipe.fuse_lora()

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    )

parser = argparse.ArgumentParser(description='Process a sentence.')

# Add argument for the sentence
parser.add_argument('sentence', metavar='sentence', type=str, nargs='+',
                    help='a sentence to process')

# Parse the arguments
args = parser.parse_args()

# Combine the sentence into a single string
sentence = ' '.join(args.sentence)

prompt = sentence
seed = random.randint(0, sys.maxsize)

images = pipe(
    prompt = prompt,
    guidance_scale = 0.0,
    num_inference_steps = num_inference_steps,
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

low_res_image = images[0].resize((256, 256))
low_res_array = np.array(low_res_image)

# Add random noise to the low-resolution image
noise_factor = 0.5  # Adjust the noise factor as needed
noisy_low_res_array = low_res_array + noise_factor * np.random.randn(*low_res_array.shape)

# Clip the pixel values to be in the valid range [0, 255]
noisy_low_res_array = np.clip(noisy_low_res_array, 0, 255)

# Convert the noisy NumPy array back to a PIL image
noisy_low_res_image = Image.fromarray(noisy_low_res_array.astype(np.uint8))

print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
media.show_images([noisy_low_res_image])
noisy_low_res_image.save("output.jpg")
output_image = Image.open("output.jpg")

# Resize the image to a smaller size
pixelated_image = output_image.resize((96,96), resample=Image.NEAREST)

# Resize the pixelated image back to the original size
pixelated_image = pixelated_image.resize(output_image.size, resample=Image.NEAREST)

# Display the pixelated image
pixelated_image.save("result.jpg")
pixelated_image= Image.open("result.jpg")
pixelated_image.show()

