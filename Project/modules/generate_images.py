# Import necessary libraries
from diffusers import StableDiffusionPipeline
import torch
import os

def generate_from_prompt(
        model_id: str,
        prompt: str,
        device: str,
        output_dir: str,
        num_images: int = 1
):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline.to(device)

    for i in range(num_images):
        with torch.no_grad():
            image = pipeline(prompt).images[0]

        # Save the generated image
        name = f"{prompt.lower().replace(' ', '_')}_{i}.png"
        output_path = os.path.join(output_dir, name)

        image.save(output_path)
        print(f"Image saved at {output_path}")