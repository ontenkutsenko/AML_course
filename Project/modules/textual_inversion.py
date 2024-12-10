import argparse
import itertools
import math
import os
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from packaging import version

from slugify import slugify
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from huggingface_hub import create_repo


#prompt templates for training
imagenet_templates_small = [
    "a photo of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the weird {}",
    "a photo of a cool {}"
]

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
            PIL_INTERPOLATION = {
                "linear": PIL.Image.Resampling.BILINEAR,
                "bilinear": PIL.Image.Resampling.BILINEAR,
                "bicubic": PIL.Image.Resampling.BICUBIC,
                "lanczos": PIL.Image.Resampling.LANCZOS,
                "nearest": PIL.Image.Resampling.NEAREST,
            }
        else:
            PIL_INTERPOLATION = {
                "linear": PIL.Image.LINEAR,
                "bilinear": PIL.Image.BILINEAR,
                "bicubic": PIL.Image.BICUBIC,
                "lanczos": PIL.Image.LANCZOS,
                "nearest": PIL.Image.NEAREST,
                }

        self.interpolation = PIL_INTERPOLATION[interpolation]

        self.templates = imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    
# what_to_teach = "object"
# placeholder_token = f"<{swapped_dict[worst_class]}>"
# # initializer_token = swapped_dict[worst_class]
# initializer_token = "flower"
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"

def initialize_model(pretrained_model_name_or_path, placeholder_token, initializer_token):
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )   

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]


    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    return text_encoder, vae, unet, tokenizer, placeholder_token_id


# train_dataset = TextualInversionDataset(
#       data_root=save_path,
#       tokenizer=tokenizer,
#       size=vae.sample_size,
#       placeholder_token=placeholder_token,
#       repeats=100,
#       learnable_property=what_to_teach, #Option selected above between object and style
#       center_crop=False,
#       set="train",
# )



# noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")


# hyperparameters = {
#     "learning_rate": 5e-04,
#     "scale_lr": True,
#     "max_train_steps": 2000,
#     "save_steps": 250,
#     "train_batch_size": 4,
#     "gradient_accumulation_steps": 1,
#     "gradient_checkpointing": True,
#     "mixed_precision": "fp16",
#     "seed": 42,
#     "output_dir": "sd-concept-output"
# }
# !mkdir -p sd-concept-output

logger = get_logger(__name__)

def save_progress(text_encoder, placeholder_token_id, accelerator, save_path, placeholder_token):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def training_function(text_encoder, vae, unet, hyperparameters, train_dataset, noise_scheduler, 
tokenizer, placeholder_token_id, placeholder_token, pretrained_model_name_or_path
):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hyperparameters["mixed_precision"]
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                 # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hyperparameters["save_steps"] == 0:
                    save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path, placeholder_token)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        save_path = os.path.join(output_dir, f"learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

# import accelerate

# accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet))

# for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
#   if param.grad is not None:
#     del param.grad  # free some memory
#   torch.cuda.empty_cache()

# save_concept_to_public_library = True
# name_of_your_concept = "sword-lily-flowers102"
# hf_token_write = ""
# path_with_images = "textual_inversion_images"
# placeholder_token = "<flower>"
# what_to_teach = "object"
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
# save_path = "textual_inversion_images"

def save_concept(
        name_of_your_concept:str,
        hyperparameters:dict,
        path_with_images:str,
        placeholder_token:str,
        what_to_teach:str,
  		pretrained_model_name_or_path:str,
        hf_token_write:str = ""

):
    repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
    output_dir = hyperparameters["output_dir"]
    if(not hf_token_write):
        with open(HfFolder.path_token, 'r') as fin: hf_token = fin.read();
    else:
        hf_token = hf_token_write
    #Join the Concepts Library organization if you aren't part of it already
    command = f"curl -X POST -H 'Authorization: Bearer {hf_token}' -H 'Content-Type: application/json' https://huggingface.co/organizations/sd-concepts-library/share/VcLXJtzwwxnHYCkNMLpSJCdnNFZHQwWywv"

    subprocess.run(command, shell=True, check=True) 
#    !curl -X POST -H 'Authorization: Bearer '$hf_token -H 'Content-Type: application/json' https://huggingface.co/organizations/sd-concepts-library/share/VcLXJtzwwxnHYCkNMLpSJCdnNFZHQwWywv
    images_upload = os.listdir(path_with_images)
    image_string = ""
    repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
    for i, image in enumerate(images_upload):
        image_string = f'''{image_string}![{placeholder_token} {i}](https://huggingface.co/{repo_id}/resolve/main/concept_images/{image})
    '''
    if(what_to_teach == "style"):
        what_to_teach_article = f"a `{what_to_teach}`"
    else:
        what_to_teach_article = f"an `{what_to_teach}`"
    readme_text = f'''---
    license: mit
    base_model: {pretrained_model_name_or_path}
    ---
    ### {name_of_your_concept} on Stable Diffusion
    This is the `{placeholder_token}` concept taught to Stable Diffusion via Textual Inversion. You can load this concept into the [Stable Conceptualizer](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb) notebook. You can also train your own concepts and load them into the concept libraries using [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).

    Here is the new concept you will be able to use as {what_to_teach_article}:
    {image_string}
    '''
    #Save the readme to a file
    readme_file = open("README.md", "w")
    readme_file.write(readme_text)
    readme_file.close()
    #Save the token identifier to a file
    text_file = open("token_identifier.txt", "w")
    text_file.write(placeholder_token)
    text_file.close()
    #Save the type of teached thing to a file
    type_file = open("type_of_concept.txt","w")
    type_file.write(what_to_teach)
    type_file.close()
    operations = [
        CommitOperationAdd(path_in_repo="learned_embeds.bin", path_or_fileobj=f"{output_dir}/learned_embeds.bin"),
        CommitOperationAdd(path_in_repo="token_identifier.txt", path_or_fileobj="token_identifier.txt"),
        CommitOperationAdd(path_in_repo="type_of_concept.txt", path_or_fileobj="type_of_concept.txt"),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="README.md"),
    ]
    create_repo(repo_id,private=True, token=hf_token)
    api = HfApi()
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload the concept {name_of_your_concept} embeds and token",
        token=hf_token
    )
    api.upload_folder(
        folder_path=path_with_images,
        path_in_repo="concept_images",
        repo_id=repo_id,
        token=hf_token
  )