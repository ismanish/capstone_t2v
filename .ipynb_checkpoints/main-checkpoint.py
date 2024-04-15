import os
import numpy as np
import argparse
import datetime
import inspect
import logging
from omegaconf import OmegaConf
import json

from typing import Dict, Optional

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler,PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
import torch
import torch.utils.checkpoint
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from t2v.models.unet import UNet3DConditionModel
from t2v.pipelines.pipeline_spatio_temporal import SpatioTemporalPipeline
from t2v.util import save_videos_grid, save_videos_per_frames_grid

from frame_generator import generate_frame_descriptions


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%H_%M_%S")
    filename = os.path.join(log_dir, f"log_{current_time}.log")

    logging.basicConfig(
        filename=filename,
        filemode='a', 
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    return logger 

def main(conf):
    
    logger = setup_logging()
    logger.info("Main function is called")

    pretrained_model_path = "./data/diffusion_weights/stable-diffusion-v1-5"
    output_dir = "./outputs"
    validation_data = conf.validation_data
    mixed_precision = conf.mixed_precision #"fp16"
    seed = conf.seed #333
    set_seed(seed)
    diversity_rand_ratio = conf.inference_config['diversity_rand_ratio']
    enable_xformers_memory_efficient_attention = True

    accelerator = Accelerator(mixed_precision=mixed_precision)
    

    logger.info("HF pipelines(tokenizer,vae,unet etc.) are called")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    
    # switching off the back propogation
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)


    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()


    # Get the validation pipeline
    logger.info("SpatioTemporalPipeline is getting initialized")
    validation_pipeline = SpatioTemporalPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        # scheduler=PNDMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        disk_store=True, #False,
        config=conf
    )
    validation_pipeline.enable_vae_slicing()
    validation_pipeline.scheduler.set_timesteps(validation_data.num_inv_steps)

    # unet = accelerator.prepare(unet)

    weight_dtype = torch.float32
    # accelerator.mixed_precision == mixed_precision
     

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)


    text_encoder.eval()
    vae.eval()
    unet.eval()

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(seed)

    samples = []

    context = "man playing with a dog"
    prompt = generate_frame_descriptions(context, custom = False)
    # print(prompt)
    

    prompt = [
    "A poodle dressed as an astronaut, helmet gleaming, floats amidst the vast expanse of space.",
    "The poodle, in a custom space suit, drifts near a space station, with Earth visible in the background.",
    "Amidst a backdrop of distant stars, the poodle maneuvers with a small jetpack, dressed in astronaut gear.",
    "The astronaut poodle examines a floating satellite, tools in paw, under the watchful gaze of a distant sun.",
    "Floating weightlessly, the poodle in its astronaut suit passes a nebula, awash in colors of blue and purple.",
    "The poodle astronaut playfully chases a stray space tool, its suit reflecting the light of the Milky Way.",
    "In zero gravity, the poodle tests out its space suit's agility, leaping between small asteroids.",
    "With the moon in the background, the poodle astronaut barks quietly, the sound muted by the vacuum of space."
]

    #Overwriting the prompt below to try out different prompts quickly
# #     #Natural transition in a forest scene from summer to autumn
#     prompt =  [
#     "A dense forest canopy is lush and vibrant, filled with the rich greens of late summer under a clear blue sky.",
#     "The first hints of autumn appear as subtle yellow and orange tinges begin to dot the canopy, contrasting with the green.",
#     "Leaves turn brighter shades of orange, red, and gold, creating a colorful mosaic that blankets the entire forest.",
#     "A crisp autumn breeze causes leaves to gently fall, swirling through the air and slowly carpeting the forest floor.",
#     "The forest is now ablaze with full autumn colors; red, orange, and yellow leaves dominate the landscape under a soft, overcast sky.",
#     "As late autumn sets in, the trees are left bare with most leaves fallen, and a thin layer of frost begins to coat the now-visible forest floor."
# ]

#     #Growth of a Coral Reef
#     prompt = [
#     "A barren underwater rock starts to accumulate small clusters of coral polyps that anchor firmly to its surface.",
#     "Over time, the polyps multiply and begin to form intricate structures, creating a small but growing reef.",
#     "Diverse marine life starts to inhabit the reef; small fish weave through the corals, which are now vibrant with color.",
#     "The reef expands, forming a complex ecosystem with various types of corals, sponges, and anemones.",
#     "Years pass, and the reef becomes a large, bustling hub of aquatic activity, hosting thousands of species.",
#     "The mature reef is now a critical marine habitat, playing a vital role in the ocean's biodiversity and health."
# ]
    
# #     #Construction of a Skyscraper:
#     prompt = [
#     "A vast barren land stretches under the open sky, earmarked for a monumental skyscraper.",
#     "Surveyors traverse the site, placing markers and measuring for precise foundational work.",
#     "The first signs of activity begin as small teams assess and prepare the ground for excavation.",
#     "Specialized ground-penetrating equipment rolls in to test soil stability and composition deep below the surface.",
#     "Temporary fencing is erected around the perimeter, signaling the impending commencement of major construction.",
#     "Workers start arriving in groups, setting up temporary offices and storage for construction materials.",
#     "Initial digging operations start, with teams using hand tools to outline the future foundation trenches.",
#     "Heavy excavation machinery is delivered to the site, ready to begin the major earth-moving work.",
#     "Massive diggers and bulldozers start reshaping the landscape, carving out the designated foundation pit.",
#     "Piling machines drive deep into the earth, installing the sturdy pilings needed to support the skyscraper’s massive weight.",
#     "The site is a hive of activity, with workers coordinating the pouring of concrete to form the robust base of the skyscraper.",
#     "Cranes begin to dot the skyline, lifting steel beams and construction materials into place.",
#     "The construction of sublevels progresses, with concrete and steel forming the underground support structure.",
#     "Above ground, the steel skeleton starts to rise, with each beam meticulously positioned and secured.",
#     "More machinery and cranes operate in tandem, speeding up the assembly of the skyscraper’s core and frame.",
#     "Construction teams work on multiple floors simultaneously, adding structural elements and floor slabs.",
#     "The structure's outline becomes more defined, with the lower floors fully framed and mid-levels beginning to take shape.",
#     "Glass and cladding materials start being installed on the completed lower floors, encasing the building.",
#     "The higher levels of the skyscraper take form, with steelwork completing and window installations beginning.",
#     "The final structural components are put in place, with the topmost beams and architectural features defining the skyscraper’s height and silhouette."
# ]

    logger.info(prompt)
    negative_prompt = conf['validation_data']['negative_prompt']
    negative_prompt = [negative_prompt] * len(prompt)

    with (torch.no_grad()):
        logger.info("prepare_latents from SpatioTemporalPipeline is called for X_base")
        x_base = validation_pipeline.prepare_latents(batch_size=1,
                                                     num_channels_latents=4,
                                                     video_length=len(prompt),
                                                     height=512,
                                                     width=512,
                                                     dtype=weight_dtype,
                                                     device=unet.device,
                                                     generator=generator,
                                                     store_attention=True,
                                                     frame_same_noise=True)

        logger.info("prepare_latents from SpatioTemporalPipeline is called for X_res")
        x_res = validation_pipeline.prepare_latents(batch_size=1,
                                                    num_channels_latents=4,
                                                    video_length=len(prompt),
                                                    height=512,
                                                    width=512,
                                                    dtype=weight_dtype,
                                                    device=unet.device,
                                                    generator=generator,
                                                    store_attention=True,
                                                    frame_same_noise=False)

        x_T = np.cos(diversity_rand_ratio* np.pi / 2) * x_base + np.sin(
            diversity_rand_ratio * np.pi / 2) * x_res

        validation_data.pop('negative_prompt')
        logger.info(f"x_T of shape {x_T.shape} is generated")
        # key frame
        key_frames, text_embedding = validation_pipeline(prompt, video_length=len(prompt), generator=generator,
                                                         latents=x_T.type(weight_dtype),
                                                         negative_prompt=negative_prompt,
                                                         output_dir=output_dir,
                                                         return_text_embedding=True,
                                                         **validation_data)
        torch.cuda.empty_cache()
        logger.info("validation pipeline i.e. SpatioTemporalPipeline is called to get final frames and text_embeddings")
        logger.info(f"key_frames of shape {key_frames.videos.shape} generated")
        # logger.info(f"key_frames:- {key_frames} generated")
        samples.append(key_frames[0])
        logger.info(f"samples generated are of shape - {len(samples)}")
    samples = torch.concat(samples)
    logger.info(f"samples shape after concat - {samples.shape}")
    # logger.info(f"samples:- {samples} generated")
    save_path = f"{output_dir}/samples/sample.gif"
    # save_videos_grid(samples, save_path, n_rows=6)
    # save_videos_per_frames_grid(samples, f'{output_dir}/img_samples', n_rows=6)
    save_videos_grid(samples, save_path, n_rows=len(prompt))
    save_videos_per_frames_grid(samples, f'{output_dir}/img_samples', n_rows=len(prompt))
    logger.info(f"Saved samples to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    main(conf)