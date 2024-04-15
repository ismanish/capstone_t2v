# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import copy
import inspect
import os.path
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision.utils
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
# from diffusers.loaders import TextualInversionLoaderMixin # a1
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.utils import is_accelerate_available
from einops import rearrange, repeat
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.unet import UNet3DConditionModel
from ..prompt_attention import attention_util 

logger = logging.get_logger(__name__) 


import logging
import functools
import datetime

@dataclass
class SpatioTemporalPipelineOutput(BaseOutput):
    """Dataclass for the output of SpatioTemporalPipeline."""
    videos: Union[torch.Tensor, np.ndarray]


class SpatioTemporalPipeline(DiffusionPipeline):
    """
    SpatioTemporalPipeline handles the generation process of videos given textual prompts.
    It integrates various components like VAE, text encoder, tokenizer, UNet, and schedulers.
    """
    
    _optional_components = []
    

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet3DConditionModel,
            # text_inv: TextualInversionLoaderMixin # a1
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            noise_scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ] = None,
            disk_store: bool = True,
            config=None
    ):
        """Initializes the pipeline with the necessary models and schedulers."""
        super().__init__()
        logger.info("SpatioTemporalPipeline Class initiated")
        
        logger.info("HF vae, encoder, tokenizer etc. modules are registered in the Pipeline")
        # Register necessary components with the pipeline
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            # text_inv = text_inv # a1
        )
        
        # Compute the scale factor based on the VAE architecture
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        logger.info(f"vae_scale_factor:- {self.vae_scale_factor}")

        # Initialize the attention mechanism controller if required
        self.controller = attention_util.AttentionStore(disk_store=True, config=config)
        # self.load_textual_inversion("/home/jupyter/manish/Free-Bloom/diffusers/examples/textual_inversion/textual_inversion_cat") #a1
        logger.info("AttentionTest from prompt_attention.attention_util.py has been assigned to controller")
        self.hyper_config = config

    def enable_vae_slicing(self):
        """Enables slicing in VAE to manage memory usage during decoding."""
        logger.info("vae_enable_slicing initiated")
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """Disables slicing in VAE, allowing for full image generation at once."""
        logger.info("disable_vae_slicing initiated")
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        """Property to get the device on which the model is running."""
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        """
        Encodes the text prompt into embeddings using the text encoder, handling classifier-free guidance if needed.
        """
        
        # Tokenize the prompt and encode it to get text embeddings
        text_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeddings = self.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.get('attention_mask')).last_hidden_state

        # If doing classifier-free guidance, prepare the unconditioned embeddings and concatenate with conditioned ones
        if do_classifier_free_guidance:
            uncond_prompt = negative_prompt or [""]
            uncond_inputs = self.tokenizer(uncond_prompt, return_tensors="pt", padding=True, truncation=True, max_length=text_inputs.input_ids.size(1)).to(device)
            uncond_embeddings = self.text_encoder(uncond_inputs.input_ids, attention_mask=uncond_inputs.get('attention_mask')).last_hidden_state

            # Ensure both sets of embeddings have the same sequence length
            # Pad embeddings to have matching sequence lengths and concatenate
            seq_len = max(text_embeddings.size(1), uncond_embeddings.size(1))
            if text_embeddings.size(1) < seq_len:
                padding = seq_len - text_embeddings.size(1)
                text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, padding))
            if uncond_embeddings.size(1) < seq_len:
                padding = seq_len - uncond_embeddings.size(1)
                uncond_embeddings = torch.nn.functional.pad(uncond_embeddings, (0, 0, 0, padding))

            text_embeddings = torch.cat([uncond_embeddings.repeat_interleave(num_videos_per_prompt, dim=0), text_embeddings])

        return text_embeddings.repeat_interleave(num_videos_per_prompt, dim=0)

    def decode_latents(self, latents, return_tensor=False):
        """
        Decodes the latent variables into video frames using the VAE decoder.
        """
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents  # Scale the latents to match VAE's expected distribution
        latents = rearrange(latents, "b c f h w -> (b f) c h w")  # Flatten the batch and frame dimensions
        video = self.vae.decode(latents).sample  # Decode latents to video frames
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)  # Reshape back to original dimensions
        video = (video / 2 + 0.5).clamp(0, 1)  # Normalize the video frames to [0, 1]

        if return_tensor:
            return video  # Return as a tensor
        return video.cpu().float().numpy()  # Return as a NumPy array


    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, frame_same_noise=False, store_attention=True):
        """
        Prepares the latent variables for the diffusion process, optionally reusing provided latents.
        """
        logger.info("prepare_latents in SpatioTemporalPipeline called")

        scaled_height = height // self.vae_scale_factor # Adjust dimensions according to VAE scale factor
        scaled_width = width // self.vae_scale_factor
        temporal_dimension = 1 if frame_same_noise else video_length
        shape = (batch_size, num_channels_latents, temporal_dimension, scaled_height, scaled_width)

        if latents is None:
            # Generate random latents if not provided
            rand_device = device 

            latents_shape = shape

            latents = torch.randn(latents_shape, generator=generator, device=rand_device, dtype=dtype)
            if isinstance(generator, list):
                latents = torch.cat([torch.randn(latents_shape, generator=g, device=rand_device, dtype=dtype) for g in generator], dim=0)
            if frame_same_noise:
                latents = latents.expand(-1, -1, video_length, -1, -1) # Repeat latents across frames if needed

        latents = latents.to(device) * self.scheduler.init_noise_sigma # Scale latents by noise level
        return latents


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            fixed_latents: Optional[dict[torch.FloatTensor]] = None,
            fixed_latents_idx: list = None,
            inner_idx: list = None,
            init_text_embedding: torch.Tensor = None,
            mask: Optional[dict] = None,
            save_vis_inner=False,
            return_tensor=False,
            return_text_embedding=False,
            output_dir=None,
            **kwargs,
    ):
        if self.controller is not None:
            self.controller.reset()
            self.controller.batch_size = video_length

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device
        # print(device)
        
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # print(timesteps)

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(latents_dtype)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if self.controller:
                    latents = self.controller.step_callback(latents, inner_idx).to(latents.dtype)

                self.controller.empty_cache()

                progress_bar.update()

        video = self.decode_latents(latents, return_tensor)
        if output_type == "tensor" and isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        if return_dict:
            result = SpatioTemporalPipelineOutput(videos=video)
            return (result, text_embeddings) if return_text_embedding else result

        return (video, text_embeddings) if return_text_embedding else video


