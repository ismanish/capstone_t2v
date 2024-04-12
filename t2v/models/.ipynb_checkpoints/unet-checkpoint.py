from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import json

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Configuration and model utilities from the diffusers library
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
# from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
# Time embedding and timestep utilities
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
# Importing the custom UNet blocks defined elsewhere
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
# Importing the custom ResNet blocks
from .resnet import InflatedConv3d, InflatedGroupNorm

# Initialize a logger for the module
logger = logging.get_logger(__name__)


# Output class for the UNet model
@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor # The output sample from the UNet model


# Main class for the 3D conditional UNet model
class UNet3DConditionModel(ModelMixin, ConfigMixin):
    # Indicates that this model supports gradient checkpointing to save memory during training
    _supports_gradient_checkpointing = True
    
    # Constructor method with various parameters for the UNet architecture
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,  # Size of the generated samples
        in_channels: int = 4,  # Number of channels in the input tensor
        out_channels: int = 4,  # Number of channels in the output tensor
        center_input_sample: bool = False,  # If True, centers the input sample
        flip_sin_to_cos: bool = True,  # If True, flips the sin embedding to cos
        freq_shift: int = 0,  # Frequency shift for the time embedding
        down_block_types: Tuple[str] = (  # Types of blocks to use in the downsampling path
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",  # Type of block to use in the middle of UNet
        up_block_types: Tuple[str] = (  # Types of blocks to use in the upsampling path
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,  # If True, uses only cross-attention layers
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),  # Number of channels in each block
        layers_per_block: int = 2,  # Number of layers in each block
        downsample_padding: int = 1,  # Padding used in downsampling
        mid_block_scale_factor: float = 1,  # Scaling factor for the mid block
        act_fn: str = "silu",  # Activation function to use
        norm_num_groups: int = 32,  # Number of groups for group normalization
        norm_eps: float = 1e-5,  # Epsilon value for normalization
        cross_attention_dim: int = 1280,  # Dimension for cross-attention layers
        attention_head_dim: Union[int, Tuple[int]] = 8,  # Dimension of attention heads
        dual_cross_attention: bool = False,  # If True, uses dual cross-attention mechanism
        use_linear_projection: bool = False,  # If True, uses linear projection in attention
        class_embed_type: Optional[str] = None,  # Type of class embedding
        num_class_embeds: Optional[int] = None,  # Number of class embeddings
        upcast_attention: bool = False,  # If True, upcasts attention tensors
        resnet_time_scale_shift: str = "default",  # Time scale shift strategy for ResNet blocks
    
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Initialize the first convolution layer to process the input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

         #time - changes numeric timestep and transforms it using sine and cosine into a vector space of block_out_channels[0], 320 here
        # Initialize the time projection layer to process the timestep information
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        # Embedding for time, converting timestep into a high dimensional representation
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Conditional class embedding initialization, if applicable
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None
            
         # Lists to hold the down-sampling and up-sampling blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # Normalizing the `only_cross_attention` and `attention_head_dim` parameters across the blocks
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # Initialize the down-sampling layers
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # Create down-sampling block based on its type and configuration
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

        # mid
        # Initialize the mid-block of the UNet
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )

        # count how many layers upsample the videos
        # Prepare for constructing up-sampling layers
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        # Initialize the up-sampling layers
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        # Initialize the output convolutional layers
        self.conv_norm_out = InflatedGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups,
                                               eps=norm_eps)

        self.conv_act = nn.SiLU()  # Using SiLU (swish) as the activation function
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        
        
        # Determine the overall upscaling factor for the U-Net
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False  # Determines if the upsample size needs to be forwarded
        upsample_size = None  # Holds the target upsample size if necessary
        
        # prepare attention_mask
        # Prepare the attention mask for cross-attention layers
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        dtype = torch.float64

        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])  # Expand to match batch size

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        
        
        # Generate time embeddings from the provided timestep
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        
        
        # Process class embeddings if applicable
        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        # Initial convolution layer on the input
        sample = self.conv_in(sample)

        # down
        # Down-sampling path
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # mid
        # Mid block processing
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        # up
        # Up-sampling path
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            upsample_kwargs = {
                "hidden_states": sample,
                "temb": emb,
                "res_hidden_states_tuple": res_samples,
                "upsample_size": upsample_size
            }

            # Include additional arguments based on the presence of cross attention
            if getattr(upsample_block, "has_cross_attention", False):
                upsample_kwargs["encoder_hidden_states"] = encoder_hidden_states
                upsample_kwargs["attention_mask"] = attention_mask

            # Call the upsample block with the constructed arguments
            sample = upsample_block(**upsample_kwargs)

        # post-process
        # Post-processing steps: normalization, activation, and final convolution
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return UNet3DConditionOutput(sample=sample)
    

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
        # Construct the full path to the pretrained model directory
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        # Load the configuration file from the pretrained model directory
        config_file = os.path.join(pretrained_model_path, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)
            
        # Update the class name in the configuration to match the current class    
        config["_class_name"] = cls.__name__
        
        # Define the block types for down and up sampling in the U-Net model
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        # Specify the mid-block type for the U-Net model
        config['mid_block_type'] = 'UNetMidBlock3DCrossAttn'

        from diffusers.utils import WEIGHTS_NAME
        # Load the model with the updated configuration
        model = cls.from_config(config)
        # Load the pretrained weights
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        state_dict = torch.load(model_file, map_location="cpu")
        # origin_state_dict = torch.load('/root/code/Tune-A-Video/checkpoints/stable-diffusion-v1-4/unet/diffusion_pytorch_model.bin', map_location='cpu')
        
        # Update the model's state dict with the loaded weights
        for k, v in model.state_dict().items():
            if '_temp.' in k:
                state_dict.update({k: v})


        # Apply the loaded weights to the model
        model.load_state_dict(state_dict)

        return model