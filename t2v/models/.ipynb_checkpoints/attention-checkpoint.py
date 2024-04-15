# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import os
import matplotlib.pyplot as plt
import datetime

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
# from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat
import random

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers


    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                    self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask,
                                       video_length=video_length) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            hidden_states = (
                    self.attn2(
                        norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask
                    )
                    + hidden_states
            )
            

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class SparseCausalAttention(CrossAttention):


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        print(sequence_length)

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query_2 = query.clone()
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)
        # print(f"shape of query: {query.shape}")

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        key_2 = key.clone()
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        # print(f"shape of key: {key.shape}")

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)

            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # Compute raw attention scores
        attention_scores = torch.matmul(query_2, key_2.transpose(-2, -1))
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        selected_heads = [2]#[0, 1, 2, 3]
        output_folder = './attention_images'
        os.makedirs(output_folder, exist_ok=True)
        
        for head_index in selected_heads:
            attention_matrix = attention_probs[head_index].detach().cpu().numpy()
            import numpy as np
            attention_matrix_log = np.log(attention_matrix + 1e-9)

            plt.figure(figsize=(10, 8))
            plt.imshow(attention_matrix, cmap='hot') #viridis
            plt.colorbar()
            # plt.title(f'Denoising Step {step_number} - Attention Matrix for Head {head_index}')
            plt.title(f'Attention Matrix for Head {head_index}')
            plt.xlabel('Key positions (frame/context)')
            plt.ylabel('Query positions (current frame)')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the figure with unique filenames for each denoising step and head
            fig_filename = os.path.join(output_folder, f'head_{head_index}_{timestamp}.png')
            plt.savefig(fig_filename)
            plt.close()

        return hidden_states

# class SparseCausalAttention(CrossAttention):

    
#     def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
#         batch_size, sequence_length, _ = hidden_states.shape

#         encoder_hidden_states = encoder_hidden_states

#         if self.group_norm is not None:
#             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        
#         query = self.to_q(hidden_states)
#         dim = query.shape[-1]
#         query = self.reshape_heads_to_batch_dim(query)
        
#         encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#         key = self.to_k(encoder_hidden_states)
#         value = self.to_v(encoder_hidden_states)
        
#         attn_map = vis_attn(q=self.to_q(hidden_states),k=key,v=value)

#         former_frame_index = torch.arange(video_length) - 1
#         former_frame_index[0] = 0
        
#         former2_frame_index = torch.arange(video_length) - 2
#         former2_frame_index[0] = 0
#         former2_frame_index[1] = 0
        
#         former3_frame_index = torch.arange(video_length) - 3
#         former3_frame_index[0] = 0
#         former3_frame_index[1] = 0
#         former3_frame_index[2] = 0
        
#         former4_frame_index = torch.arange(video_length) - 4
#         former4_frame_index[0] = 0
#         former4_frame_index[1] = 0
#         former4_frame_index[2] = 0
#         former4_frame_index[3] = 0
        
#         rn = random.random()
        
#         key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        
#         if rn <= .5:
#             key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index], key[:, former2_frame_index],key[:, former3_frame_index],
#                           key[:, former4_frame_index]], dim=2) 
#         else:
#             key = torch.cat([key[:, former_frame_index], key[:, former2_frame_index],key[:, former3_frame_index],
#                           key[:, former4_frame_index]], dim=2)
            
#         # key = torch.cat([key[:, [0] * video_length]*.5,key[:, former_frame_index],key[:, former2_frame_index],key[:, former3_frame_index],
#         #      key[:, former4_frame_index]], dim=2)
        
#         key = rearrange(key, "b f d c -> (b f) d c")


        
#         value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        
#         if rn <= .5:
#             value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index], value[:, former2_frame_index],value[:, former3_frame_index],
#                           value[:, former4_frame_index]], dim=2) 
#         else:
#             value = torch.cat([value[:, former_frame_index], value[:, former2_frame_index],value[:, former3_frame_index],
#                           value[:, former4_frame_index]], dim=2) 
            
#         # value = torch.cat([value[:, [0] * video_length]*.5, value[:, former_frame_index], value[:, former2_frame_index],value[:, former3_frame_index],
#         #                   value[:, former4_frame_index]], dim=2) 
#         value = rearrange(value, "b f d c -> (b f) d c")

#         key = self.reshape_heads_to_batch_dim(key)
#         value = self.reshape_heads_to_batch_dim(value)
        

#         if attention_mask is not None:
#             if attention_mask.shape[-1] != query.shape[1]:
#                 target_length = query.shape[1]
#                 attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                 attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

#         # attention, what we cannot get enough of
#         if self._use_memory_efficient_attention_xformers:
#             hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
#             # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#             hidden_states = hidden_states.to(query.dtype)
#         else:
#             if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#                 hidden_states = self._attention(query, key, value, attention_mask)
#             else:
#                 hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

#         # linear proj
#         hidden_states = self.to_out[0](hidden_states)

#         # dropout
#         hidden_states = self.to_out[1](hidden_states)
#         return hidden_states

    
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# head_counter = 0

# def vis_attn(q, k, v):
#     global head_counter
#     head_counter += 1
#     if head_counter % 16 == 2:
#         print(f"Call number: {head_counter}")
#         print(f"shape of q: {q.shape}")
#         print(f"shape of k: {k.shape}")
#         print(f"shape of v: {v.shape}")
        
#         # Compute attention scores
#         attention_scores = torch.matmul(q, k.transpose(-2, -1))
#         # Normalize scores
#         attention_scores_normalized = torch.nn.functional.softmax(attention_scores, dim=-1)
        
#         # Prepare the directory to save the images
#         save_dir = "attention_maps"
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Visualization (for the first item in the batch for simplicity)
#         attention_matrix_to_plot = attention_scores_normalized[0].detach().cpu().numpy()
#         # attention_matrix_to_plot = attention_scores
#         plt.figure(figsize=(10, 8))
#         plt.imshow(attention_matrix_to_plot, cmap='hot', aspect='auto')
#         plt.colorbar()
#         plt.title('Attention Heatmap')
#         plt.xlabel('Key Sequences')
#         plt.ylabel('Query Sequences')
        
#         # Save as PNG
#         filename = os.path.join(save_dir, f"attention_map_{head_counter}.png")
#         plt.savefig(filename)
#         plt.close()
#         print(f"Saved attention heatmap to {filename}")

# head_counter = 0
# def vis_attn(q, k,v):
#     global head_counter
#     head_counter += 1
#     if head_counter == 16:
#         batch_size, seq_len, dim = q.shape
#         assert q.shape == k.shape, "Shapes of q and k must be the same"

#         # Create a figure with constrained layout
#         fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), constrained_layout=True)
#         fig.suptitle('Attention Heatmaps')

#         for i in range(batch_size):
#             # Compute the attention matrix for each pair (q_i, k_i)
#             q_i = q[i]  # Shape: (4096, 320)
#             k_i = k[i]  # Shape: (4096, 320)
#             attention_scores = torch.matmul(q_i, k_i.transpose(0, 1))  # Shape: (4096, 4096)

#             # Normalize for visualization purposes
#             attention_scores_normalized = torch.nn.functional.softmax(attention_scores, dim=-1)
#             attention_matrix = attention_scores_normalized.detach().cpu().numpy()

#             # Plot each heatmap in a grid
#             ax = axes[i // 4, i % 4]  # Determine the position in the grid
#             cax = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
#             ax.set_title(f'Heatmap {i+1}')
#             ax.axis('off')  # Hide axes to make it cleaner

#         # Add a color bar
#         fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal')
#         plt.savefig('combined_attention_heatmaps.png')
#         plt.close()
#         print("Saved combined heatmap to 'combined_attention_heatmaps.png'")
        
        
