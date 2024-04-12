import os
import imageio
import numpy as np
from PIL import Image
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange


def save_tensor_img(img, save_path):
    img = (img * 255).byte().numpy().transpose(1, 2, 0)
    Image.fromarray(img).save(save_path)


# def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
#     videos = rearrange(videos, "b c t h w -> t b c h w")
#     outputs = []
#     for x in videos:
#         x = torchvision.utils.make_grid(x, nrow=n_rows)
#         x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
#         if rescale:
#             x = (x + 1.0) / 2.0  # -1,1 -> 0,1
#         x = (x * 255).numpy().astype(np.uint8)
#         outputs.append(x)
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     imageio.mimsave(path, outputs, fps=fps)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    # Rearrange the videos tensor to shape [time, batch, channels, height, width]
    videos = rearrange(videos, "b c t h w -> t b c h w")

    # Prepare the frames for the GIF
    outputs = [
        torchvision.utils.make_grid(frame, nrow=n_rows).permute(1, 2, 0).numpy()
        for frame in videos
    ]

    # Rescale images to [0, 255] if required
    if rescale:
        outputs = [(frame + 1) / 2 * 255 for frame in outputs]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the frames as an animated GIF
    imageio.mimsave(path, outputs, fps=fps, format='GIF')


def save_videos_per_frames_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Iterate over each video in the batch
    for i, video in enumerate(videos):
        # video shape: [channels, frames, height, width]
        video = rearrange(video, 'c f h w -> f c h w')

        # Create a grid for each frame in the video
        grid = torchvision.utils.make_grid(video, nrow=n_rows)

        # Normalize and convert to numpy array for saving
        if rescale:
            grid = (grid + 1.0) / 2.0  # Normalize to [0, 1]
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        # Ensure grid_np is in uint8 format
        if grid_np.dtype != np.uint8:
            grid_np = (grid_np * 255).astype(np.uint8)

        # Save the grid image
        grid_image = Image.fromarray(grid_np)
        grid_image.save(os.path.join(path, f'{i}_all.jpg'))

        # Save individual frames
        for j, frame in enumerate(video):
            # Normalize and convert to numpy array for saving
            if rescale:
                frame = (frame + 1.0) / 2.0
            frame_np = frame.permute(1, 2, 0).cpu().numpy()

            # Ensure frame_np is in uint8 format
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)

            # Save the frame image
            frame_image = Image.fromarray(frame_np)
            frame_image.save(os.path.join(path, f'{i}_{j}.jpg'))