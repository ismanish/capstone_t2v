import os
import json
import torch
from datetime import datetime
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Resize, ToPILImage
import matplotlib.pyplot as plt


# to get current timestamp
def get_time_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Store and Manage Attention
class AttentionStore:
    def __init__(self, disk_store = True,store_dir = None, config = None):
        # Counter to track the current processing step
        self.current_step = 0
        # Flag to determine if data should be stored on disk
        self.disk_store = disk_store
        # Directory where attention data will be stored, if disk_store is True
        self.store_dir = store_dir or self._setup_store_directory()
        # Dictionary to store latent representations at each step
        self.latent_store = {}
        # Dictionary to store attention matrices at each step
        self.step_store = {}
        # Optional configuration object
        self.config = config
        
    # Private method to setup the directory for storing attention data on disk
    def _setup_store_directory(self):
        # Check if disk storage is enabled
        if self.disk_store:
            # Get the current time as a string to create a unique directory name
            time_string = get_time_string()
            # Construct the directory path where attention data will be stored
            directory = os.path.join('./temp', f'attention_cache_{time_string}')
            # Create the directory, including any necessary parent directories
            os.makedirs(directory, exist_ok=True)
            os.makedirs("./temp/final", exist_ok=True) # new
            # Return the created directory path
            return directory
        # Return None if disk storage is not enabled
        return None
    
    
    # Method called at each step of the processing to update the attention data
    def step_callback(self, x_t, inner_idx = None, momentum=0.1):
        # Increment the step counter
        self.current_step += 1
        # Assuming we want to store x_t as part of step_store for some reason
        self.step_store[self.current_step] = x_t.clone()
        # Perform interpolation on the attention data if inner indices are provided
        if inner_idx:
            x_t = self.interpolate_attention(x_t, inner_idx, momentum)
        # Save the current step's attention data to disk if disk storage is enabled
        if self.disk_store and self.store_dir:
            self._save_step_store()
        # Return the updated latent representation
        return x_t
    
    
    # Method to interpolate attention between frames, given a set of indices and a momentum term
    def interpolate_attention(self,x_t,inner_idx, momentum):
        # Loop over each index where interpolation is needed
        for idx in inner_idx:
            # Find the neighboring frames for interpolation
            pre_idx, next_idx = self._find_neighbours(x_t, idx)
            # Compute the interpolation weight (alpha) based on the position of idx
            alpha = (idx - pre_idx)/ (next_idx - pre_idx)
            # Apply the interpolation to the latent representation at the current index
            x_t[:,:,idx] = self._apply_interpolation(x_t,pre_idx,next_idx,idx,alpha,momentum)
            # Return the updated latent representation with interpolated frames
        return x_t
    
    # Method to find the neighboring indices for a target index, to be used in interpolation
    def _find_neighbours(self,x_t, target_idx):
        # Create a list of indices excluding the target index
        original_idx = [i for i in range(x_t.shape[2]) if i not in target_idx]
        # Sort the combined list of original indices and the target index
        sorted_idx = sorted(original_idx + [target_idx])
        # Find the position of the target index in the sorted list
        target_position = sorted_idx.index(target_idx)
        # Return the indices immediately before and after the target index
        return sorted_idx[target_position - 1], sorted_idx[target_position + 1]
    
    # Method to apply interpolation between two frames at specified indices, with a given blending factor
    def _apply_interpolation(self, x_t, pre_idx, next_idx, idx, alpha, momentum):
        # Calculate the interpolated value based on the alpha blending factor and momentum
        return (1 - momentum) * ((next_idx - idx) / (next_idx - pre_idx) * x_t[:, :, pre_idx] + alpha * x_t[:, :, next_idx]) + momentum * x_t[:, :, idx]

    
#     # Method to save the current step's attention data to disk
#     def _save_step_store(self):
#         # Check if disk storage is enabled and the store directory is set
#         if not self.disk_store or not self.store_dir:
#             return

#         # Initialize a transform to resize images to nxn pixels
#         resize_transform = Resize((64, 64))
#         # Initialize a transform to convert tensors to PIL images
#         to_pil_transform = ToPILImage()

#         # Iterate over each tensor in the step store
#         for key, tensor in self.step_store.items():
#             # Skip tensors that are not 4D (we expect tensors of shape batch x channels x height x width)
#             if tensor.dim() < 4:
#                 continue

#             # Extract batch size and number of channels from tensor shape
#             batch_size, channels, *spatial_dims = tensor.shape

#             # Iterate over the batch dimension
#             for b in range(batch_size):
#                 # Iterate over the frame/time dimension (assuming it's the 3rd dimension)
#                 for frame_idx in range(tensor.size(2)):
#                     # Extract the specific frame from the tensor
#                     frame_tensor = tensor[b, :, frame_idx]

#                     # If tensor has only one channel, repeat it 3 times to convert to RGB
#                     if channels == 1:
#                         frame_tensor = frame_tensor.repeat(3, 1, 1)
#                     # If tensor has more than 3 channels, keep only the first 3
#                     elif channels > 3:
#                         frame_tensor = frame_tensor[:3]

#                     # Ensure tensor values are in the [0, 1] range
#                     frame_tensor = torch.clamp(frame_tensor, 0, 1)
#                     # Convert the tensor to a PIL image
#                     pil_img = to_pil_transform(frame_tensor)

#                     # Resize the image to nxn pixels
#                     resized_img = resize_transform(pil_img)

#                     # Construct the file path where the image will be saved
#                     frame_path = os.path.join(self.store_dir, f'{self.current_step:03d}_{frame_idx}_.png')
#                     # Save the resized image
#                     resized_img.save(frame_path)

#         # Clear the step store after saving its contents
#         self.step_store = {}

    def _save_step_store(self):
        # print(f"Disk Store: {self.disk_store}, Store Directory: {self.store_dir}")

        if not self.disk_store or not self.store_dir:
            # print("Disk store not enabled or store directory not set.")
            return

        if not self.step_store:
            # print("Step store is empty.")
            return

        for key, tensor in self.step_store.items():
            # print(f"Processing tensor: {key}, shape: {tensor.shape}")

            # Assuming tensor shape is [batch_size, num_channels, num_frames, height, width]
            if tensor.dim() != 5:
                # print(f"Unexpected tensor dimension {tensor.dim()} for {key}, expected 5.")
                continue

            tensor = tensor.detach()

            for b in range(tensor.size(0)):
                for frame_idx in range(tensor.size(2)):
                    frame_tensor = tensor[b, :, frame_idx]
                    frame_path = os.path.join(self.store_dir, f"{key}_{b}_{frame_idx}.pt")
                    frame_path = os.path.join("./temp/final", f"{key}_{b}_{frame_idx}.pt") # new
                    # print(f"Saving frame {frame_idx} of batch {b} to {frame_path}")
                    try:
                        torch.save(frame_tensor, frame_path)
                    except Exception as e:
                        print(f"Error saving frame to {frame_path}: {e}")

        self.step_store = {}
        # print("Step store saved and cleared.")


    
    
    # Method to reset the AttentionStore, clearing all stored data and counters
    def reset(self):
        # Reset the step counter
        self.current_step = 0
        # Clear the dictionaries storing latent and attention data
        self.step_store = {}
        self.latent_store = {}
        
    # Method to clear the cache, removing temporary attention data
    def empty_cache(self):
        # Clear the temporary dictionaries storing step-wise and attention data
        self.step_store = {}
        self.attention_store = {}
        
        
        
        
    

        

