# Assuming unet.py is in the same directory or in the Python path
from freebloom.models.unet import UNet3DConditionModel
import torch

def run_model():
    # Parameters
    batch_size = 1
    channels = 4
    depth, height, width = 64, 64, 64  # Adjust these dimensions based on your model's requirements

    # Initialize the model
    model = UNet3DConditionModel(
        in_channels=channels,
        out_channels=channels,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
        mid_block_type="UNetMidBlock3DCrossAttn",
        num_class_embeds=10  # Example parameter
    )

    # Create a dummy input tensor
    input_tensor = torch.rand(batch_size, channels, depth, height, width)

    # Get the model output
    output = model(sample=input_tensor, timestep=torch.tensor([0.5]))

    # Print the output shape
    print(f"Output shape: {output.sample.shape}")

if __name__ == "__main__":
    run_model()
