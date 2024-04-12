# Text-to-Video Generation Using Stable Diffusion

This project implements a system for generating videos from textual descriptions using a modified UNet architecture that integrates temporal dimensions for frame coherence.

## Setup

### Requirements

To set up the required environment:

```bash
conda create -n fb python=3.10
conda activate fb
pip install -r requirements.txt

Installing xformers is highly recommended for enhanced efficiency and performance on GPUs. To enable xformers, set enable_xformers_memory_efficient_attention=True in the configuration (default is set to True).

Data Setup
The data folder should contain the Stable Diffusion model files, which can be downloaded from the Hugging Face website. These files are essential for the model's operation.


Citation
If you use this implementation or the underlying concepts in your work, please cite the following paper:

bibtex
Copy code
@article{freebloom,
  title={Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator},
  author={Huang, Hanzhuo and Feng, Yufan and Shi, Cheng and Xu, Lan and Yu, Jingyi and Yang, Sibei},
  journal={arXiv preprint arXiv:2309.14494},
  year={2023}
}
arduino
Copy code

This README provides clear instructions on setting up the environment, running the model, and acknowledges the use of pre-trained Stable Diffusion files, ensuring users understand how to operate the project effectively.





