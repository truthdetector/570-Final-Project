#Imports copy and pasted from the given file

import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import OrderedDict

#the utils files are the most important part of the imports
#they are imports that were included and modified by the original authors

from utils import utils_model
from utils import utils_logger
from utils import utils_image as util
from utils.utils_inpaint import mask_generator

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


# Define a wrapper for the model to reduce output channels
class ModifiedModel(nn.Module):
    def __init__(self, base_model):
        super(ModifiedModel, self).__init__()
        self.base_model = base_model
        self.final_conv = nn.Conv2d(6, 3, kernel_size=1)  # 1x1 convolution to reduce channels

    def forward(self, x, t):
        x = self.base_model(x, t)  # Get the 6-channel output
        x = self.final_conv(x)     # Convert to 3 channels
        return x


def main():
    #keeping mostly the same format for model loading as I'm using their util functions
    #and the given model for this implementation
    model_config = dict(
        model_path=os.path.join(os.path.join('', 'model_zoo'), 'diffusion_ffhq_10m'+'.pt'),
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    )

    #creating an argument parser, resulting with args holding the configuration settings
    args = utils_model.create_argparser(model_config).parse_args([])
    #defintion of main model used for inpainting, with diffusion specified for the individual steps
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    #loading the pre-specified weights rather than random weights
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    #guranteeing consistent results for inference
    model.eval()

    for k, v in model.named_parameters():
       v.requires_grad = False
    # model = ModifiedModel(model)
    #we need to determine whether or not a GPU is available as the device for the code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    #this points to the testsets, results, and testsets folders respectively in the directory
    L_path                  = os.path.join('', 'testsets')      # L_path, for Low-quality images
    E_path                  = os.path.join('', 'results')       # E_path, for Estimated images

    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, 1000, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    def test_rho(lambda_value=1.0, model_output_type='pred_xstart', regularization_strength=0.1):
        # Set model output type
        model_output = model_output_type

        # Load the target image and create a degraded version with masking
        image_path = util.get_image_paths(L_path)[2]
        target_image = util.imread_uint(image_path, n_channels=3)

        # Initialize the mask generator
        mask_gen = mask_generator(mask_type='extreme', mask_len_range=[128, 129], mask_prob_range=[0.5, 0.5])
        np.random.seed(0)
        mask = mask_gen(util.uint2tensor4(target_image)).numpy()
        mask = np.squeeze(mask)
        mask = np.transpose(mask, (1, 2, 0))

        # Degrade the image using the generated mask
        masked_image = target_image * mask / 255.0

        # Normalize and add noise to the degraded image
        np.random.seed(0)
        masked_image = (masked_image * 2) - 1  # Normalize to [-1, 1]
        masked_image += np.random.normal(0, 0, masked_image.shape)  # Add Gaussian noise
        masked_image = (masked_image / 2) + 0.5  # Scale back to [0, 1]
        masked_image = masked_image * mask

        # Convert images to tensors for inpainting
        degraded_tensor = util.single2tensor4(masked_image).to(device)
        degraded_tensor = (degraded_tensor * 2) - 1  # Normalize again to [-1, 1]
        mask_tensor = util.single2tensor4(mask.astype(np.float32)).to(device)

        # Add noise to the initial degraded image based on diffusion parameters
        initial_noise_level = utils_model.find_nearest(reduced_alpha_cumprod, 0)
        effective_alpha = sqrt_alphas_cumprod[1000 - 1] / sqrt_alphas_cumprod[initial_noise_level]
        x = effective_alpha * degraded_tensor + torch.sqrt(sqrt_1m_alphas_cumprod[1000 - 1] ** 2 - \
                                                           effective_alpha ** 2 * sqrt_1m_alphas_cumprod[
                                                               initial_noise_level] ** 2) * torch.randn_like(
            degraded_tensor)

        # Set up diffusion parameters
        noise_schedule = []
        regularization_weights = []
        for i in range(1000):
            noise_schedule.append(reduced_alpha_cumprod[1000 - 1 - i])
            regularization_weights.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]) * lambda_value)

        noise_schedule, regularization_weights = torch.tensor(noise_schedule).to(device), torch.tensor(
            regularization_weights).to(device)

        # Define the timestep sequence for the sampling process
        seq = np.sqrt(np.linspace(0, 1000 ** 2, 20)).astype(int)
        seq = [int(s) for s in seq]
        seq[-1] = seq[-1] - 1  # Ensure the last value is within bounds

        # Progressively inpaint the image through the reverse diffusion process
        for i in range(len(seq)):
            curr_sigma = noise_schedule[seq[i]].cpu().numpy()
            t_step = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)

            if t_step > 1000:
                continue

            if model_output == 'pred_xstart':
                predicted_x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255, model_out_type=model_output,
                                                    model_diffusion=model, diffusion=diffusion, ddim_sample=False,
                                                    alphas_cumprod=alphas_cumprod)

            # Apply regularization and plug-and-play inpainting for missing regions
            if model_output == 'pred_xstart' and i < 1000:
                predicted_x0 = (
                            mask_tensor * degraded_tensor + regularization_weights[t_step].float() * predicted_x0).div(
                    mask_tensor + regularization_weights[t_step])
            else:
                model_output = 'pred_x_prev'
                predicted_x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255, model_out_type=model_output,
                                                    model_diffusion=model, diffusion=diffusion, ddim_sample=False,
                                                    alphas_cumprod=alphas_cumprod)

            # Diffusion-based image update
            if i < len(seq) - 1:  # Ensure that i + 1 is within bounds
                next_noise_level = utils_model.find_nearest(reduced_alpha_cumprod,
                                                            noise_schedule[seq[i + 1]].cpu().numpy())
                epsilon = (x - sqrt_alphas_cumprod[t_step] * predicted_x0) / sqrt_1m_alphas_cumprod[t_step]
                eta_sigma = 0.0
                x = sqrt_alphas_cumprod[next_noise_level] * predicted_x0 + np.sqrt(1 - regularization_strength) * (
                        torch.sqrt(sqrt_1m_alphas_cumprod[next_noise_level] ** 2 - eta_sigma ** 2) * epsilon \
                        + eta_sigma * torch.randn_like(x)) + np.sqrt(regularization_strength) * sqrt_1m_alphas_cumprod[
                        next_noise_level] * torch.randn_like(x)

        # Apply final masked areas back to original regions
        x[mask_tensor.to(torch.bool)] = degraded_tensor[mask_tensor.to(torch.bool)]

        # Save the resulting image
        final_output = util.tensor2uint(x)
        util.imsave(final_output, os.path.join(E_path, 'inpainted_output.png'))

    lambda_ = 0.1  # Set a fixed value for lambda
    regularization_strength = 1.0  # Set a fixed value for zeta
    test_rho(lambda_, regularization_strength=regularization_strength)  # Call with fixed parameters

    print("can't believe it runs")

if __name__ == '__main__':

    main()