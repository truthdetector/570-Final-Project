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

    def test_rho(lambda_=1.0, model_out_type_='pred_xstart', zeta=0.1):
        model_out_type = model_out_type_

        img_path = util.get_image_paths(L_path)[0]

        good_img = util.imread_uint(img_path, n_channels=3)

        mask_gen = mask_generator(mask_type='box', mask_len_range=[128, 129],mask_prob_range=[0.5, 0.5])
        np.random.seed(seed=0)
        mask = mask_gen(util.uint2tensor4(good_img)).numpy()
        mask = np.squeeze(mask)
        mask = np.transpose(mask, (1, 2, 0))

        #put the mask over the good image to create the image that needs to be in-painted
        degraded_img = good_img * mask / 255.

        np.random.seed(seed=0)  # for reproducibility
        degraded_img = degraded_img * 2 - 1
        degraded_img += np.random.normal(0, 0, degraded_img.shape)  # add AWGN
        degraded_img = degraded_img / 2 + 0.5
        degraded_img = degraded_img * mask

        y = util.single2tensor4(degraded_img).to(device)  # (1,3,256,256)
        y = y * 2 - 1  # [-1,1]
        mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # for y with given noise level, add noise from t_y
        t_y = utils_model.find_nearest(reduced_alpha_cumprod, 0)
        sqrt_alpha_effective = sqrt_alphas_cumprod[1000 - 1] / sqrt_alphas_cumprod[t_y]
        x = sqrt_alpha_effective * y + torch.sqrt(sqrt_1m_alphas_cumprod[1000 - 1] ** 2 - \
                                                  sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[
                                                      t_y] ** 2) * torch.randn_like(y)

        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(1000):
            sigmas.append(reduced_alpha_cumprod[1000 - 1 - i])
            sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
            rhos.append(lambda_ * (0.001 ** 2) / (sigma_ks[i] ** 2))

        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(
            sigma_ks).to(device)

        progress_img = []
        # create sequence of timestep for sampling
        seq = np.sqrt(np.linspace(0, 1000 ** 2, 20))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
        progress_seq = seq[::(len(seq) // 10)]
        progress_seq.append(seq[-1])

        # reverse diffusion for one image from random noise
        for i in range(len(seq)):
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            # time step associated with the noise level sigmas[i]
            t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
            # skip iters
            if t_i > 1000:
                continue

            if model_out_type == 'pred_xstart':
                x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255, model_out_type=model_out_type,
                                          model_diffusion=model, diffusion=diffusion, ddim_sample=False,
                                          alphas_cumprod=alphas_cumprod)

            testingtesting = True
            if (testingtesting):
                if model_out_type == 'pred_xstart':
                    # when noise level less than given image noise, skip
                    if i < 1000:
                        x0_p = (mask * y + rhos[t_i].float() * x0).div(mask + rhos[t_i])
                        x0 = x0 + 1.0 * (x0_p - x0)
                    else:
                        model_out_type = 'pred_x_prev'
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255,
                                                  model_out_type=model_out_type,
                                                  model_diffusion=model, diffusion=diffusion,
                                                  ddim_sample=False, alphas_cumprod=alphas_cumprod)
                        pass
            else:
                # TODO: first order solver
                # x = x - 1 / (2*rhos[t_i]) * (x - y_t) * mask
                pass

            if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1]):

                t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy())
                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = 0.0
                x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                            torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma ** 2) * eps \
                            + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[
                        t_im1] * torch.randn_like(x)

            sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
            x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i] ** 2 - \
                                                          sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[
                                                              t_im1] ** 2) * torch.randn_like(x)

            x_0 = (x / 2 + 0.5)

        x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

        img_E = util.tensor2uint(x_0)
        output_path = os.path.join(E_path, 'output_img.png')
        util.imsave(img_E, output_path)
        output_path_bad = os.path.join(E_path, 'output_img_mask.png')
        util.imsave(util.single2uint(degraded_img), output_path_bad)

    lambda_ = 0.1  # Set a fixed value for lambda
    zeta = 1.0  # Set a fixed value for zeta
    test_rho(lambda_, zeta=zeta)  # Call with fixed parameters

    print("can't believe it runs")

if __name__ == '__main__':

    main()