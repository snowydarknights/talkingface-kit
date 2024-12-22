import cv2
import numpy as np
import os
from scipy.ndimage import convolve
from scipy.special import gamma

from utils import reorder_image, to_y_channel, imresize


def estimate_aggd_param(block):
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = np.mean(np.abs(block))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    
    # 基于r_gam和rhatnorm的最小距离估算alpha和beta
    alpha = gam[np.argmin((r_gam - rhatnorm)**2)]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return alpha, beta_l, beta_r


def compute_feature(block):
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    features = [alpha, (beta_l + beta_r) / 2]

    for shift in [[0, 1], [1, 0], [1, 1], [1, -1]]:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        features.extend([alpha, mean, beta_l, beta_r])
    
    return features


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    assert img.ndim == 2
    
    # 将图像裁剪成块
    h, w = img.shape
    num_block_h = h // block_size_h
    num_block_w = w // block_size_w
    img = img[:num_block_h * block_size_h, :num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        features = [
            compute_feature(img_normalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                         idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale])
            for idx_h in range(num_block_h) for idx_w in range(num_block_w)
        ]
        distparam.append(np.array(features))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True) * 255.

    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    cov_distparam = np.cov(distparam[~np.isnan(distparam).any(axis=1)], rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(np.matmul((mu_pris_param - mu_distparam), invcov_param), (mu_pris_param - mu_distparam).T)

    return np.sqrt(float(np.squeeze(quality)))


def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y', **kwargs):
    # 加载预先计算的NIQE参数
    niqe_pris_params = np.load("./niqe_pris_params.npz")
    mu_pris_param, cov_pris_param, gaussian_window = niqe_pris_params['mu_pris_param'], niqe_pris_params['cov_pris_param'], niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    img = img.round()
    return niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
