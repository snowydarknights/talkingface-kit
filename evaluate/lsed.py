import torch
import numpy as np
from scipy.ndimage import sobel

# 计算局部的均方误差（MSE）
def calculate_mse(image1, image2, size=64):
    _, height, width = image1.shape
    #print(f"Height: {height}, Width: {width}")  # 打印图像的高度和宽度
    mse = 0
    count = 0

    for i in range(0, height - size, size):
        for j in range(0, width - size, size):
            patch1 = image1[i:i+size, j:j+size]
            patch2 = image2[i:i+size, j:j+size]

            # 确保 patch1 和 patch2 是 NumPy 数组类型
            if isinstance(patch1, torch.Tensor):
                patch1 = patch1.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            if isinstance(patch2, torch.Tensor):
                patch2 = patch2.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组

            mse += np.sum((patch1 - patch2) ** 2) / (size * size)
            count += 1

    mse_value = mse / count if count > 0 else 0
    #print(f"MSE: {mse_value}")  # 打印 MSE 值
    return mse_value

# 计算图像的局部纹理（使用 Sobel 算子计算梯度）
def calculate_texture(image, size=64):
    _, height, width = image.shape
    texture = np.zeros((height, width))

    for i in range(size, height - size):
        for j in range(size, width - size):
            patch = image[:,i-size:i+size, j-size:j+size]
            #print(f"Patch shape: {patch.shape}")  # 打印 patch 的形状
            patch = patch.cpu().numpy()
            # 使用 Sobel 算子计算梯度，提取纹理信息
            grad_x = sobel(patch, axis=1, mode='reflect')
            grad_y = sobel(patch, axis=2, mode='reflect')
            #print(f"Grad x shape: {grad_x}, Grad y shape: {grad_y}")  # 打印梯度图的形状
            texture[i, j] = np.sum(np.abs(grad_x) + np.abs(grad_y)) / (size * size)  # 计算纹理强度
    #print(f"Texture: {np.sum(texture)}")  # 打印纹理的总值
    return texture

# 计算 LSE-D 值
def calculate_lsed(image1, image2, size=64):
    # 计算均方误差（MSE）
    mse = calculate_mse(image1, image2, size)

    # 计算图像纹理差异
    texture1 = calculate_texture(image1, size)
    texture2 = calculate_texture(image2, size)

    # 使用纹理差异加权 MSE
    texture_diff = np.sum(np.abs(texture1 - texture2))
    weighted_mse = mse * (texture_diff + 1e-6)  # 添加小常数避免除零错误
    return weighted_mse