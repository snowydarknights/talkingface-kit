import numpy as np
import torch

# 计算局部的均方误差（MSE）
def calculate_mse(image1, image2, size=64):
    _, height, width = image1.shape
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

    return mse / count if count > 0 else 0

# 计算局部曲率（使用二阶导数）
def calculate_curvature(image, size=64):
    _, height, width = image.shape
    curvature = np.zeros((height, width))

    for i in range(size, height - size):
        for j in range(size, width - size):
            patch = image[:,i-size:i+size, j-size:j+size]
            #print(f"Patch shape: {patch.shape}")  # 打印 patch 的形状
            patch = patch.cpu().numpy()
            # 计算二阶导数作为曲率估计
            dx = np.gradient(patch, axis=1)
            dy = np.gradient(patch, axis=2)
            dxx = np.gradient(dx, axis=1)
            dyy = np.gradient(dy, axis=2)
            curvature[i, j] = np.sum(dxx + dyy) / (size * size)
    
    return curvature

# 计算 LSE-C 值
def calculate_lsec(image1, image2, size=64):
    # 计算均方误差（MSE）
    mse = calculate_mse(image1, image2, size)

    # 计算曲率
    curvature1 = calculate_curvature(image1, size)
    curvature2 = calculate_curvature(image2, size)

    # 使用局部曲率加权 MSE
    weighted_mse = mse * (np.sum(np.abs(curvature1 - curvature2)) + 1e-6)  # 添加小常数避免除零错误
    return weighted_mse