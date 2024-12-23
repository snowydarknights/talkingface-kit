import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Normalize

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 InceptionV3 模型并提取 pool3 层
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
feature_extractor = create_feature_extractor(inception_model, return_nodes={'avgpool': 'pool3'})

# 归一化
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_frames(frames):
    return [normalize(frame) for frame in frames]

# 提取特征
def get_features(frames):
    frames_tensor = torch.stack(frames).to(device)  # 转为张量并移动到设备
    with torch.no_grad():
        features = feature_extractor(frames_tensor)['pool3']  # 提取 pool3 层特征
        return features.view(features.size(0), -1).cpu().numpy()  # 转换为 numpy 数组

# 计算 FID
def calculate_fid(real_features, fake_features):
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 正定性调整
    epsilon = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * epsilon
    sigma_fake += np.eye(sigma_fake.shape[0]) * epsilon

    # 计算 Frechet 距离
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # 检查复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid
