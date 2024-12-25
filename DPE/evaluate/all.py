import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = torch.manual_seed(123)

def psnr_torch(img1, img2):
    img1 = img1.to(device)
    img2 = img2.to(device)
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    print("PNSR: ", 20 * torch.log10(1.0 / torch.sqrt(mse)).mean())
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()

def psnr(img1, img2):
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    metric = PeakSignalNoiseRatio().to(device)
    print("PNSR1: ", metric(img1, img2))
    return metric(img1, img2)

def ssim(img1, img2):
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    print("SSIM: ", metric(img1, img2))
    return metric(img1, img2)

def lpips(img1, img2):
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    print("LPIPS: ", metric(img1, img2))
    return metric(img1, img2)
