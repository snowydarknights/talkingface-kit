from torchvision import transforms
from PIL import Image
import cv2
import niqe
import fid
# import lsec
# import lsed
import torch
import all
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将帧转换为所需格式的图像
def process_frame(frame, size):
    # 转换为 RGB 格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转换为 PIL 图像
    frame_pil = Image.fromarray(frame_rgb)
    # 打印图像信息
    #print(f"Original frame shape: {frame.shape}")
    #print(f"Converted frame shape: {frame_pil.size}")
    # 转换为 RGB 并调整大小
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    # 打印转换后的信息
    #print(f"Transformed frame shape: {transform(frame_pil).shape}")
    return transform(frame_pil).to(device)

# 计算视频的函数
def compute_video(video_1, video_2):
    # 打开视频文件
    cap1 = cv2.VideoCapture(video_1)
    cap2 = cv2.VideoCapture(video_2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video files.")
        return None
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_niqe_1 = 0
    total_niqe_2 = 0
    # total_lsec = 0
    # total_lsed = 0
    frame_count = 0

    frames_1 = []
    frames_2 = []

    while True:
        # 读取帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break  # 如果任一视频结束，则停止
        
        # 处理帧图像
        frames_1.append(process_frame(frame1, 299))
        frames_2.append(process_frame(frame2, 299))
        img1 = process_frame(frame1, 256)
        img2 = process_frame(frame2, 256)

        # 计算PSNR值
        psnr_value = all.psnr_torch(img1, img2)
        # 计算 SSIM 值
        ssim_value = all.ssim(img1, img2)
        # 计算 LPIPS 值
        lpips_value = all.lpips(img1, img2)
        # 计算video1当前帧的NIQE值
        niqe_value_1 = niqe.calculate_niqe(frame1, crop_border=0)
        # 计算video2当前帧的NIQE值
        niqe_value_2 = niqe.calculate_niqe(frame2, crop_border=0)
        # # 计算LSEC值
        # lsec_value = lsec.calculate_lsec(img1, img2)
        # # 计算LSED值
        # lsed_value = lsed.calculate_lsed(img1, img2)
        #print(f"LSEC value: {lsec_value}")
        #print(f"LSED value: {lsed_value}")
        
        # 累加所有值
        total_psnr += psnr_value.item()  
        total_ssim += ssim_value.item()
        total_lpips += lpips_value.item()
        total_niqe_1 += niqe_value_1
        total_niqe_2 += niqe_value_2
        # total_lsec += lsec_value.item()
        # total_lsed += lsed_value.item()
        frame_count += 1

    # 释放视频文件
    cap1.release()
    cap2.release()  

    fid.preprocess_frames(frames_1)
    fid.preprocess_frames(frames_2)
    # 提取每一帧的特征
    features_1 = fid.get_features(frames_1)
    features_2 = fid.get_features(frames_2)
    # 计算 FID
    fid_value = fid.calculate_fid(features_1, features_2)

    if frame_count > 0:
        # 计算所有帧的平均PSNR
        average_psnr = total_psnr / frame_count
        average_ssim = total_ssim / frame_count
        average_lpips = total_lpips / frame_count
        average_niqe_1 = total_niqe_1 / frame_count
        average_niqe_2 = total_niqe_2 / frame_count
        # average_lsec = total_lsec / frame_count
        # average_lsed = total_lsed / frame_count
    
        # 打印结果
        print("PSNR value:", average_psnr)
        print("SSIM value:", average_ssim) 
        print("LPIPS value:", average_lpips)
        print("NIQE value for video 1:", average_niqe_1)
        print("NIQE value for video 2:", average_niqe_2)
        print("FID value:", fid_value)
        # print("LSEC value:", average_lsec)
        # print("LSED value:", average_lsed)
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Compute metrics between two videos.")
    
    # 添加命令行参数，分别获取两个视频文件路径
    parser.add_argument("video_1", type=str, help="Path to the source video.")
    parser.add_argument("video_2", type=str, help="Path to the generated video.")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 获取视频文件路径
    video_1 = args.video_1
    video_2 = args.video_2

    # 调用函数计算视频的相关指标
    compute_video(video_1, video_2)