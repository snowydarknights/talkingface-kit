import cv2
import numpy as np
import dlib
from scipy.stats import entropy
from fastdtw import fastdtw

# 加载 Dlib 的人脸检测器和面部关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # 下载自 Dlib 官方

def normalize_lip_points(lip_points):
    
    #将嘴部关键点坐标归一化到 [0, 1] 范围。
    
    min_vals = np.min(lip_points, axis=0)
    max_vals = np.max(lip_points, axis=0)
    #避免除数说=0
    return (lip_points - min_vals) / (max_vals - min_vals + 1e-8)

def extract_lip_features(frame):
    
    #视频帧中提取嘴部关键点的特征向量。

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #灰度
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        lip_points = []
        for i in range(48, 68):  # 68点模型中，嘴部特征为48-67点
            lip_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return np.array(lip_points)  # 返回归一化后的嘴部关键点
    return None

import numpy as np

def calculate_naturalness_with_windows(face_features, window_size=50):
    
    #计算面部表情的自然度评分，通过滑动窗口方法分析面部表情特征的变化。
    
    # 检查输入数据的有效性
    if len(face_features) < 2:
        raise ValueError("至少需要两帧来计算自然度")
    
    # 初始化自然度评分列表
    scores = []
    
    # 遍历滑动窗口
    for start in range(len(face_features) - window_size + 1):
        # 提取滑动窗口内的特征
        window = face_features[start:start + window_size]

        #print(window)
        
        # 计算相邻帧之间的距离。L2范数。
        distances = np.linalg.norm(np.diff(window, axis=0), ord=2, axis=1)
        
        # 计算窗口内所有相邻帧距离的平均值
        score = np.mean(distances)
        
        # 将评分添加到列表中
        scores.append(score)
    
    # 计算所有窗口自然度评分的平均值
    naturalness_score = np.mean(scores)
    
    return naturalness_score


def calculate_expression_entropy(face_features):
    
    #使用熵度量嘴部运动的多样性。

    deltas = [np.linalg.norm(face_features[i+1] - face_features[i], ord="fro") for i in range(len(face_features) - 1)]
    histogram, _ = np.histogram(deltas, bins=10, density=True)  # 构建直方图，10区间
    return entropy(histogram + 1e-8)  # 计算熵（加小量避免零）

def process_video(video_path, frame_interval=5, window_size=50):
    
    #从视频中提取嘴部关键点并计算自然度评分。

    cap = cv2.VideoCapture(video_path)
    face_features = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:  # 每隔 frame_interval 帧处理一次
            lip_feature = extract_lip_features(frame)
            if lip_feature is not None:
                face_features.append(lip_feature)
        frame_index += 1

    cap.release()

    if len(face_features) < 2:
        raise ValueError("视频中无法检测到足够的面部特征点进行自然度计算。")
    
    # 计算自然度评分和熵
    naturalness_score = calculate_naturalness_with_windows(face_features, window_size=window_size)
    entropy_score = calculate_expression_entropy(face_features)
    
    return {
        "naturalness_score": naturalness_score,
        "entropy_score": entropy_score
    }

if __name__ == "__main__":
    import argparse

    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="计算视频的面部表情自然度评分")
    parser.add_argument("video_file", help="输入视频文件路径")
    args = parser.parse_args()

    # 处理视频并输出自然度评分
    video_file = args.video_file
    try:
        results = process_video(video_file)
        print(f"视频的面部表情自然度评分: {results['naturalness_score']:.4f}")
        print(f"视频的嘴部运动熵: {results['entropy_score']:.4f}")
    except Exception as e:
        print(f"处理视频时出错: {e}")
