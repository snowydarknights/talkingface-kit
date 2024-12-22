import cv2
import argparse
import os
import numpy as np
import dlib
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# 加载 Dlib 的人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def extract_lip_features(frame):

    #从一帧图像中提取嘴部关键点的特征并将其转换为非负值的展平一维向量

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        lip_points = []
        for i in range(48, 68):  # 嘴部关键点为 48-67
            lip_points.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # 转为 NumPy 数组并计算非负化嘴部特征
        lip_points = np.array(lip_points)
        lip_points_positive = lip_points - np.min(lip_points, axis=0)  # 平移到非负范围
        
        #print(lip_points_positive.flatten())  # 打印处理后的坐标
        return lip_points_positive.flatten()  # 转换为一维向量  20 x 2==40
    return None

def extract_audio_features(audio_file, sr=16000, target_dim=40, frame_duration=0.04):
 
    #提取音频文件的 Mel 频谱特征，并通过 PCA 降维到指定维度

    # 加载音频
    y, _ = librosa.load(audio_file, sr=sr)
    
    # 计算 Mel 频谱
    hop_length = int(sr * frame_duration)  # 每帧步长
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length)
    mel_frames = mel_spectrogram.T  # 转置为 (num_frames, n_mels)
    
    # PCA 降维
    pca = PCA(n_components=target_dim)
    mel_reduced = pca.fit_transform(mel_frames)  # 降维后的特征 (num_frames, target_dim)
    
    # 平移到非负范围
    mel_reduced_positive = mel_reduced - np.min(mel_reduced, axis=0)  # 将每列最小值移到零

    # 归一化到 [0, 1] 范围
    mel_reduced_normalized = mel_reduced_positive / np.max(mel_reduced_positive, axis=0)  # 按每列最大值归一化

    # 每行归一化到范数为 1
    mel_reduced_final = []
    for row in mel_reduced_normalized:
        norm = np.linalg.norm(row)  # 计算行的 L2 范数
        if norm == 0:
            mel_reduced_final.append(row)  # 如果范数为 0，不做变化
        else:
            mel_reduced_final.append(row / norm)  # 将行归一化到范数为 1
    mel_reduced_final = np.array(mel_reduced_final)  # 转为 NumPy 数组

    return mel_reduced_final  # 返回处理后的音频特征


def calculate_lip_sync_accuracy(lip_features, audio_features):

    #计算嘴部特征与音频特征的一维余弦相似度

    similarities = []
    for i in range(len(lip_features)):
        # 保证特征是平展的一维向量
        lip_vector = lip_features[i].flatten()
        #print(lip_vector)
        audio_vector = audio_features[i].flatten()
        #print(audio_vector)

        # 计算余弦相似度
        dot_product = np.dot(lip_vector, audio_vector)  # 向量点积
        #print(dot_product)
        norm_lip = np.linalg.norm(lip_vector)  # 嘴部特征的范数
        #print(norm_lip)
        norm_audio = np.linalg.norm(audio_vector)  # 音频特征的范数
        #print(norm_audio)

        # 避免分母为 0
        if norm_lip == 0 or norm_audio == 0:
            sim = 0
        else:
            sim = dot_product / (norm_lip * norm_audio)

        similarities.append(sim)
        print(f"Frame {i}: Cosine similarity = {sim:.2f}")  # 打印逐帧相似度

    return np.mean(similarities)  # 返回平均相似度


def process(video_file, audio_file):
    
    #主处理函数：从视频和音频中提取特征并计算同步准确度
    
    # 提取视频帧嘴部特征
    lip_features = []
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lip_feature = extract_lip_features(frame)
        if lip_feature is not None:
            lip_features.append(lip_feature)
    cap.release()

    # 提取音频特征
    audio_features = extract_audio_features(audio_file)

    # 对齐特征长度
    min_length = min(len(lip_features), len(audio_features))
    lip_features = lip_features[:min_length]
    audio_features = audio_features[:min_length]

    # 检查对齐后的特征
    if len(lip_features) == 0 or len(audio_features) == 0:
        #//输入无效
        print("输入无效视频，或者音频。")
        return

    # 归一化嘴部特征
    lip_features = np.array([normalize(f.reshape(1, -1)) for f in lip_features])

    # 计算口型同步准确度
    accuracy = calculate_lip_sync_accuracy(lip_features, audio_features)
    print(f"\nLip Sync Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取视频和音频特征并计算口型同步准确度")
    parser.add_argument("video_file", help="输入的视频文件路径")
    parser.add_argument("audio_file", help="输入的音频文件路径")
    args = parser.parse_args()

    # 主处理
    process(args.video_file, args.audio_file)
