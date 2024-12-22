import argparse
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# 提取视频中的音频
def extract_audio(video_file, audio_file):
    # 载入视频文件
    video = VideoFileClip(video_file)

    # 提取音频
    audio = video.audio

    # 保存音频文件
    audio.write_audiofile(audio_file)

    # 释放资源
    audio.close()
    video.close()

    print(f"音频已提取并保存到 {audio_file}")

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="提取视频中的音频")

    # 添加命令行参数
    parser.add_argument("video_file", help="输入的视频文件路径")
    parser.add_argument("audio_file", help="音频文件路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 将输入的相对路径转换为绝对路径
    video_file = os.path.abspath(args.video_file)
    audio_file = os.path.abspath(args.audio_file)

    # 提取音频
    extract_audio(video_file, audio_file)

if __name__ == "__main__":
    main()
