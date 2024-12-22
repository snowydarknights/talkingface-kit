import argparse
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# 将视频和音频合并
def merge_audio_video(video_file, audio_file, output_file):
    # 载入视频和音频文件
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    #//裁剪较长的视频到和较短的一样长
    if(audio.duration > video.duration):
        audio = audio.subclip(0, video.duration)
    else :
        video = video.subclip(0 , audio.duration)

    # 设置视频的音频为提取的音频
    video = video.set_audio(audio)

    # 导出合并后的视频文件
    video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # 释放资源
    audio.close()
    video.close()

    print(f"音频和视频已合并并保存到 {output_file}")

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="合并音频与视频")

    # 添加命令行参数
    parser.add_argument("video_file", help="输入的视频文件路径")
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument("output_file", help="输出的合并后视频文件路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 将输入的相对路径转换为绝对路径
    video_file = os.path.abspath(args.video_file)
    audio_file = os.path.abspath(args.audio_file)
    output_file = os.path.abspath(args.output_file)

    # 合并音频和视频
    merge_audio_video(video_file, audio_file, output_file)

if __name__ == "__main__":
    main()
