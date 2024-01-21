import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
def main_worker():
    # 读取命令行
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Input folder containing video and images')
    args = parser.parse_args()
    # 获得文件名列表
    files = sorted(file for file in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, file)))
    image_list = []
    video_path = str()
    # 读取视频文件
    for file in files:
        if file.endswith(".avi"):
            video_path = file
        else:
            image_list.append(file)
    video = cv2.VideoCapture(os.path.join(args.path, video_path))

    if not video.isOpened():
        print("Video open failed!")
        exit()
    # 获得视频的基本属性
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 创建一个可以新写入的视频
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    result_path = os.path.join(args.path, "result.avi")
    result = cv2.VideoWriter(result_path, fourcc, fps, (int(width), int(height)))
    # 设置开头动画
    s = "PC6> ip 192.168.0.92 255.255.255.0 192.168.0.1\nChecking for duplicate address...\nPC6 :192.168.0.92 255.255.255.0 gateway 192.168.0.1"
    text_size = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 3, 2)[0]
    start_position = (-text_size[0], height // 2)
    duration = 7
    print("Generating opening credits...")
    # 按照帧数变化制作动画效果
    for t in tqdm(np.linspace(0, duration, int(fps * duration))):
        x_position = 300
        fram = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.putText(
            img=fram,
            text=s,
            org=(x_position, start_position[1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

        result.write(fram)
    # 插入图片
    print("Inserting images...")

    # 释放资源
    video.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_worker()
