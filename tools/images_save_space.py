import cv2
import os
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27
SPACE_KEY = 32  # 空格键的ASCII码

# 创建保存图片的文件夹
if not os.path.exists('/home/ubuntu/桌面/dataset1003/square'):
    os.makedirs('/home/ubuntu/桌面/dataset1003/square')

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result
    
def get_next_frame_number(folder):
    """获取文件夹中最大编号的下一个编号"""
    max_number = -1
    for filename in os.listdir(folder):
        if filename.startswith('frame_') and filename.endswith('.png'):
            try:
                # 提取文件名中的编号
                number = int(filename[6:-4])  # 去掉 'frame_' 和 '.jpg'
                if number > max_number:
                    max_number = number
            except ValueError:
                continue
    return max_number + 1  # 返回下一个编号

def main():
    config = Config()
    pipeline = Pipeline()
    # 配置摄像头流
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    # 启动摄像头
    pipeline.start(config)
    # 获取下一个编号
    frame_count = get_next_frame_number('images')
    # 捕获并显示图像
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            # Resize the image to 640x640
            resized_image = cv2.resize(color_image, (640, 640))
            # Display the resized image
            cv2.imshow("Color Viewer", resized_image)
            # 检查键盘输入
            key = cv2.waitKey(1)
            if key == SPACE_KEY:  # 按下空格键保存图片
                # Save the resized image
                image_path = os.path.join('images', f'frame_{frame_count:04d}.png')
                cv2.imwrite(image_path, resized_image)
                print(f"Saved {image_path}")
                frame_count += 1  # 递增编号
            elif key == ord('q') or key == ESC_KEY:  # 按下 'q' 或 ESC 键退出
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()