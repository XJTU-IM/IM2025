# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import os
import cv2
from datetime import datetime

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image
import numpy as np

SAVE_DIR = "/home/ubuntu/桌面/dataset1003/final_circle"
os.makedirs(SAVE_DIR, exist_ok=True)

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class TemporalFilter:   # 一阶指数移动平均滤波器
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
    
def main():
    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)    # 640 , 1280, 1920
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)

        # 深度图：
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print("depth profile: ", depth_profile)
        config.enable_stream(depth_profile)

    except Exception as e:
        print(e)
        return
    pipeline.start(config)

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))

            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            # Apply temporal filtering
            depth_data= temporal_filter.process(depth_data)
            # 此时为真实的距离，而不是经过normolization
            # depth_img= temporal_filter.process(depth_data)
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            # _, depth_image = cv2.threshold(depth_image, 150, 255, cv2.THRESH_BINARY)
            # depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)

            cv2.imshow("Depth img", depth_image)
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1) & 0xFF

            # 退出
            if key == ord('q') or key == ESC_KEY:
                break
            
            # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            # filename1 = os.path.join(SAVE_DIR, f"snap_{ts}.png")
            # print(f"Saved: {filename1}")
            # cv2.imwrite(filename1, color_image)
            # import time
            # time.sleep(1)
            # 空格保存
            if key == ord(' '):  # 32 == ord(' ')
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename1 = os.path.join(SAVE_DIR, f"snap_{ts}.png")
                filename2 = os.path.join(SAVE_DIR, f"snap_depth_{ts}.png")
                cv2.imwrite(filename1, color_image)
                # cv2.imwrite(filename2, depth_image)
                print(f"Saved: {filename1}")

        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()