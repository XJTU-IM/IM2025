import cv2
from pyorbbecsdk import Config, Pipeline, FrameSet, OBSensorType, OBFormat, VideoStreamProfile, ColorFrame, DepthFrame, \
    OBError
from utils import frame_to_bgr_image
import numpy as np
import time
import os
import random
ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm
CAPTURE_INTERVAL = 0.2
SAVE_DIR = "Saved_img"
SAVE_DIR_ROUND = os.path.join(SAVE_DIR, "saved_dir_round")
SAVE_DIR_SQUARE = os.path.join(SAVE_DIR, "saved_dir_square")
SAVE_COLOR_ROUND_DIR = os.path.join(SAVE_DIR_ROUND, "saved_color_round_img")
SAVE_DEPTH_ROUND_DIR = os.path.join(SAVE_DIR_ROUND, "saved_depth_round_img")
SAVE_COLOR_SQUARE_DIR = os.path.join(SAVE_DIR_SQUARE, "saved_color_square_img")
SAVE_DEPTH_SQUARE_DIR = os.path.join(SAVE_DIR_SQUARE, "saved_depth_square_img")

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_ROUND, exist_ok=True)
os.makedirs(SAVE_DIR_SQUARE, exist_ok=True)
os.makedirs(SAVE_COLOR_ROUND_DIR, exist_ok=True)
os.makedirs(SAVE_DEPTH_ROUND_DIR, exist_ok=True)
os.makedirs(SAVE_COLOR_SQUARE_DIR, exist_ok=True)
os.makedirs(SAVE_DEPTH_SQUARE_DIR, exist_ok=True)


def generate_random_string(length=16):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_string = ''.join(random.choice(characters) for _ in range(length))
    random_string += ".png"
    return random_string


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


def save_color_frame(frame: ColorFrame, save_dir, filename):
    if frame is None:
        return
    color_image = frame_to_bgr_image(frame)
    if color_image is None:
        print("Failed to convert frame to image")
        return
    cv2.imwrite(os.path.join(save_dir, filename), color_image)


def save_depth_frame(frame: DepthFrame, save_dir, filename):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    scale = frame.get_depth_scale()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    data = data.astype(np.float32) * scale
    data = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0)
    data = data.astype(np.uint16)

    # Normalize depth data for visualization
    normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save as grayscale image
    cv2.imwrite(os.path.join(save_dir, filename), normalized_data)


def main():
    config = Config()
    pipeline = Pipeline()
    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    except OBError as e:
        print(f"Error configuring streams: {e}")
        return

    pipeline.start(config)
    last_print_time = time.time()
    last_capture_time = time.time()
    begin_times = 0

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                print("Frames error")
                continue

            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = frame_to_bgr_image(color_frame)
                if color_image is not None:
                    cv2.imshow("Color Viewer", color_image)

            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                temporal_filter = TemporalFilter(alpha=0.5)
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                depth_data = temporal_filter.process(depth_data)

                center_y = int(depth_frame.get_height() / 2)
                center_x = int(depth_frame.get_width() / 2)
                center_distance = depth_data[center_y, center_x]

                current_time = time.time()
                if current_time - last_print_time >= PRINT_INTERVAL:
                    print(f"Center distance: {center_distance} mm")
                    last_print_time = current_time

                # Normalize depth data for visualization
                normalized_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow("Depth Viewer", normalized_data)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
            elif key == ord(' '):
                begin_times += 1
                begin_times %= 2
                if begin_times == 1:
                    print("Capturing begins")
                else:
                    print("Capturing ends")
            elif key == ord('b'):
                random_name = generate_random_string()
                save_color_frame(color_frame, SAVE_COLOR_SQUARE_DIR, random_name)
                save_depth_frame(depth_frame, SAVE_DEPTH_SQUARE_DIR, random_name)
                print("Capture succeeded")

            if begin_times == 1:
                current_capture_time = time.time()
                if current_capture_time - last_capture_time >= CAPTURE_INTERVAL:
                    random_name = generate_random_string()
                    save_color_frame(color_frame, SAVE_COLOR_ROUND_DIR, random_name)
                    save_depth_frame(depth_frame, SAVE_DEPTH_ROUND_DIR, random_name)
                    last_capture_time = current_capture_time

        except KeyboardInterrupt:
            break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()