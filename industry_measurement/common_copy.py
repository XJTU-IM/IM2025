import os
import cv2
import numpy as np
import torch

from PyQt5.QtCore import QThread, pyqtSignal
from pyorbbecsdk import Config, Pipeline, FrameSet, OBSensorType, OBFormat, VideoStreamProfile, OBError
from utils_camera import frame_to_bgr_image

from results_process import merge_results
from yolo_poser import YOLOPosor
from calculate import  calculate_screw, calculate_shim
from visualize import visualize_bbox
from yolo_segor import Predictor
from For_shim import fit_and_draw_circles
from extract_desktop import extract
from for_ellipse import extract_circle

color_img = None
depth_img = None

ESC_KEY = 27


class ImageProcessYOLO(QThread):
    img_sign = pyqtSignal(np.ndarray)
    status_sign = pyqtSignal(str)   # ui右侧状态栏
    bar_msg = pyqtSignal(str)   # ui底部信息
    end_sign = pyqtSignal()

    def __init__(self, cfg, client):
        super().__init__()
        self.cfg = cfg
        self.round = cfg["round"]
        self.points = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._running = True
        self.tcp_client = client
        os.makedirs(cfg["save_path"], exist_ok=True)

        # 加载模型
        if self.round == 1:    # 第一轮
            # desk-keypoints model
            self.desk_poser = YOLOPosor(cfg["desk_poser"], self.device)
        elif self.round == 2:
            # circie-segment model
            self.circle_segor = Predictor(cfg["circle_segor"], conf_thres=0.4, iou_thres=0.4, device=self.device)
        else:
            raise ValueError("config argument 'round' should only be 1 (for desk), 2 (for circle)")
        
        # screw-keypoints model
        self.screw_poser = YOLOPosor(cfg["screw_poser"], self.device)

        # shim-segment model
        self.shim_segor = Predictor(weights=cfg["shim_segor"], conf_thres=0.4, iou_thres=0.4, device=self.device)

    def run(self):
        global color_img
        global depth_img

        while self._running:
            cliped_img = color_img
            cliped_depth_img = depth_img

            if cliped_img is None:
                continue
            self.bar_msg.emit("processing")
            self.img_sign.emit(cliped_img)  # 发送原图像

            # 第一步，如果没有剪裁图像，先剪裁
            if not self.cfg["cliped"]:
                _, cliped_img = extract_circle(cliped_img, self.circle_segor, mode=2)
                
                if(cliped_img is None):
                    continue
                
                cv2.imwrite("cliped.png", cliped_img)
                self.img_sign.emit(cliped_img)

            #第五步， 退出
            # if self.cfg["is_stop"]:
                # self.msleep(self.cfg["show_time"] * 1000)
                # self.end_sign.emit()

            self.msleep(self.cfg["show_time"] * 1000)
            continue

    def stop(self):
        print("正在停止image process")
        self._running = False
        self.quit()
        self.wait()
        print("线程已退出")


PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


class CameraNode(QThread):
    def __init__(self):
        super().__init__()

        self.config = Config()
        self.pipeline = Pipeline()
        self._running = True
        self.temporal_filter = TemporalFilter(alpha=0.5)
        # 彩色图
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
            except OBError as e:
                print(e)
                color_profile = profile_list.get_default_video_stream_profile()
                print("color profile: ", color_profile)
            self.config.enable_stream(color_profile)

            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            depth_profile = profile_list.get_default_video_stream_profile()
            assert depth_profile is not None
            print("depth profile: ", depth_profile)
            self.config.enable_stream(depth_profile)

        except Exception as e:
            print(e)
            return

        self.pipeline.start(self.config)

    def run(self):
        global color_img
        global depth_img

        while self._running:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue
                # covert to RGB format
                color_img = frame_to_bgr_image(color_frame)
                if color_img is None:
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
                depth_img = self.temporal_filter.process(depth_data)
                # 此时为真实的距离，而不是经过normolization
            except KeyboardInterrupt:
                break

    def stop(self):
        self._running = False
        self.quit()
        self.wait()

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


class CameraSimulation(QThread):
    def __init__(self, folder_path, fps=30):
        super().__init__()

        self.folder_path = folder_path
        self.fps = fps
        self.frame_interval = 1.0 / self.fps
        self._running = True

    def run(self):
        global color_img

        # 获取文件夹中的所有图片文件
        image_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("No image files found in the folder:", self.folder_path)
            return

        # 按文件名排序（可选，确保顺序一致）
        # image_files.sort()

        while self._running:
            for image_file in image_files:
                if not self._running:  # ✅ 关键：及时退出
                    break
                # 读取图片
                img = cv2.imread(image_file)
                if img is None:
                    print("Failed to read image:", image_file)
                    continue

                # 更新全局变量 color_img
                color_img = img

                # 等待一定时间以达到指定的帧率
                self.msleep(int(self.frame_interval * 1000))

    def stop(self):
        print("正在停止camera simulation")
        self._running = False
        self.quit()
        self.wait()
        print("线程已退出")
