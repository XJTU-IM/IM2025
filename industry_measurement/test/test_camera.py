import os

import cv2
import numpy as np
import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from pyorbbecsdk import Config, Pipeline, FrameSet, OBSensorType, OBFormat, VideoStreamProfile, OBError
from utils_camera import frame_to_bgr_image

ESC_KEY = 27

PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

envpath = "~/dev/anaconda3/envs/IMM/lib/python3.9/site-packages/cv2/qt/plugins/platforms"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class CameraNode(QThread):
    color_ready = pyqtSignal(np.ndarray)  # 彩色图
    depth_ready = pyqtSignal(np.ndarray)  # 伪彩色深度图
    raw_depth_ready = pyqtSignal(np.ndarray)  # 1ch uint16 真实深度

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
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
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
                depth_data = self.temporal_filter.process(depth_data)

                depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # depth_img = cv2.applyColorMap(dep th_image, cv2.COLORMAP_JET)
                self.raw_depth_ready.emit(depth_data)
                self.color_ready.emit(color_img)
                self.depth_ready.emit(depth_img)

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


class CameraTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera & Depth Viewer")
        self.resize(1280, 480)

        # 两个标签分别显示彩色与深度
        self.raw_depth = None
        self.color_label = QLabel("Color")
        self.color_label.setAlignment(Qt.AlignCenter)
        self.depth_label = QLabel("Depth")
        self.depth_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.color_label)
        layout.addWidget(self.depth_label)

        # 启动相机线程
        self.camera = CameraNode()
        self.camera.color_ready.connect(self.show_color)
        self.camera.depth_ready.connect(self.show_depth)
        self.camera.raw_depth_ready.connect(self.store_raw_depth)
        self.camera.start()

    def show_color(self, img: np.ndarray):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.color_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.color_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_depth(self, img: np.ndarray):
        h, w = img.shape
        fmt = QImage.Format_Grayscale8
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, fmt)

        self.depth_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.depth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def store_raw_depth(self, depth_raw):
        self.raw_depth = depth_raw     # uint16 (H,W)
        self.depth_label.setMouseTracking(True)
        self.depth_label.mouseMoveEvent = self.on_depth_mouse_move

    def on_depth_mouse_move(self, ev):
        if self.raw_depth is None:
            return
        # QLabel 里的坐标 → 深度矩阵坐标
        lx, ly = ev.pos().x(), ev.pos().y()
        label_h, label_w = self.depth_label.height(), self.depth_label.width()
        img_h, img_w = self.raw_depth.shape
        mx = int(lx * img_w / max(label_w, 1))
        my = int(ly * img_h / max(label_h, 1))

        if 0 <= mx < img_w and 0 <= my < img_h:
            depth_mm = self.raw_depth[my, mx]
            # 实时显示在状态栏/工具提示
            self.depth_label.setToolTip(f"({mx},{my}) = {depth_mm} mm")
        else:
            self.depth_label.setToolTip("")

    def closeEvent(self, event):
        self.camera.stop()
        super().closeEvent(event)


# ————————————————————————————————————
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraTestWindow()
    win.show()
    sys.exit(app.exec_())