import cv2
import socket
import time
import os
import sys
from utils import load_model_config

start = time.time()

cfg = load_model_config("./cfg/config.json")
cfg["round"] = 2
cfg["judge"] = False
# cfg["test"] = True

if cfg["round"] == 1:
    cfg["img_path"] = "/home/ubuntu/桌面/data/desk"
else:
    cfg["img_path"] = "/home/ubuntu/桌面/data/circle"

tcp_client = None
if cfg["judge"]:
    # 链接裁判盒
    our_ip = "192.168.173.182"
    # our_ip = "192.168.101.144"
    our_port = 0
    judger_ip = "192.168.173.252"
    # judger_ip = "192.168.101.148"
    judger_port = 6666
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client.bind((our_ip, our_port))
    tcp_client.connect((judger_ip, judger_port))
    print("连接成功")

    # 发送队名,
    id = bytes("xjtuwhfz", 'UTF-8').hex()
    tcp_client.send(bytes.fromhex("00000000" + "00000008" + id))


from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import  Qt
from PyQt5.QtGui import QImage, QPixmap

from ui_mainwindow_copy import UiMainWindow
from common_copy import CameraNode, CameraSimulation, ImageProcessYOLO


# linux下要加这两行
envpath = "~/dev/anaconda3/envs/IMM/lib/python3.9/site-packages/cv2/qt/plugins/platforms"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class MainWindow(UiMainWindow):
    def __init__(self, cfg, tcp_client):
        super().__init__()
        self.cfg = cfg
        self.tcp_client = tcp_client
            
        # -------------------- 业务线程 --------------------
        if not cfg["test"]:
            self.camera_node = CameraNode()
            self.camera_node.start()
        else:
            self.camera_simulation = CameraSimulation(folder_path=cfg["img_path"], fps=10)
            self.camera_simulation.start()

        self.image_process = ImageProcessYOLO(cfg, self.tcp_client)
        self.image_process.img_sign.connect(self.show_img)
        # self.image_process.status_sign.connect(self.show_log)
        # self.image_process.bar_msg.connect(self.show_bar)
        self.image_process.end_sign.connect(self.end)
        self.image_process.start()

    def show_img(self, img):
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        # 按 label 当前尺寸、保持比例、平滑缩放
        pix = pix.scaled(self.image_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def show_log(self, status):
        self.log_edit.append(status)

    def show_bar(self, status):
        self.status_bar.showMessage(status)

    def end(self):
        print("正在退出程序...")

        # self.camera_simulation.stop()
        # self.image_process.stop()
        QApplication.quit()

    def closeEvent(self, event):
        self.end()
        event.accept()


app = QApplication(sys.argv)
w = MainWindow(cfg, tcp_client)
w.show()

rec = app.exec_()
print("用时:", time.time() - start)
sys.exit(rec)
