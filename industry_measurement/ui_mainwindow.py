# ui_mainwindow.py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSplitter, QFrame, QLabel,
                             QTextEdit, QStatusBar)


class UiMainWindow(QMainWindow):
    """纯界面，不含业务线程、网络、配置等任何逻辑"""
    def __init__(self):
        super().__init__()
        self._setup_ui()

    # ---------- 以下全是界面细节 ----------
    def _setup_ui(self):
        self.setWindowTitle("IM2025")
        self.resize(1600, 900)

        # 中心控件
        central = QWidget(self)
        self.setCentralWidget(central)

        # 主分割器
        splitter = QSplitter(Qt.Horizontal, central)
        lay_main = QVBoxLayout(central)
        lay_main.setContentsMargins(4, 4, 4, 4)
        lay_main.addWidget(splitter)

        # 左侧图像区
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setMaximumWidth(1280)  
        self.image_frame.setStyleSheet("""
            QFrame{
                border-radius:10px;
                background-color:#1e1e1e;
                border:2px solid #444;
            }
        """)
        self.image_label = QLabel("Waiting for image …")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False) 
        self.image_label.setStyleSheet("color:#ddd; font-size:18px;")

        lay_img = QVBoxLayout(self.image_frame)
        lay_img.setContentsMargins(5, 5, 5, 5)
        lay_img.addWidget(self.image_label)
        splitter.addWidget(self.image_frame)

        # 右侧日志区
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumWidth(400)
        self.log_edit.setStyleSheet("""
            QTextEdit{
                border-radius:6px;
                background:#252525;
                color:#f0f0f0;
                font:14px "Consolas";
                border:1px solid #555;
            }
        """)
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(self.log_edit)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")