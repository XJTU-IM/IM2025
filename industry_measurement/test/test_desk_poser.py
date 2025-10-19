import glob
import os
import pprint

import numpy as np
from ultralytics import YOLO
import torch

import cv2
import matplotlib.pyplot as plt


from visualize import visualize_with_keypoints


class YOLOPosor():
    def __init__(self, path, device):
        super().__init__()

        self.model = YOLO(path)
        self.device = device

        self.model.to(device)

    def run(self, img):
        results = self.model(img)

        # 获取 bbox
        num_bboxes = len(results[0].boxes.cls)
        # print(num_bboxes)
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

        # 获取关键点
        keypoints = results[0].keypoints.cpu().xy.numpy()

        return bboxes_xyxy, keypoints   # [11, 4], [11, 4, 2]


def main(folder_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model_path = "/home/ubuntu/桌面/IM2025/industry_measurement/yolov5_seg/weights/desk_s_7000_best.pt"
    model = YOLOPosor(model_path, device)

    # 读取文件夹下所有图像
    img_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    img_paths = [p for p in img_paths
                 if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not img_paths:
        print("文件夹内未找到图像！")
        return

    idx = 0
    while True:
        img_path = img_paths[idx]
        print(f"\n[{idx+1}/{len(img_paths)}] 正在处理: {os.path.basename(img_path)}")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("读取失败，跳过")
            idx = (idx + 1) % len(img_paths)
            continue

        # 推理
        bboxes, keypoints = model.run(img_bgr)

        # 可视化
        visualized_img = visualize_with_keypoints(img_bgr, bboxes, keypoints, region=None)

        cv2.imshow("visualize", visualized_img)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):            # 按 's' 切下一张
            idx = (idx + 1) % len(img_paths)
        elif key == ord('q') or key == 27:  # 按 'q' 或 ESC 退出
            break
        else:                          # 其他键默认也切下一张，可自行修改
            idx = (idx + 1) % len(img_paths)

    cv2.destroyAllWindows()


if __name__ =="__main__":
    # desk 还要提, 这里用来测试
    folder = r"/home/ubuntu/桌面/DEBUG/GT"
    main(folder)