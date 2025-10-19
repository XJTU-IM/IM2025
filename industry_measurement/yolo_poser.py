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

        return bboxes_xyxy, keypoints   # [nums_object, 4], [nums_object, keypoints_per_obj, 2]


if __name__ =="__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model_path = "yolov5_seg/weights/screw_s_2000.pt"
    model = YOLOPosor(model_path, device)

    img_bgr = cv2.imread("/home/ubuntu/桌面/IM2025_circle/industry_measurement/cliped_origin.png")
    # img_bgr = cv2.resize(img_bgr, (640, 640))
    bboxes, keypoints = model.run(img_bgr)
    print(type(keypoints))

    # 可视化
    visualized_img = visualize_with_keypoints(img_bgr, bboxes, keypoints)
    cv2.imshow("visualize", visualized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
