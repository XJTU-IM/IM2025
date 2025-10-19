# YOLOv5-seg 🚀 by Ultralytics, AGPL-3.0 license
import glob

import numpy as np
from loguru import logger
import os
import sys
from pathlib import Path
import torch
import cv2
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / Path("yolov5_seg")  # YOLOv5 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.utils.plotting import Annotator
from yolov5_seg.models.common import DetectMultiBackend
# from utils.dataloaders import  LoadImages
from yolov5_seg.utils.general import (
    Profile,
    cv2,
    scale_boxes,
    non_max_suppression,
)
from yolov5_seg.utils.segment.general import process_mask
from yolov5_seg.utils.torch_utils import select_device
from yolov5_seg.utils.plots import Annotator,colors
import math
from visualize import visualize_with_seg

class Predictor():
    def __init__(self,
        weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
        device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        line_thickness=1,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    ):
    # Load model
        logger.info("LOADING MODEL ------- WAIT")
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=None, fp16=half)
        self.names= self.model.names
        self.model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
        self.dt = (Profile(), Profile(), Profile())
        self.line_thickness=line_thickness
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres

    def save_mask(self, img):

        if isinstance(img, str):
            img = cv2.imread(img)

        with self.dt[0]:
            # im = torch.from_numpy(img).to(self.model.device)
            im = torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:    # preds包括边界框、对象置信度、类别概率和实例掩码  # proto为原型掩码
            pred, proto = self.model(im, augment=False, visualize=False)[:2]

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred,self. conf_thres, self.iou_thres,classes= None, agnostic=False, max_det=30, nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = img.copy()

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # show img
                labels = det[:, 5]
                bboxs = det[:, :4]
                bbox_result = []
                # 我们现在要做的使用mask信息保存一个掩码图像， 然后用这个图像去计算外径和内径
                masks = masks.cpu()

                masked = img.copy()

                blank_img = np.zeros_like(img)
                for i,bbox in enumerate(bboxs):
                    # x1, y1, x2, y2 = map(int, bbox.tolist())
                    # cv2.rectangle(masked,(x1, y1),(x2, y2),color=[255,0,0],thickness=1)

                    if labels[i]==0:    # wai
                        bbox = [int(x) for x in bbox]
                        bbox_result.append(bbox)
                        blank_img[masks[i] != 0 ] = 255
                        masked[masks[i]!=0]=[255,0,0]
                    # if labels[i]==1:  # nei
                    #     masked[masks[i]!=0]=[0, 0, 255]

                # cv2.imshow("masked", blank_img)
                # cv2.waitKey(0)
                return bbox_result, masked, blank_img
            else:
                print("未检测到垫片")
                return None, None, None

    def calculate_shim(self, bboxes, mask):
        results = []
        for bbox in bboxes:
            result = {}
            result["Goal_ID"] = 1

            print(bbox)
        return results


def main(folder_path):
    device = 'cuda:0'
    print('device:', device)

    weight = "./yolov5_seg/weights/shim_s_4000.pt"
    predictor = Predictor(weights=Path(weight), device=device)

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
        bbox, visualized, mask = predictor.save_mask(img_bgr)

        # 可视化
        visualized_img = visualize_with_seg(img_bgr, bbox, mask, region=None)

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

if __name__ == "__main__":
    img_path = "/home/ubuntu/桌面/cliped/cross"
    main(img_path)

