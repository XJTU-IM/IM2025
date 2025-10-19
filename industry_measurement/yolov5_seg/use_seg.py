# YOLOv5-seg 🚀 by Ultralytics, AGPL-3.0 license
import numpy as np
from loguru import logger
import os
import sys
from pathlib import Path
import torch
import cv2
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from ultralytics.utils.plotting import Annotator
from models.common import DetectMultiBackend
# from utils.dataloaders import  LoadImages
from utils.general import (
    Profile,
    cv2,
    scale_boxes,
    non_max_suppression,
)
from utils.segment.general import process_mask
from utils.torch_utils import select_device
from utils.plots import Annotator,colors
import math


class Predictor():
    def __init__(self,
        weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
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


    def calculate(self,img):
        # img=cv2.medianBlur(img,5)
        img=cv2.bilateralFilter(img,5,100,75)
        im0s=img.copy()
        im=torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.model.device)
        # print(self.dt)
        with self.dt[0]:
            # im.to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with self.dt[1]:
            # im.to(self.model.device)
            # print(im.device)
            pred, proto = self.model(im, augment=False, visualize=False)[:2]
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred,self. conf_thres, self.iou_thres,classes= None, agnostic=False, max_det=30, nm=32)
        # Process predictions
        if len(pred)==0:
            return im0s,False,False,False,False

        for i, det in enumerate(pred):  # per image
            im0=  im0s.copy()
            # annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                labels=det[:,5]
                bboxs=det[:,:4]
                # Mask plotting
                # colors=[]
                # for _ in det[:,5]:
                    # colors.append((0,255,0))
                # annotator.masks(
                    # masks,
                    # colors=[colors(x, True) for x in det[:, 5]],
                    # colors=colors,
                    # im_gpu=im[i],
                # )
                # im0 = annotator.result()
                import numpy as np
                if self.device!="cpu":masks=np.array(masks.cpu())
                areas=[np.count_nonzero(mask) for mask in masks]
                a=[]
                b=[]
                '''-------------------------------------------'''
                for i,bbox in enumerate(det[:,:4]):
                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    cv2.rectangle(im0,(x1, y1),(x2, y2),color=[255,0,0],thickness=1)
                    # print(bbox)
                    if labels[i]==0:
                        B=math.sqrt((bbox[3]-bbox[1])**2+(bbox[2]-bbox[0])**2)
                        A=areas[i]/B
                        im0[masks[i]!=0]=[255,0,0]
                        # print(B,areas[i])
                        A=A*0.3+0.8
                        B=B*0.72
                        # print("-----------screw")
                    elif labels[i]==1:
                        # B=math.sqrt(areas[i]/math.pi)
                        # A=B*1.71+3.8
                        A=min(bbox[3]-bbox[1],bbox[2]-bbox[0]).item()
                        A=A*0.93
                        # B=0.6874*A-5.711
                        B=0.6*A

                        # B=B*1.18-3.7=0.6874A-5.7111
                        # B=A*0.468+1.73
                        # print(B,"-----------washer")
                        im0[masks[i]!=0]=[0,255,0]


                    a.append(round(A,1))
                    b.append(round(B,1))
                # cv2.imwrite('/media/omnisky/sdc/2021/hwx/assignment/yolov5/runs/desk.jpg',mask)
                # cv2.imwrite('/media/omnisky/sdc/2021/hwx/assignment/yolov5/runs/im0.jpg',im0)
                # print(labels,areas,a,b,sep="\n")
                return im0,labels,bboxs,a,b

            return im0s,False,False,False,False

    def save_mask(self, img):

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
                        bbox_result.append(bboxs[i])
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


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight = "./weights/300_pretrain_best.pt"
    predictor = Predictor(weights=Path(weight), device=device)  # 500非常好

    img_path = "./example"
    save_path = "./mask"

    os.makedirs(save_path, exist_ok=True)
    for file_name in os.listdir(img_path):
        file_base_name = file_name.split(".")[0]
        mask_name = file_base_name + 'mask' + '.png'
        mased_img_name = file_base_name + "visualized" + '.png'
        mask_save_path = os.path.join(save_path, mask_name)
        masked_img_save_path = os.path.join(save_path , mased_img_name)

        file_path = os.path.join(img_path, file_name)

        if file_name.lower().endswith("png"):
            img = cv2.imread(file_path)

            bbox_result, masked, blank_img = predictor.save_mask(img)   # bbox_result是每个外径的检测框，可以用来记数， masked是可视化后的图像，没有用， blank_img是二值掩码
            if masked is not None:
                cv2.imwrite(mask_save_path, blank_img)
                cv2.imwrite(masked_img_save_path, masked)
                print("已保存至: ", mask_save_path)



