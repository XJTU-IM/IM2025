# YOLOv5-seg 🚀 by Ultralytics, AGPL-3.0 license
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
from yolov5_seg.utils.augmentations import letterbox


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
                print("未检测到")
                return None, None, None

    def calculate_shim(self, bboxes, mask):
        results = []
        for bbox in bboxes:
            result = {}
            result["Goal_ID"] = 1

            print(bbox)
        return results

def point_on_line(a: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        len_ab = np.linalg.norm(ab)
        if len_ab == 0:
            raise ValueError("a 和 b 不能是同一个点")
        unit = ab / len_ab          # 单位方向向量
        c = b - d * unit
        return c


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight = "./yolov5_seg/weights/circle-4000.pt"
    predictor = Predictor(weights=Path(weight), device=device)  # 500非常好

    img_path = "/home/ubuntu/桌面/dataset1003/snap_20251012_224445_102.png"
    img = cv2.imread(img_path)
    img = letterbox(img, new_shape=(736, 1280), auto=False)[0]
    cv2.imwrite("origin.png", img)
    # 第一步， 拿到椭圆
    bbox, visualized, mask = predictor.save_mask(img) # 拿到掩码， mask:(h, w, c)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)   # 拿到掩码， 接下来做椭圆拟合


    # edge = cv2.Canny(mask, 127, 255)
    # contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = max(contours, key=cv2.contourArea)    # 拿到最大的轮廓
    # retval = cv2.fitEllipse(cnt)
    # fixed_mask = np.zeros_like(mask)
    # ellipse_mask = np.zeros_like(mask)
    # cv2.ellipse(ellipse_mask, retval, 255, -1)
    # dst = cv2.ellipse(fixed_mask, retval, (255, 255, 255), 2)
    
    # cv2.imwrite("mask.png", ellipse_mask)


    edge = cv2.Canny(mask, 127, 255)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)    # 拿到最大的轮廓
    retval = cv2.fitEllipse(cnt) # 椭圆的旋转矩形((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(retval)        # 椭圆旋转矩形的四个顶点，左上，右上，右下，左下 返回 4×2 的 numpy 数组，float
    # print("旋转矩形顶点:", box)
    print(type(retval))
    print(box)
    points = np.float32(box)
    dstp = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
    M = cv2.getPerspectiveTransform(points, dstp)
    cliped_img = cv2.warpPerspective(img, M, (640, 640))
    cv2.imwrite("test.png", cliped_img)

    # 第二步， 计算椭圆的相关参数：圆心， 长轴端点，短轴端点(浮点型)
    center, (w, h), angle = retval 

    pt_long1 = (box[1] + box[2])/2
    pt_long2 = (box[0] + box[3])/2
    pt_short1 = (box[0] + box[1])/2
    pt_short2 = (box[2] + box[3])/2

    print("中心：", center)
    print("椭圆的长端点：", pt_long1, "短端点", pt_short1)

    # 第三步， 计算圆的映射中心，我们认为偏移距离与椭圆的扁率有关, 如果是圆的话，扁率为0
    a = max(retval[1])/2  # 长轴
    b = min(retval[1])/2  # 短轴
    e = (a - b) / (a + b)
    print("扁率:", e)
    bias = 50   # 偏移系数，需要反复调整
    residual = 50 * e 
    c = point_on_line(pt_short1, center, residual)  # 圆的映射中心
    print("圆的映射中心:", c)

    # 第四步， 根据圆的映射中心计算外接矩形的四个顶点
    slope = 0.9 # 斜率，也需要反复调整，目前还不清楚该怎么把斜率和偏移系数联合调参
    from for_ellipse import quad_ADBC_general
    
    A, D, B, C_prime = quad_ADBC_general(center, pt_long1, pt_short1, c, slope)

    # 第五步， 可视化， 画椭圆 -> 画中心 -> 画透视中心 -> 画外接梯形
    vis = img
    cv2.ellipse(vis, retval, (0, 255, 0), 2)
    cv2.circle(vis, np.intp(center), 2, (255, 0, 0), 3)
    cv2.circle(vis, np.intp(c), 2, (0, 255, 255), 3)

    A, D, B, Cp = map(lambda p: (int(round(p[0])), int(round(p[1]))), (A, D, B, C_prime))
    cv2.line(vis, A, D, (255, 0, 0), 2)
    cv2.line(vis, B, Cp, (255, 0, 0), 2)
    cv2.line(vis, A, Cp, (255, 0, 0), 2)
    cv2.line(vis, B, D, (255, 0, 0), 2)
    cv2.imshow("img", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

