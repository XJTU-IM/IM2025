import cv2
import numpy as np


def visualize_bbox(img_bgr, bboxes1, bboxes2 = None):
    # 比赛时只画框， 节省时间
    # 画框
    for bbox_idx, bbox in enumerate(bboxes1):  # 遍历每个检测框
        img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)

    if bboxes2 is not None:
        for bbox_idx, bbox in enumerate(bboxes2):  # 遍历每个检测框
            img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

    return img_bgr


def visualize_with_keypoints(img, bboxes, keypoints, region=None):
    # 检测框的颜色
    bbox_color = (150,0,0)
    # # 检测框的线宽
    bbox_thickness = 1
    kpt_radius = 1
    if region is not None:
        img = draw_grid(img, 5, (0, 255, 0))

    if(bboxes is None or len(bboxes)==0):
        return img
    
    for bbox_idx, bbox in enumerate(bboxes):  # 遍历每个检测框
        img_bgr = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thickness)

        if region is not None:
            text = str(region[bbox_idx])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)  # 黄色
            text_thickness = 1
            text_pos = (bbox[0] + 2, bbox[1] - 5)
            cv2.putText(img, text, text_pos, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    for kpt_xys in keypoints:  # 遍历该检测框中的每一个关键点
        for kpt_xy in kpt_xys:
            img_bgr = cv2.circle(img, (int(kpt_xy[0]), int(kpt_xy[1])), kpt_radius, [0, 0, 255], -1)

    return img_bgr

def visualize_with_seg(img, bboxes, mask, region=None):
    # 检测框的颜色
    bbox_color = (0,0,255)
    # # 检测框的线宽
    bbox_thickness = 1
    if region is not None:
        img = draw_grid(img, 5, (0, 255, 0))
    if(bboxes is None or len(bboxes)==0):
        return img
    
    for bbox_idx, bbox in enumerate(bboxes):  # 遍历每个检测框
        img_bgr = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thickness)

        if region is not  None:
            text = str(region[bbox_idx])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)  # 黄色
            text_thickness = 1
            text_pos = (bbox[0] + 2, bbox[1] - 5)
            cv2.putText(img, text, text_pos, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 画掩码
    img_bgr[mask!=0] = [0, 0, 255]

    return img_bgr


def draw_grid(image, r=5, color=(0, 255, 0), thickness=1, dash_length=5):
    """
    在图像上绘制 r x r 的虚线网格。

    参数:
        image (np.ndarray): 输入图像，形状为 (H, W, 3) 或 (H, W)。
        r (int): 网格的行数和列数，必须大于 0。
        color (tuple): 虚线的颜色，格式为 (B, G, R) 的元组，默认为绿色。
        thickness (int): 线宽，默认为 1。
        dash_length (int): 虚线中每个线段的长度，默认为 5 像素。

    返回:
        np.ndarray: 绘制了虚线网格的图像，形状与输入相同。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image 必须是 numpy.ndarray 类型")
    if not (isinstance(r, int) and r > 0):
        raise ValueError("r 必须是正整数")

    # 确保图像是 3 通道 BGR 格式
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass  # 已经是 BGR
    else:
        raise ValueError("image 的形状必须是 (H, W) 或 (H, W, 3)")

    h, w = image.shape[:2]
    step_x = w // r
    step_y = h // r

    def draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
        """在给定两点之间绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        dx /= length
        dy /= length
        for i in range(0, int(length), 2 * dash_len):
            start = i
            end = min(i + dash_len, int(length))
            if end <= start:
                continue
            sx = int(x1 + dx * start)
            sy = int(y1 + dy * start)
            ex = int(x1 + dx * end)
            ey = int(y1 + dy * end)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    # 绘制垂直虚线
    for i in range(1, r):
        x = i * step_x
        draw_dashed_line(image, (x, 0), (x, h), color, thickness, dash_length)

    # 绘制水平虚线
    for j in range(1, r):
        y = j * step_y
        draw_dashed_line(image, (0, y), (w, y), color, thickness, dash_length)

    return image


if __name__ == "__main__":
    image = cv2.imread("./examples/screenshot_20250428_211937.png")
    result = draw_grid(image, r=5, color=(0, 0, 255), thickness=2, dash_length=15)

    # 显示结果（需要GUI支持）
    cv2.imwrite("test.png", result)
    cv2.imshow("Grid Image", result)
    cv2.waitKey(2)
    
    