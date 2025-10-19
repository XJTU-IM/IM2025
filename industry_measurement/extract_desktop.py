import cv2
import numpy as np
from for_ellipse import order_box_points, point_on_line, parallel_intersect_ellipse
from yolov5_seg.utils.augmentations import letterbox

def extract(color, processor, mode):
    if mode == 1: # 方桌
        _, points = processor.desk_poser.run(color)    # [1, 4, 2]
        if(points is None or len(points)==0):
            print("未检测到桌面")
            return False, None, None    # 如果没有检测到关键点怎么办
        if(len(points[0]) <4):
            print("顶点不全")
            return False, None, None 
        print(points)
        points = np.squeeze(points, axis=0).astype(np.float32)   #[4 , 2]
            
        vis = color.copy()
        for kpt_xy in points:
            vis = cv2.circle(vis, (int(kpt_xy[0]), int(kpt_xy[1])), 2, [0, 0, 255], -1)
        cv2.imwrite("./saves/cliped_desk.png", vis)  # 保存方桌的关键点可视化图像

        dstp = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(color, M, (640, 640))
        return True, cliped_img, vis
    
    elif mode == 2: # 圆桌
        img = letterbox(color, new_shape=(736, 1280), auto=False)[0]
        vis = img
        _, _, mask = processor.circle_segor.save_mask(img) 
        if(mask is None):
            print("检测圆桌失败")
            # TODO： 如果予以分割实效了，用深度图， hzw做接口 cliped_img = seg_with_depth(depth_img)
            return False, color, vis
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)   # 拿到掩码， 接下来做椭圆拟合
        edge = cv2.Canny(mask, 127, 255)
        contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)    # 拿到最大的轮廓
        retval = cv2.fitEllipse(cnt) # 椭圆的旋转矩形((center_x, center_y), (width, height), angle)
        box = cv2.boxPoints(retval)        # 椭圆旋转矩形的四个顶点，左上，右上，右下，左下 返回 4×2 的 numpy 数组，float
        box = order_box_points(box)
        ellipse_mask = np.zeros_like(mask)
        cv2.ellipse(ellipse_mask, retval, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=ellipse_mask)

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
        residual = 430 * e 
        c = point_on_line(pt_short1, center, residual)  # 圆的映射中心
        print("圆的映射中心:", c)

        # 第四步， 根据圆的映射中心找到圆的直径两点
        origin_point = parallel_intersect_ellipse(pt_long1, pt_long2, c, retval)
        if(len(origin_point) == 0):
            print("无交点")
            return False, color, vis
        origin_left, origin_right = origin_point
        
        cv2.ellipse(vis, retval, (0, 255, 0), 2)
        cv2.polylines(vis, [np.intp(box)], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.circle(vis, np.intp(pt_short1), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(pt_short2), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(origin_left), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(origin_right), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(center),2, (0, 255, 0), 3)
        
        points = np.float32([pt_short1, origin_right, pt_short2, origin_left])
        dstp = np.float32([[320, 0], [640, 320], [320, 640], [0, 320]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(masked, M, (640, 640))
        cv2.imwrite("./saves/cliped_circle.png", vis)

        return True, cliped_img, vis

    else:
        raise ValueError("config argument 'round' should only be 1 (for desk), 2 (for circle)")

