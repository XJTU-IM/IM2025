import numpy as np
import cv2
from typing import Tuple
import math
from yolov5_seg.utils.augmentations import letterbox

Point = Tuple[float, float]

def _vec(p: Point, q: Point) -> Point:
    """向量 q - p"""
    return (q[0] - p[0], q[1] - p[1])

def _dot(u: Point, v: Point) -> float:
    return u[0]*v[0] + u[1]*v[1]

def _add(p: Point, v: Point) -> Point:
    return (p[0] + v[0], p[1] + v[1])

def _scale(v: Point, s: float) -> Point:
    return (v[0]*s, v[1]*s)

def quad_ADBC_general(O: Point, Ax: Point, Ay: Point, C: Point, k1: float):
    """
    在由 O,Ax,Ay 张成的新仿射坐标系下完成题意所有计算，
    返回 ADBC 四个点在**原始直角坐标系**中的坐标。
    """
    # 1. 构造新基底
    e1 = _vec(O, Ax)
    e2 = _vec(O, Ay)
    a = math.hypot(*e1)
    b = math.hypot(*e2)
    print("a点坐标:", a, "b点坐标:", b)
    c = math.hypot(*_vec(O, C))
    if a == 0 or b == 0:
        raise ValueError("Ax 或 Ay 与 O 重合，无法建立坐标系")
    e1 = _scale(e1, 1/a)
    e2 = _scale(e2, 1/b)
    # print("新基底", e1, e2)
    # 2. 把 C 投影到新坐标系，验证它确实在「新 y 轴」上（即 u≈0）
    u_c = _dot(_vec(O, C), e1)
    v_c = _dot(_vec(O, C), e2)
    print("c点坐标:", u_c, v_c)
    if abs(u_c) > 0.2:
        print("C 点不在 O-Ay 直线上，无法继续")
        return None
    if not (0 < v_c < b):
        print("C 点新 y 坐标不在 0~b 之间")
        return None

    # 3. 计算 k2
    if abs(k1) < 1e-12:
        k2 = math.copysign(math.inf, -(b*b - c*c))
    else:
        k2 = -(b*b - c*c) / (a*a) / k1

    # 4. 在新坐标系里求交点
    # 直线： v - v_c = k * u   =>  v = k*u + v_c
    # 与 v = b  交点： u = (b  - v_c)/k
    # 与 v = -b 交点： u = (-b - v_c)/k
    def intersect(k):
        if math.isinf(k):
            # 竖直线 u = u_c
            return (u_c, b), (u_c, -b)
        if abs(k) < 1e-12:
            # 水平线 v = v_c
            return (0, b), (0, -b)
        u1 = (b  - v_c) / k
        u2 = (-b - v_c) / k
        return (u1, b), (u2, -b)

    A_uv, B_uv = intersect(k1)
    C_uv, D_uv = intersect(k2)

    # 5. 把四点转回原始坐标系
    def uv_to_xy(u, v):
        return _add(O, _add(_scale(e1, u), _scale(e2, v)))

    A = uv_to_xy(*A_uv)
    B = uv_to_xy(*B_uv)
    C_prime = uv_to_xy(*C_uv)
    D = uv_to_xy(*D_uv)

    # 6. 按 ADBC 顺序返回
    return A, D, B, C_prime

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

import numpy as np

def order_box_points(pts):
    """
    将 cv2.boxPoints 得到的 4 个顶点排序为
    [左上, 右上, 右下, 左下]

    pts : (4, 2) 的 numpy 数组，float32 或 float64
    返回：ordered_pts，shape 同样为 (4, 2)
    """
    assert pts.shape == (4, 2), "输入必须是 4×2 的点集"

    # 1. 按 y 坐标升序，分成上下两组
    ysorted = pts[pts[:, 1].argsort()]
    top2, bottom2 = ysorted[:2], ysorted[2:]

    # 2. 对上下分别按 x 升序，得到 left 和 right
    tl, tr = top2[top2[:, 0].argsort()]
    bl, br = bottom2[bottom2[:, 0].argsort()]

    # 3. 按 [tl, tr, br, bl] 返回
    return np.array([tl, tr, br, bl], dtype=pts.dtype)

def extract_circle_1015(color, processor, mode = 2):

    img = letterbox(color, new_shape=(736, 1280), auto=False)[0]
    vis = img
    _, _, mask = processor.save_mask(img) 
    if(mask is None):
        print("seg failed, use depth data")
        # TODO： 如果予以分割实效了，用深度图， hzw做接口 cliped_img = seg_with_depth(depth_img)
        return color, None
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

    if mode == 1:   # 旋转矩形的做法
        points = np.float32(box)
        dstp = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(masked, M, (640, 640))
        return cliped_img, None

    elif mode == 2:
        # 第一步， 拿到椭圆
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

        # 第四步， 根据圆的映射中心计算外接矩形的四个顶点
        slope = 0.84 # 斜率，也需要反复调整，目前还不清楚该怎么把斜率和偏移系数联合调参
        
        if(quad_ADBC_general(center, pt_long1, pt_short1, c, slope) is None):
            vis = img
            cv2.ellipse(vis, retval, (0, 255, 0), 2)
            cv2.circle(vis, np.intp(center), 2, (255, 0, 0), 3)
            cv2.circle(vis, np.intp(c), 2, (0, 255, 255), 3)
            return vis, None
        A, D, B, C_prime = quad_ADBC_general(center, pt_long1, pt_short1, c, slope)

        points = np.float32([A, D, B, C_prime])
        dstp = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(masked, M, (640, 640))

        # 第五步， 可视化， 画椭圆 -> 画中心 -> 画透视中心 -> 画外接梯形
        box = np.intp(box)
        cv2.drawContours(vis, [box], 0, (0,0,255), 2)

        cv2.ellipse(vis, retval, (0, 255, 0), 2)
        cv2.circle(vis, np.intp(center), 2, (255, 0, 0), 3)
        cv2.circle(vis, np.intp(c), 2, (0, 255, 255), 3)

        
        A, D, B, Cp = map(lambda p: (int(round(p[0])), int(round(p[1]))), (A, D, B, C_prime))
        cv2.line(vis, A, D, (255, 0, 0), 2)
        cv2.line(vis, B, Cp, (255, 0, 0), 2)
        cv2.line(vis, A, Cp, (255, 0, 0), 2)
        cv2.line(vis, B, D, (255, 0, 0), 2)

        return cliped_img, vis
    
     
        # cv2.imshow("img", vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def parallel_intersect_ellipse(a, b, c, retval):
    (cx, cy), (w, h), angle_deg = retval
    w = w/2
    h = h/2
    angle = math.radians(angle_deg)
    
    # 1. 方向向量 u = b - a，并单位化
    u = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    if np.allclose(u, 0):
        raise ValueError("a 和 b 不能重合")
    u = u / np.linalg.norm(u)
    
    # 2. 将 c 和方向向量 u 平移到以椭圆中心为原点的坐标系
    c_shift = np.array([c[0] - cx, c[1] - cy], dtype=float)
    
    # 3. 旋转坐标系，使椭圆轴对齐
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    c_rot = R @ c_shift
    u_rot = R @ u
    
    # 4. 直线参数方程：P = c_rot + t * u_rot
    #    代入椭圆方程 x^2/w^2 + y^2/h^2 = 1
    #    得到关于 t 的二次方程 A t^2 + B t + C = 0
    xr, yr = c_rot
    ux, uy = u_rot
    A = (ux**2) / (w**2) + (uy**2) / (h**2)
    B = 2 * (xr * ux / (w**2) + yr * uy / (h**2))
    C = (xr**2) / (w**2) + (yr**2) / (h**2) - 1
    
    disc = B**2 - 4*A*C
    if disc < 0:
        return []          # 无交点
    
    sqrt_d = math.sqrt(disc)
    t1 = (-B - sqrt_d) / (2*A)
    t2 = (-B + sqrt_d) / (2*A)
    
    # 5. 求交点并旋转/平移回原坐标系
    def point_from_t(t):
        x_rot, y_rot = c_rot + t * u_rot
        # 先逆旋转，再平移回去
        x, y = R.T @ np.array([x_rot, y_rot])
        return (x + cx, y + cy)
    
    p1 = point_from_t(t1)
    p2 = point_from_t(t2)
    
    # 6. 按 x 从小到大排序
    return sorted([p1, p2], key=lambda pt: pt[0])

def extract_circle(color, processor, mode = 2):

    img = letterbox(color, new_shape=(736, 1280), auto=False)[0]
    vis = img
    _, _, mask = processor.save_mask(img) 
    if(mask is None):
        print("seg failed, use depth data")
        # TODO： 如果予以分割实效了，用深度图， hzw做接口 cliped_img = seg_with_depth(depth_img)
        return color, None
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

    if mode == 1:   # 旋转矩形的做法
        points = np.float32(box)
        dstp = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(masked, M, (640, 640))
        return cliped_img, None

    elif mode == 2:
        # 第一步， 拿到椭圆
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
            return color, color
        origin_left, origin_right = origin_point
        cv2.circle(vis, np.intp(pt_short1), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(pt_short2), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(origin_left), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(origin_right), 2, (0, 0, 255), 2)
        cv2.circle(vis, np.intp(center),2, (0, 255, 0), 3)
        cv2.ellipse(vis, retval, (0, 255, 0), 2)
        cv2.polylines(vis, [np.intp(box)], isClosed=True, color=(255, 0, 0), thickness=2)

        points = np.float32([pt_short1, origin_right, pt_short2, origin_left])
        dstp = np.float32([[320, 0], [640, 320], [320, 640], [0, 320]])
        M = cv2.getPerspectiveTransform(points, dstp)
        cliped_img = cv2.warpPerspective(masked, M, (640, 640))
        return cliped_img, vis

# ------------------- 简单测试 -------------------
if __name__ == "__main__":

    O = (669.9677, 420.4250)
    a = (967.17, 431.26)
    b = (679.33, 163.64)
    c = (671.79, 370.46)
    k1 = 0.85

    A, D, B, C_prime = quad_ADBC_general(O, a, b, c, k1)
    A, D, B, Cp = map(lambda p: (int(round(p[0])), int(round(p[1]))), (A, D, B, C_prime))
    print("输出四边形 ADBC 顶点：")
    print("A =", A)
    print("D =", D)
    print("B =", B)
    print("C=", C_prime)
    import cv2
    import numpy as np
    img = cv2.imread("/home/ubuntu/桌面/dataset1003/snap_20251012_224445_102.png")
    from yolov5_seg.utils.augmentations import letterbox
    img = letterbox(img, new_shape=(736, 1280), auto=False)[0]
    cv2.line(img, A, D, (255, 0, 0), 2)
    cv2.line(img, B, Cp, (255, 0, 0), 2)
    cv2.line(img, A, Cp, (255, 0, 0), 2)
    cv2.line(img, B, D, (255, 0, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()