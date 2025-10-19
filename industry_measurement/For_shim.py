import cv2
import numpy as np
import random
from math import hypot
from utils import locate_region
import math

# ---- 辅助：由三点得到圆心与半径 ----
def circle_from_3pts(p1, p2, p3):
    (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
    # 矩阵法求解，避免精度问题
    A = np.array([[x2-x1, y2-y1],
                  [x3-x1, y3-y1]], dtype=float)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    B = np.array([((x2**2 - x1**2) + (y2**2 - y1**2)) / 2.0,
                  ((x3**2 - x1**2) + (y3**2 - y1**2)) / 2.0])
    center = np.linalg.solve(A, B)
    cx = center[0] + x1
    cy = center[1] + y1
    r = np.hypot(cx-x1, cy-y1)
    return (cx, cy, r)

# ---- RANSAC 圆心拟合 ----
def ransac_circle(points, iterations=1500, tol=2.5):
    best = None
    best_inliers = []
    n = len(points)
    if n < 3:
        return None, []
    for _ in range(iterations):
        a,b,c = random.sample(points, 3)
        circ = circle_from_3pts(a,b,c)
        if circ is None:
            continue
        cx,cy,r = circ
        # 计数内点
        diffs = [abs(hypot(x-cx,y-cy) - r) for (x,y) in points]
        inliers = [p for p,d in zip(points,diffs) if d <= tol]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best = (cx,cy,r)
    return best, best_inliers

# ---- 1D k-means 用于半径分成内外两类 ----
def kmeans_1d(values, k=2, maxit=50):
    vals = np.array(values)
    # 初始化：用两端点
    centers = np.percentile(vals, [25,75]) if k==2 else np.linspace(vals.min(), vals.max(), k)
    for _ in range(maxit):
        dists = np.abs(vals[:,None] - centers[None,:])
        labels = dists.argmin(axis=1)
        new_centers = np.array([vals[labels==i].mean() if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers
    return centers, labels

def Incomplete_circle_process(img):
    # ---- 主流程 ----
    # 二值或 Canny
    _,th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(th, 50, 150)

    # 提取边缘点坐标
    ys, xs = np.nonzero(edges)
    points = list(zip(xs.tolist(), ys.tolist()))

    # RANSAC 找中心（不敏感于缺口）
    best, inliers = ransac_circle(points, iterations=2000, tol=2.5)
    if best is None:
        raise RuntimeError("无法拟合，边缘点不足或全共线")
    cx, cy, r_est = best

    # 用所有边缘点到中心的距离进行 1D 聚类得内外半径
    dists = [hypot(x-cx, y-cy) for (x,y) in points]
    centers, labels = kmeans_1d(dists, k=2)
    r_inner, r_outer = sorted(centers)  # 小的为内圆

    # 更稳健：用中位数而不是均值
    inner_d = np.median([d for d,l in zip(dists,labels) if l==np.argmin(centers)])
    outer_d = np.median([d for d,l in zip(dists,labels) if l==np.argmax(centers)])
    r_inner, r_outer = sorted([inner_d, outer_d])

    # print("center:", (cx,cy))
    # print("radii:", r_inner, r_outer)

    # # 可视化
    # out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.circle(out, (int(round(cx)),int(round(cy))), int(round(r_inner)), (0,255,0), 1) # 内：绿
    # cv2.circle(out, (int(round(cx)),int(round(cy))), int(round(r_outer)), (0,0,255), 1) # 外：红
    # cv2.imwrite('fitted_result.png', out)

    return [int(round(cx)),int(round(cy))], int(round(r_inner)), int(round(r_outer))


###################################################################################################



# ---- 1D k-means 用于半径分成内外两类 ----
def kmeans_1d(values, k=2, maxit=50):
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return [], []
    # 初始化：用两端分位数作为初值
    centers = np.percentile(vals, [25, 75]) if k == 2 else np.linspace(vals.min(), vals.max(), k)
    for _ in range(maxit):
        dists = np.abs(vals[:, None] - centers[None, :])
        labels = dists.argmin(axis=1)
        new_centers = np.array([
            vals[labels == i].mean() if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers
    return centers, labels

# ---- 主流程：已知外圆圆心和半径 ----
def Incomplete_circle_process_known_outer(img, center, r_outer_known, tol_margin=2.0, Default_coefficient=0.5):
    """
    img           : 输入灰度图
    center        : (cx, cy) 已知外圆圆心
    r_outer_known : 已知外圆半径
    tol_margin    : 内外圆半径区分的安全裕量
    """
    cx, cy = center

    # print(f"cx = {cx}, cy = {cy}")

    # 二值化 & Canny 边缘
    _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(th, 50, 150)

    # 提取边缘点
    ys, xs = np.nonzero(edges)
    points = list(zip(xs.tolist(), ys.tolist()))
    if not points:
        # raise RuntimeError("未检测到边缘点")
        print("未检测到边缘点")
        return [int(cx), int(cy)], int(r_outer_known * Default_coefficient), r_outer_known

    # 计算所有边缘点到已知圆心的距离
    dists = [hypot(x - cx, y - cy) for (x, y) in points]

    # 仅保留小于外圆半径 - tol_margin 的点用于内圆估计
    inner_dists = [d for d in dists if d < r_outer_known - tol_margin]
    if not inner_dists:
        # raise RuntimeError("未检测到内圆边缘点")
        print("未检测到内圆边缘点")
        return [int(cx), int(cy)], int(r_outer_known * Default_coefficient), r_outer_known

    # 用中位数估计内圆半径（抗缺口和噪声）
    r_inner = np.median(inner_dists)

    # 返回结果（圆心取整数，半径四舍五入）
    return [int(round(cx)), int(round(cy))], int(round(r_inner)), int(round(r_outer_known))


####################################################################################################
def count_children(hierarchy, index):
    """
    计算给定索引处的轮廓有多少个子轮廓。
    
    :param hierarchy: 轮廓的层次结构数组。
    :param index: 要计算子轮廓数的轮廓索引。
    :return: 子轮廓的数量。
    """
    child_count = 0
    if index == -1:
        return child_count  # 如果没有子轮廓，返回计数0
    
    current_index = hierarchy[index][2]  # 获取第一个子轮廓的索引
    while current_index != -1:
        child_count += 1
        current_index = hierarchy[current_index][0]  # 移动到下一个同级轮廓
        
    return child_count



def Connected_circles_process(img):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    inner_circles = []
    outer_circles = []

    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent == -1:
            outer_contour = contours[i]
        else:
            (x, y), r = cv2.minEnclosingCircle(contours[i])
            inner_circles.append(((x, y), r))

    if len(inner_circles) != 2:
        raise ValueError("检测到的内圆数量不是2个")

    points = outer_contour.reshape(-1, 2).astype(np.float32)

    c1 = np.array(inner_circles[0][0])
    c2 = np.array(inner_circles[1][0])

    # 存储分组
    group_points = [[], []]

    # 距离越大，权重越低，这里用权重 = 1 / (距离 + 1e-6)
    for p in points:
        d1 = np.linalg.norm(p - c1)
        d2 = np.linalg.norm(p - c2)

        # w1 = 1.0 / (d1 + 1e-6)
        # w2 = 1.0 / (d2 + 1e-6)

        sigma = 99.0  # 控制衰减速度
        w1 = np.exp(-(d1**2) / (2*sigma**2))
        w2 = np.exp(-(d2**2) / (2*sigma**2))


        # 权重大的一组作为归属
        if w1 > w2:
            group_points[0].append(p)
        else:
            group_points[1].append(p)

    # 拟合外圆
    for k in range(2):
        cluster_points = np.array(group_points[k], dtype=np.float32)
        (x, y), r = cv2.minEnclosingCircle(cluster_points)
        outer_circles.append(((x, y), r))

    # 画结果
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for (x, y), r in inner_circles:
    #     cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)  # 绿 内圆
    # for (x, y), r in outer_circles:
    #     cv2.circle(output, (int(x), int(y)), int(r), (255, 0, 0), 2)  # 蓝 外圆

    cv2.imshow("Detected Circles", output)
    cv2.waitKey(0)

    print("Inner circles:", inner_circles)
    print("Outer circles:", outer_circles)


def process_father_contour(cropped):
    # 查找轮廓，使用 RETR_CCOMP 模式来获取裁剪部分的外轮廓和内轮廓
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化变量
    max_area = 0
    max_contour = None
    max_index = -1  # 用于保存最大面积父轮廓的索引
    father_contour_num = 0

    # 确保有轮廓被找到并且 hierarchy 不为空
    if contours is not None and hierarchy is not None:
        hierarchy = hierarchy[0]  # 展平 hierarchy 数组

        for index, contour in enumerate(contours):
            # 获取当前轮廓的层次信息
            _, _, _, parent = hierarchy[index]
            
            # 只考虑最外层轮廓 (parent == -1)
            if parent == -1:
                area = cv2.contourArea(contour)
                # print(f"父轮廓 {index} 的面积: {area:.2f} 像素")
                father_contour_num += 1
                
                # 更新最大面积的轮廓及其索引
                if area > max_area:
                    max_area = area
                    max_contour = contour
                    max_index = index  # 保存索引

    # print(f"有{father_contour_num}个外轮廓")

    return father_contour_num, max_contour, max_index, contours, hierarchy

def get_father_contour_count(cropped):
    # 查找轮廓，使用 RETR_CCOMP 模式来获取裁剪部分的外轮廓和内轮廓
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    father_contour_num = 0

    # 确保有轮廓被找到并且 hierarchy 不为空
    if contours is not None and hierarchy is not None:
        hierarchy = hierarchy[0]  # 展平 hierarchy 数组

        for index, contour in enumerate(contours):
            # 获取当前轮廓的层次信息
            _, _, _, parent = hierarchy[index]
            
            # 只考虑最外层轮廓 (parent == -1)
            if parent == -1:
                father_contour_num += 1

    print(f" （1）有{father_contour_num}个外轮廓")

    return father_contour_num






def fit_and_draw_circles(mask_image, bbox_list, round=1, DEBUG=False, Default_coefficient=0.5):

    results = []
    if mask_image is None:
        return []
    # mask_image = mask_image_input.copy()

    # 转换为灰度图
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # 对灰度图进行二值化处理
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)


    # --- 裁剪并显示每个 bbox 区域 ---
    for i, bbox in enumerate(bbox_list):
        if DEBUG:
            print(f"\n 显示裁剪的 第 { i+1 } 个 bbox")
        # 将 tensor 转为 CPU numpy 并取整
        # box = bbox.numpy().astype(int)
        x1, y1, x2, y2 = bbox

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(binary.shape[1], x2)
        y2 = min(binary.shape[0], y2)

        # 计算宽和高
        cropped_width = x2 - x1
        cropped_height = y2 - y1
        
        # 打印每个bbox的宽和高
        if DEBUG:
            print(f"Boundary Box: ({x1}, {y1}, {x2}, {y2}) - Width: {cropped_width}, Height: {cropped_height}")
        cropped_avg = (cropped_width + cropped_height)/4

        # 裁剪图像
        cropped = binary[y1:y2, x1:x2]

        # 检查裁剪区域是否为空
        if cropped.size == 0:
            print(f"⚠️  第 {i+1} 个 bbox 裁剪区域为空，跳过显示")
            continue

        # if DEBUG:
        #     # 显示裁剪图像
        #     cv2.imshow('cropped_image', cropped)
        #     cv2.waitKey(0)


        father_contour_num, max_contour, max_index, contours, hierarchy = process_father_contour(cropped)

        if DEBUG:
            print(f" 有 {father_contour_num} 个外轮廓")


        if  max_contour is not None:
            # 获取当前最大外轮廓的层次信息
            next_contour, previous_contour, first_child, parent = hierarchy[max_index]

            # 判断是否为外轮廓（无父轮廓）
            if parent == -1:
                # 外轮廓
                children_count = count_children(hierarchy, max_index)
                if DEBUG:
                    print(f"该最大外轮廓 有 {children_count} 个子轮廓")

                outer_contour = contours[max_index]
                (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(outer_contour)
                outer_center = (int(outer_x + x1), int(outer_y + y1))
                outer_radius = int(outer_radius)

                ERROR_FLAG = False
                if outer_radius < cropped_avg * 0.8 or outer_radius > cropped_avg * 1.1:
                    print(f"⚠️    拟合外圆过小或者过大，使用bbox边框边长代替直径赋值")
                    outer_radius = int(cropped_avg)
                    outer_center = (int((x1+x2)/2), int((y1+y2)/2))
                    ERROR_FLAG = True
                
                # 绘制外轮廓的圆
                # cv2.circle(mask_image, outer_center, outer_radius, (0, 255, 0), 2)  # 绿色
                # cv2.circle(image, outer_center, 2, (0, 0, 255), 3)  # 红色圆心
                if DEBUG:
                    cv2.imshow('draw_circle', mask_image)
                    cv2.waitKey(0)
                # 输出外轮廓圆的尺寸
                # print(f"Outer circle diameter: {outer_radius * 2} pixels")
                

                #####################################################################################
                # 检查是否有子轮廓（内轮廓）
                if children_count == 1:
                    # 存在子轮廓
                    inner_contour = contours[first_child]
                    (inner_x, inner_y), inner_radius = cv2.minEnclosingCircle(inner_contour)
                    inner_center = (int(inner_x + x1), int(inner_y + y1))
                    inner_radius = int(inner_radius)
                    
                    # 绘制内轮廓的圆
                    # cv2.circle(mask_image, inner_center, inner_radius, (255, 0, 0), 2)  # 蓝色
                    # cv2.circle(mask_image, inner_center, 2, (0, 0, 255), 3)  # 红色圆心
                    
                    # 输出内轮廓圆的尺寸
                    # print(f"Inner circle diameter: {inner_radius * 2} pixels")
                elif children_count == 0:
                    # 没有子轮廓，视为异常值
                    # print(f"Warning: Outer contour at {outer_center} has no inner contour (anomaly detected)")

                    inner_center, inner_radius, _ = Incomplete_circle_process_known_outer(cropped, (outer_x, outer_y), outer_radius, Default_coefficient=Default_coefficient)
                    # inner_center, inner_radius, _ = Incomplete_circle_process(cropped)

                    # print(f"Inner circle center: {inner_center}")
                    inner_center[0] += x1
                    inner_center[1] += y1

                    # cv2.circle(mask_image, (inner_center[0], inner_center[1]), inner_radius, (255, 0, 0), 2)

                    inner_center = tuple(inner_center)      # 统一转换为 (x, y) 的形式

                    # print(f"Inner circle diameter: {inner_radius * 2} pixels")
                elif children_count > 1:
                    # Connected_circles_process(roi)
                    inner_center = outer_center
                    inner_radius = int(outer_radius * Default_coefficient)

                distance_between_centers = math.sqrt((inner_center[0] - outer_center[0]) ** 2 + (inner_center[1] - outer_center[1]) ** 2)
                if DEBUG:
                    print(f"Distance between centers: {distance_between_centers} pixels")

                if distance_between_centers >= inner_radius * 0.8:
                    print(f"⚠️    内外圆心之间的距离大于内圆半径的80%，可能有错误")
                    ERROR_FLAG = True

                if ERROR_FLAG:
                    inner_center = outer_center
                    inner_radius = int(outer_radius * Default_coefficient)

                # cv2.circle(mask_image, inner_center, inner_radius, (255, 0, 0), 2)  # 蓝色
                if DEBUG:
                    cv2.imshow('draw_circle', mask_image)
                    cv2.waitKey(0)

                    print(f"内圆圆心 ：{inner_center}，内圆半径 ：{inner_radius}，外圆圆心 ：{outer_center}，外圆半径 ：{outer_radius}")
                
                results.append((inner_center, inner_radius, outer_center, outer_radius))

            else:
                # 内轮廓已经在其父轮廓处理时被处理，跳过
                continue

    # 显示结果
    if DEBUG:
        cv2.imshow("Detected Circles", mask_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results



if __name__ == "__main__":
    image_path = '/home/hence/WorkPlace/RoboCup/IM2025/IM2025/working/circle_mask/images/003.png'
    results = fit_and_draw_circles(image_path)

    # result 格式为：[(inner_center, inner_radius, outer_center, outer_radius), ...]
    # print(result)