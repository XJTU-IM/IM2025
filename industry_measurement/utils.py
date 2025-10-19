import json
import math
import cv2
import numpy as np


# def fit_circle(point1, point2, point3):
#
#     # 提取点的坐标
#     x1, y1 = point1
#     x2, y2 = point2
#     x3, y3 = point3
#
#     # 计算中间变量
#     A = x2 - x1
#     B = y2 - y1
#     C = x3 - x1
#     D = y3 - y1
#     E = (x2 ** 2 - x1 ** 2) + (y2 ** 2 - y1 ** 2)
#     F = (x3 ** 2 - x1 ** 2) + (y3 ** 2 - y1 ** 2)
#     G = 2 * (x2 * y1 - x1 * y2)
#     H = 2 * (x3 * y1 - x1 * y3)
#
#     # 求解圆心坐标
#     if (A * D - B * C) == 0:
#         return None
#
#     cx = (D * E - B * F) / (2 * (A * D - B * C))
#     cy = (A * F - C * E) / (2 * (A * D - B * C))
#
#     return (cx, cy)


def l2_distance(point1, point2):
    """
    计算两点之间的距离。

    参数:
        point1, point2: 两点的坐标，格式为 (x, y)

    返回:
        两点之间的距离
    """
    # 提取点的坐标
    x1, y1 = point1
    x2, y2 = point2

    # 计算距离
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance

def locate_region(img, point, split_num):
    # split_num = 5是第一轮， =2是第二轮

    width, height = img.shape[0:2]

    # 计算每个区域的宽度和高度
    region_width = width / split_num
    region_height = height / split_num

    # 获取点的坐标
    x, y = point

    # 判断点所在的列和行
    col = int(x // region_width) + 1
    row = int(y // region_height) + 1

    # 计算区域索引
    if split_num == 2:
        # 使用自定义的编号顺序：1 2 / 4 3
        if row == 1:
            region_index = col
        elif row == 2:
            region_index = 5 - col  # 第二行：col=1 -> 4, col=2 -> 3
    else:
        region_index = (row - 1) * split_num + col

    return region_index

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        return model_config

