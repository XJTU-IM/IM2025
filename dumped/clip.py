import os
import json
import numpy as np
import cv2
import time


def extract_points(data):
    points = {
        'lt': None,
        'rt': None,
        'rb': None,
        'lb': None
    }

    for shape in data['shapes']:
        label = shape['label']
        if label in points:
            points[label] = shape['points'][0]

    return points


def clip_image(folder_path, output_path):
    count = 0
    for filename in os.listdir(folder_path):
        count += 1
        if filename.endswith("json"):
            file_path = os.path.join(folder_path, filename)
            # print(f"file:{file_path}")
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                points = extract_points(data)
                lt, rt, rb, lb = points["lt"], points["rt"], points["rb"], points["lb"]

                # 对原始图像做变换
                img_file_name = f'{os.path.splitext(os.path.basename(file_path))[0]}.png'
                img_file_path = os.path.join(folder_path, img_file_name)
                img = cv2.imread(img_file_path)
                print("processing: ", img_file_name)
                rect = np.array([lt, rt, rb, lb], dtype="float32")
                pts = np.array([[0,0], [640-1,0], [640-1,640-1], [0,640-1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, pts)
                warped = cv2.warpPerspective(img, M, (640,640))
                # cv2.imshow("warp", warped)
                # cv2.imshow("orign", img)
                # cv2.waitKey(0)
                # return
                opath = os.path.join(output_path, img_file_name)
                cv2.imwrite(opath, warped)

    print(count/2)

source = r'C:\Users\Wtz\Desktop\mmb'
output_path = r"C:\Users\Wtz\Desktop\cliped_data"
clip_image(source, output_path)

# with open(r"C:\Users\Wtz\Desktop\mmb\screenshot_20250428_201218.json") as f:
#     data = json.load(f)
#     lt = data['shapes'][1]['points'][0]
#     print(lt)
#     rt = data['shapes'][2]['points'][0]
#     print(rt)
#     rb = data['shapes'][3]['points'][0]
#     print(rb)
#     lb = data['shapes'][4]['points'][0]
#     print(lb)

