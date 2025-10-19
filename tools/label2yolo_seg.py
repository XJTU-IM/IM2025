"""
本脚本用于将标注的垫片json文件转为yolo-seg所需的txt文件，主要分为两步
第一步，将circle转为polygon
第二步，将json转为txt
"""
import json
import math
import os
from pathlib import Path

import random, shutil
import cv2
import numpy as np
from tqdm import tqdm


def circle_to_polygon(cx, cy, r, n=16):
    """返回 [[x1,y1], [x2,y2], ...] 的多边形点"""
    return [[cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)]
            for i in range(n)]


def convert_one(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for shape in data['shapes']:
        if shape.get('shape_type') == 'circle':
            (cx, cy), (x2, y2) = shape['points']
            r = math.dist((cx, cy), (x2, y2))
            shape['points'] = circle_to_polygon(cx, cy, r)
            shape['shape_type'] = 'polygon'
            changed = True

    if changed:                       # 只有真的改了才写回
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def json2txt(json_path, txt_path, mode):
    if mode == "shim":
        classes = ['wai', 'nei']
    elif mode == "circle":
        classes = ['circle']

    Path(txt_path).mkdir(parents=True, exist_ok=True)

    for json_file in Path(json_path).glob('*.json'):
        name_stem = json_file.stem

        # 1. 先找 png，再找 jpg/JPG
        img_file = None
        for ext in ['.png', '.jpg', '.JPG']:
            tmp = json_file.with_suffix(ext)
            if tmp.exists():
                img_file = tmp
                break
        if img_file is None:
            print(f'跳过：找不到图片 {json_file.with_suffix(".*")}')
            continue

        # 读图片拿宽高
        image = cv2.imread(str(img_file))
        if image is None:
            print(f'跳过：cv2 读图失败 {img_file}')
            continue
        h, w = image.shape[:2]

        # 读 JSON
        with open(json_file, encoding='utf-8') as f:
            masks = json.load(f)['shapes']

        # 写 txt
        txt_out = Path(txt_path) / f'{name_stem}.txt'
        with open(txt_out, 'w', encoding='utf-8') as f_txt:
            for idx, mask_data in enumerate(masks):
                mask_label = mask_data['label']
                if '_' in mask_label:
                    mask_label = mask_label.split('_')[0]
                idx_cls = classes.index(mask_label)

                mask = np.array(mask_data['points'], dtype=np.float32)
                mask[:, 0] /= w
                mask[:, 1] /= h
                mask = mask.reshape(-1)

                if idx != 0:
                    f_txt.write('\n')
                f_txt.write(f'{idx_cls} {" ".join(map(lambda x: f"{x:.6f}", mask))}')
            

def split_dataset(base_path, dataset_path, val_size=0.1):
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/val',   exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/val',   exist_ok=True)

    # 收集所有有 txt 的 basename
    path_list = np.array([p.stem for p in Path(base_path).glob('*.txt')])
    random.shuffle(path_list)
    n_val = int(len(path_list) * val_size)
    train_id = path_list[:-n_val]
    val_id   = path_list[-n_val:]

    def copy_with_ext(stem, split):
        """把 stem.png/jpg/JPG 和 stem.txt 一起拷到对应目录"""
        for ext in ['.png', '.jpg', '.JPG']:
            img_src = Path(base_path) / f'{stem}{ext}'
            if img_src.exists():
                shutil.copy(img_src,
                            f'{dataset_path}/images/{split}/{stem}{ext}')
                break
        else:
            print(f'警告：未找到 {stem} 的图片')
        shutil.copy(f'{base_path}/{stem}.txt',
                    f'{dataset_path}/labels/{split}/{stem}.txt')

    for i in train_id:
        copy_with_ext(i, 'train')
    for i in val_id:
        copy_with_ext(i, 'val')


def main(mode):
    # 输入文件夹
    json_path = r"C:\Users\Wtz\Desktop\data\LABELME_circle"
    txt_path = r"C:\Users\Wtz\Desktop\data\LABELME_circle"
    dataset_path = r"C:\Users\Wtz\Desktop\data\YOLO_circle"

    json2txt(json_path, txt_path, mode)
    print("完成")

    # 划分数据集
    print("正在划分数据集")
    split_dataset(json_path, dataset_path)
    print("划分数据集完成")

if __name__ == "__main__":
    # mode 从 "circle" 和 "shim" 中选择
    main("circle")