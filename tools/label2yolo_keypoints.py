import json
import os
import glob
import shutil
import random
from pathlib import Path
from typing import List, Tuple

bbox_class = None
keypoint_class = None


# 划分训练集和测试集
def split_dataset_yolo(src_dir: str,
                  dst_dir: str,
                  split_ratio: float = 0.8,
                  seed: int = 42) -> None:
    # 1. 设置随机种子
    random.seed(seed)

    # 2. 收集所有图像文件及其对应 json
    img_exts = ('*.jpg', '*.jpeg', '*.png')
    img_files: List[Path] = []
    for ext in img_exts:
        img_files.extend(Path(src_dir).glob(ext))
    img_files = sorted(img_files)

    pairs: List[Tuple[Path, Path]] = []
    for img in img_files:
        ann = img.with_suffix('.json')
        if ann.exists():
            pairs.append((img, ann))
        else:
            print(f'[Warn] 未找到对应标注：{img.name}')

    if not pairs:
        raise RuntimeError('未找到任何图像或对应的 json 文件')

    # 3. 随机划分
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # 4. 创建输出目录
    for sub in ('images/train', 'images/val', 'labels_json/train', 'labels_json/val'):
        (Path(dst_dir) / sub).mkdir(parents=True, exist_ok=True)

    # 5. 复制文件
    def copy_pair(pairs: List[Tuple[Path, Path]], mode: str) -> None:
        for img, ann in pairs:
            shutil.copy2(img,  Path(dst_dir) / 'images' / mode / img.name)
            shutil.copy2(ann, Path(dst_dir) / 'labels_json' / mode / ann.name)

    copy_pair(train_pairs, 'train')
    copy_pair(val_pairs, 'val')

    print(f'完成！共 {len(pairs)} 张图，训练集 {len(train_pairs)}，验证集 {len(val_pairs)}')


def process_single_json(labelme_path, save_folder):
    """
    将单个 labelme JSON 转换为 YOLO 关键点格式 txt。
    """
    global bbox_class, keypoint_class

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']
    img_height = labelme['imageHeight']

    # 输出 txt 路径
    txt_name = Path(labelme_path).stem + '.txt'
    yolo_txt_path = Path(save_folder) / txt_name

    # 先按 group_id 把“框”和“点”分别收集
    box_dict = {}      # {group_id: {...框信息}}
    kpts_dict = {}     # {group_id: {label: [x, y]}}

    for shape in labelme['shapes']:
        gid = shape.get('group_id')
        if gid is None:          # 跳过未编组
            continue

        if shape['shape_type'] == 'rectangle':
            # 一个 group 只能有一个框，若出现多个以最后一条为准
            box_dict[gid] = shape
        elif shape['shape_type'] == 'point':
            # 同一个 group 下可以有多类关键点
            kpts_dict.setdefault(gid, {})[shape['label']] = shape['points'][0]

    # 写 YOLO txt
    with open(yolo_txt_path, 'w', encoding='utf-8') as f_txt:
        for gid, box_shape in box_dict.items():
            # ---------------- 框信息 ----------------
            label = box_shape['label']
            class_id = bbox_class[label]

            pts = box_shape['points']
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1

            cx /= img_width
            cy /= img_height
            w  /= img_width
            h  /= img_height

            line = f"{class_id} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}"

            # ---------------- 关键点信息 ----------------
            # 按 keypoint_class 顺序输出
            for kp_name in keypoint_class:
                if gid in kpts_dict and kp_name in kpts_dict[gid]:
                    x, y = kpts_dict[gid][kp_name]
                    x /= img_width
                    y /= img_height
                    line += f" {x:.5f} {y:.5f} 2"   # 可见
                else:
                    line += " 0 0 0"               # 不存在

            f_txt.write(line + '\n')

    print(f'{labelme_path} --> {yolo_txt_path}  转换完成')


def batch_convert(labelme_path, yolo_path):
    # 训练集
    train_json_dir = Path(yolo_path) / 'labels_json' / 'train'
    save_folder = Path(yolo_path) / 'labels' / 'train'
    for json_file in train_json_dir.glob('*.json'):
        try:
            process_single_json(str(json_file), save_folder=str(save_folder))
        except Exception as e:
            print('******有误******', json_file, e)

    # 验证集
    val_json_dir = Path(yolo_path) / 'labels_json' / 'val'
    save_folder = Path(yolo_path) / 'labels' / 'val'
    for json_file in val_json_dir.glob('*.json'):
        try:
            process_single_json(str(json_file), save_folder=str(save_folder))
        except Exception as e:
            print('******有误******', json_file, e)


def main(mode:str):
    global bbox_class, keypoint_class
    if mode == "screw":
        bbox_class = {"screw": 0}  # {"screw": 0, ...}
        keypoint_class = ['top_left', 'top_right', 'bot_left', 'bot_right']
        labelme_path = r"C:\Users\Wtz\Desktop\data\LABELME_screw"
        yolo_path = r"C:\Users\Wtz\Desktop\data\YOLO_screw"
    elif mode == "shim":
        bbox_class = {"shim": 0}  # {"screw": 0, ...}
        keypoint_class = ['wai_1', 'wai_2', 'wai_3', 'nei_1', 'nei_2', 'nei_3']
        labelme_path = r"C:\Users\Wtz\Desktop\data\LABELME_shim"
        yolo_path = r"C:\Users\Wtz\Desktop\data\YOLO_shim"
    elif mode == "desk":
        bbox_class = {"desk": 0}  # {"screw": 0, ...}
        keypoint_class = ['lt', 'rt', 'rb', 'lb']
        labelme_path = r"C:\Users\Wtz\Desktop\data\LABELME_desk"
        yolo_path = r"C:\Users\Wtz\Desktop\data\YOLO_desk"
    else:
        raise ValueError("请选择正确的模式 screw or shim, desk")

    # 第一步 划分数据集
    split_dataset_yolo(labelme_path, yolo_path)

    # 第二步， json转txt
    batch_convert(labelme_path=labelme_path, yolo_path=yolo_path)

    # 第三步， 删除多余的json标注
    json_folder = yolo_path + "\\labels_json"
    if os.path.exists(json_folder):
        shutil.rmtree(json_folder)
        print("已删除 ", json_folder)
    else:
        print(f"{json_folder} 不存在，无需删除。")
    print("完成")

if __name__ == "__main__":
    """
    关键点检测的标注转换工具
    我想实现的功能是什么呢，当我已经把所有的标注和图片放到一个LABEL_{MODE}文件夹后(已添加group id),这个脚本可以一键完成接下来所有操作，包括：
    1.划分数据集
    2.将json转为txt
    3.删除原来的json标注
    参数只有"screw"和"shim"
    在此之前需要根据文件所在位置修改目录， 并设置id
    """
    main(mode="screw")