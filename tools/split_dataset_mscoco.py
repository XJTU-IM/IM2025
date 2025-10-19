import os
import shutil
import random

def divide_dataset(input_folder, output_folder):
    # 检查输入和输出文件夹是否存在，不存在则创建
    if not os.path.exists(input_folder):
        print("输入文件夹不存在")
        return

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train_label"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "val_label"), exist_ok=True)

    # 获取输入文件夹中的所有文件
    files = os.listdir(input_folder)

    # 遍历文件，分离图片和 JSON 文件
    images = []
    json_files = []
    for file in files:
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            # 假设图片文件的扩展名是 .jpg、.png 等，可以根据实际需要修改
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(file)
            elif file.lower().endswith('.json'):
                json_files.append(file)

    # 确保图片和 JSON 文件数量一致（假设每个图片对应一个 JSON 文件）
    if len(images) != len(json_files):
        print("图片和 JSON 文件数量不一致")
        return

    # 按 4:1 的比例划分训练集和验证集
    random.shuffle(json_files)
    train_ratio = 0.8  # 训练集比例
    train_count = int(len(json_files) * train_ratio)
    train_json = json_files[:train_count]
    val_json = json_files[train_count:]

    # 复制文件到目标文件夹
    for image in images:
        # 复制所有图片到 image 文件夹
        src_image_path = os.path.join(input_folder, image)
        dst_image_path = os.path.join(output_folder, "images", image)
        shutil.copy2(src_image_path, dst_image_path)

    for json_file in train_json:
        # 复制训练集 JSON 文件到 train_label 文件夹
        src_json_path = os.path.join(input_folder, json_file)
        dst_json_path = os.path.join(output_folder, "train_label", json_file)
        shutil.copy2(src_json_path, dst_json_path)

    for json_file in val_json:
        # 复制验证集 JSON 文件到 val_label 文件夹
        src_json_path = os.path.join(input_folder, json_file)
        dst_json_path = os.path.join(output_folder, "val_label", json_file)
        shutil.copy2(src_json_path, dst_json_path)

    print("数据集划分完成")

# 使用示例
input_folder = r"C:\Users\Wtz\Desktop\data\LABELME_desk"
output_folder = r"C:\Users\Wtz\Desktop\data\MSCOCO_desk"
divide_dataset(input_folder, output_folder)