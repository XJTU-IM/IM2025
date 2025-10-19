import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.9):
    """
    将图片和标签文件按比例分为训练集和验证集。

    参数:
        images_dir (str): 存放图片的文件夹路径。
        labels_dir (str): 存放标签的文件夹路径。
        output_dir (str): 输出文件夹路径，用于保存划分后的数据集。
        train_ratio (float): 训练集的比例，默认为 0.9（即 9:1 划分）。

    返回:
        None
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建训练集和验证集的子文件夹
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图片文件的文件名（不带扩展名）
    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 按比例划分训练集和验证集
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 复制训练集文件
    for file in train_files:
        # 复制图片
        shutil.copy(
            os.path.join(images_dir, file + ".jpg"),
            os.path.join(train_images_dir, file + ".jpg")
        )
        # 复制标签
        shutil.copy(
            os.path.join(labels_dir, file + ".txt"),
            os.path.join(train_labels_dir, file + ".txt")
        )

    # 复制验证集文件
    for file in val_files:
        # 复制图片
        shutil.copy(
            os.path.join(images_dir, file + ".jpg"),
            os.path.join(val_images_dir, file + ".jpg")
        )
        # 复制标签
        shutil.copy(
            os.path.join(labels_dir, file + ".txt"),
            os.path.join(val_labels_dir, file + ".txt")
        )

    print(f"数据集划分完成！\n训练集: {len(train_files)} 个样本\n验证集: {len(val_files)} 个样本")

if __name__ == "__main__":
    # img_path = "../dataset/train/images" 
    # label_path = "../dataset/train/labels"
    img_path = "../yihan/pic"
    label_path = "../yihan/labels"
    output_dir = "spliy"
    # 找出不匹配的文件
    split_dataset(img_path, label_path, output_dir=output_dir)

 
