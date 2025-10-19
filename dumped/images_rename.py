import os
from pathlib import Path
import shutil

def rename_images(folder1_path: str, folder2_path: str, output_path: str):
    """
    批量重命名两个文件夹中的图片文件并保存到输出文件夹
    
    Args:
        folder1_path: 第一个源文件夹路径
        folder2_path: 第二个源文件夹路径
        output_path: 输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 获取两个文件夹中的所有图片文件
    images1 = [f for f in Path(folder1_path).iterdir() if f.suffix.lower() in image_extensions]
    images2 = [f for f in Path(folder2_path).iterdir() if f.suffix.lower() in image_extensions]
    
    # 合并两个文件夹的图片列表
    all_images = images1 + images2
    
    # 按文件名排序（可选）
    all_images.sort()
    
    # 重命名并复制文件
    for index, image_path in enumerate(all_images, start=1):
        # 生成新文件名（4位数字，例如：0001）
        new_name = f"{index:04d}{image_path.suffix}"
        
        # 构建目标路径
        destination = os.path.join(output_path, new_name)
        
        # 复制并重命名文件
        shutil.copy2(image_path, destination)
    
    print(f"完成重命名！共处理了 {len(all_images)} 个图片文件")

def rename_images_from_number(folder_path: str, output_path: str, start_number: int):
    """
    批量重命名单个文件夹中的图片文件，从指定数字开始编号
    
    Args:
        folder_path: 源文件夹路径
        output_path: 输出文件夹路径
        start_number: 起始编号
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 获取文件夹中的所有图片文件
    images = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in image_extensions]
    
    # 按文件名排序
    images.sort()
    
    # 重命名并复制文件
    for index, image_path in enumerate(images):
        # 生成新文件名（4位数字，从start_number开始）
        new_name = f"{start_number + index:04d}{image_path.suffix}"
        
        # 构建目标路径
        destination = os.path.join(output_path, new_name)
        
        # 复制并重命名文件
        shutil.copy2(image_path, destination)
    
    print(f"完成重命名！共处理了 {len(images)} 个图片文件")

# 使用示例
if __name__ == "__main__":
    # 替换为实际的文件夹路径
    # folder1 = r"C:\Users\Wtz\Desktop\pyorbbecsdk\examples\images"
    # folder2 = r"C:\Users\Wtz\Desktop\pyorbbecsdk\examples\save"
    # output = r"C:\Users\Wtz\Desktop\images"
    
    # rename_images(folder1, folder2, output)

    # 使用示例
    folder = r"F:\IM2025\images\circle"
    # folder = r"F:\IM2025\images\desk"
    output = r"C:\Users\Wtz\Desktop\aa"
    start_num = 499  # 起始编号，例如从123开始
    
    rename_images_from_number(folder, output, start_num)
