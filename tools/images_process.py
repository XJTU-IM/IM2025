import os
from PIL import Image

def process_images(folder_path):
    # 支持的图片格式
    supported_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.webp']
    
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 创建输出文件夹（如果不存在）
    output_folder = os.path.join(folder_path, 'processed')
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为文件且扩展名在支持列表中
        if os.path.isfile(file_path) and any(filename.lower().endswith(fmt) for fmt in supported_formats):
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 转换为RGB模式（处理PNG等带透明通道的图片）
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # 调整图片大小为640x640，保持宽高比
                    img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    
                    # 创建640x640的白色背景
                    new_img = Image.new('RGB', (640, 640), (255, 255, 255))
                    
                    # 将调整后的图片居中粘贴到白色背景上
                    offset = ((640 - img.size[0]) // 2, (640 - img.size[1]) // 2)
                    new_img.paste(img, offset)
                    
                    # 生成新的文件名（使用原始文件名，但改为.jpg扩展名）
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    output_path = os.path.join(output_folder, new_filename)
                    
                    # 保存处理后的图片
                    new_img.save(output_path, 'JPEG', quality=95)
                    print(f"已处理: {filename} -> {new_filename}")
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
    
    print("处理完成！")

if __name__ == "__main__":
    # 获取用户输入的文件夹路径
    folder_path = r"C:\Users\Wtz\Desktop\train"
    process_images(folder_path)
