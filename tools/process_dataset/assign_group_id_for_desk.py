import os
import json


def update_group_ids(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 更新所有shapes的group_id为0
            if 'shapes' in data:
                for shape in data['shapes']:
                    shape['group_id'] = 0

            # 将修改后的数据写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)

            print(f"Updated {filename}")


# 使用示例：将'your_folder_path'替换为实际文件夹路径
update_group_ids(r'C:\Users\Wtz\Desktop\data\LABELME_desk')