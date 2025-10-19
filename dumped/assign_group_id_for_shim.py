import json
import os
import shutil


def assign_group_ids(input_json_path, output_json_path):
    # 读取JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有已存在的group_id
    existing_group_ids = set()
    for shape in data['shapes']:
        if shape['group_id'] is not None:
            existing_group_ids.add(shape['group_id'])

    # 为检测框分配group_id
    next_group_id = 0
    while next_group_id in existing_group_ids:
        next_group_id += 1

    rectangle_shapes = [shape for shape in data['shapes'] if shape['shape_type'] == 'rectangle' and shape['group_id'] is None]

    for rectangle in rectangle_shapes:
        while next_group_id in existing_group_ids:
            next_group_id += 1
        rectangle['group_id'] = next_group_id
        existing_group_ids.add(next_group_id)
        next_group_id += 1

    # 为关键点分配group_id
    point_shapes = [shape for shape in data['shapes'] if shape['shape_type'] == 'point' and shape['group_id'] is None]

    for point in point_shapes:
        point_x, point_y = point['points'][0]
        assigned = False

        for rectangle in data['shapes']:
            if rectangle['shape_type'] != 'rectangle':
                continue

            # 提取矩形的四个顶点坐标
            rect_points = rectangle['points']
            rect_left = min(p[0] for p in rect_points)
            rect_right = max(p[0] for p in rect_points)
            rect_top = min(p[1] for p in rect_points)
            rect_bottom = max(p[1] for p in rect_points)

            # 判断点是否在矩形内
            if rect_left <= point_x <= rect_right and rect_top <= point_y <= rect_bottom:
                point['group_id'] = rectangle['group_id']
                assigned = True
                break

        if not assigned:
            # 如果没有找到对应的矩形，也可以分配一个新的group_id，但根据需求可能不需要
            # while next_group_id in existing_group_ids:
            #     next_group_id += 1
            # point['group_id'] = next_group_id
            # existing_group_ids.add(next_group_id)
            # next_group_id += 1
            pass

    # 写回JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"处理完成，已保存到 {output_json_path}")

def batch_assign(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith("json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            assign_group_ids(input_path, output_path)
            print(f"已处理：{filename}")

            # 把对应的图像也复制过去
            image_filename = os.path.splitext(filename)[0] + '.png'
            image_input_path = os.path.join(input_dir, image_filename)
            image_output_path = os.path.join(output_dir, image_filename)

            if os.path.exists(image_input_path):
                shutil.copy2(image_input_path, image_output_path)
                print(f"已复制：{image_filename}")
            else:
                print(f"未找到同名的PNG图像文件：{image_filename}")
if __name__ == "__main__":

    batch_assign(r"C:\Users\Wtz\Desktop\data\LABELME_shim", r"C:\Users\Wtz\Desktop\data\LABELME_shim2")