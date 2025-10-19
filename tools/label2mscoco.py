import os
import json


coco = {}

Dataset_root = 'stm_pose_Dataset'

# 桌子配置
class_list = {
    'supercategory': 'desk',
    'id': 1,
    'name': 'desk',
    'keypoints': ['lt', 'rt', 'rb', 'lb'], # 大小写敏感
    'skeleton':[[0,1], [1,2], [2,3], [3, 0]] # 可有可无
}

# class_list = {
#     'supercategory': 'shim',
#     'id': 1,
#     'name': 'shim',
#     'keypoints': ['wai_1', 'wai_2', 'wai_3', 'nei_1', 'nei_2', 'nei_3'], # 大小写敏感
#     'skeleton':[[0,1], [1,2], [2,0], [3,4], [4,5], [5,3]] # 可有可无
# }

# 螺钉的配置
# class_list = {
#     'supercategory': 'screw',
#     'id': 1,
#     'name': 'screw',
#     'keypoints': ['top_left', 'top_right', 'bot_left', 'bot_right'], # 大小写敏感
#     'skeleton':[[0,1], [1,2], [2,3], [3,0]] # 可有可无
# }
#
coco['categories'] = []
coco['categories'].append(class_list)

coco['images'] = []
coco['annotations'] = []

def process_single_json(labelme, image_id=1):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''

    global ANN_ID

    coco_annotations = []

    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注

        if each_ann['shape_type'] == 'rectangle':  # 筛选出个体框

            # 个体框元数据
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []

            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = ANN_ID
            # print(ANN_ID)
            ANN_ID += 1

            # 获取个体框坐标
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]), int(each_ann['points'][2][0]),int(each_ann['points'][3][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]), int(each_ann['points'][2][1]), int(each_ann['points'][3][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]), int(each_ann['points'][2][0]),int(each_ann['points'][3][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]), int(each_ann['points'][2][1]), int(each_ann['points'][3][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
            bbox_dict['area'] = bbox_w * bbox_h

            # 筛选出分割多段线
            for each_ann in labelme['shapes']:  # 遍历所有标注
                if each_ann['shape_type'] == 'polygon':  # 筛选出分割多段线标注
                    # 第一个点的坐标
                    first_x = each_ann['points'][0][0]
                    first_y = each_ann['points'][0][1]
                    if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & (
                            first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_dict['segmentation'] = list(
                            map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))  # 坐标保留两位小数
                        # bbox_dict['segmentation'] = each_ann['points']

            # 筛选出该个体框中的所有关键点
            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:  # 遍历所有标注

                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                    # 关键点横纵坐标
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                            y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_keypoints_dict[label] = [x, y]

            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            # print(bbox_keypoints_dict)

            # 把关键点按照类别顺序排好
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class in bbox_keypoints_dict:
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                    bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                else:  # 不存在的点，一律为0
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations


def process_folder(folder_path):
    IMG_ID = 0
    ANN_ID = 0
    # 遍历所有 labelme 格式的 json 文件
    for labelme_json in os.listdir(folder_path):

        if labelme_json.split('.')[-1] == 'json':

            with open(folder_path+'\\\\'+labelme_json, 'r', encoding='utf-8') as f:

                labelme = json.load(f)

                ## 提取图像元数据
                img_dict = {}
                img_dict['file_name'] = labelme['imagePath']
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['id'] = IMG_ID
                coco['images'].append(img_dict)

                ## 提取框和关键点信息
                coco_annotations = process_single_json(labelme, image_id=IMG_ID)
                coco['annotations'] += coco_annotations

                IMG_ID += 1

                print(labelme_json, '已处理完毕')

        else:
            pass



# 转换训练集
# IMG_ID = 0
# ANN_ID = 0
#
# train_label_path = r"C:\Users\Wtz\Desktop\data\MSCOCO_desk\train_label"
# process_folder(train_label_path)
#
# coco_path = r'C:\Users\Wtz\Desktop\data\MSCOCO_desk\train_coco.json'
# with open(coco_path, 'w') as f:
#     json.dump(coco, f, indent=2)
#
# # 测试是否导入成功
# from pycocotools.coco import COCO
# mycoco = COCO(coco_path)
# print(mycoco)   # 如果成功会显示index create!

# 转换验证集
IMG_ID = 0
ANN_ID = 0

val_label_path = r"C:\Users\Wtz\Desktop\data\MSCOCO_desk\val_label"
process_folder(val_label_path)

coco_path = r'C:\Users\Wtz\Desktop\data\MSCOCO_desk\val_coco.json'
with open(coco_path, 'w') as f:
    json.dump(coco, f, indent=2)

from pycocotools.coco import COCO
import pprint
my_coco = COCO(coco_path)
pprint.pprint(my_coco)