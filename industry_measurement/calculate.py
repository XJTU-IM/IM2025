from utils import l2_distance

def calculate_screw(img, bboxes, keypoints, depth_img=None, round=1):
    """
    bbox : (num_obj, 4), ndarray uint32
    keypoints : (num_obj, 4, 2) np.float32/64    top_left, top_right, bot_left, bot_right
    depth_img : (height, width) ndarray
    圆桌和方桌的Goal_A, Goal_B是相同的, 只是Goal_C, Goal_D 不同
    """

    if round == 1:  # 方桌
        gamma = 550 / 640
        results = []
        
        for idx, keypoint in enumerate(keypoints):
            bbox = bboxes[idx]
            result = {}
            result["Goal_ID"] = 1

            # 如果一个目标检测不全， 粗略估计
            if len(keypoint) !=4:
                print("发现一个螺钉目标关键点缺失, bbox信息", bbox)
                x1, y1, x2, y2 = bbox
                result["Goal_B"] = l2_distance((x1, y1), (x2, y2)) * gamma  # 把长度设为对角线长度
                result["Goal_A"] = abs(x1 - x2) * gamma # 把宽度设为bbox宽度
                center = ((x1 + x2)/2, (y1+y2)/2)
                
                result["Goal_C"] = center[0] * gamma
                result["Goal_D"] = center[1] * gamma

                results.append(result)

                continue

            top_left, top_right, bot_left, bot_right = keypoint

            # 计算区域
            center_x = (top_left[0] + top_right[0] + bot_left[0] + bot_right[0]) / 4
            center_y = (top_left[1] + top_right[1] + bot_left[1] + bot_right[1]) / 4
            center = (center_x, center_y)

            result["Goal_C"] = center_x * gamma
            result["Goal_D"] = center_y * gamma

            # 计算Goal_A - 螺纹部分宽度
            result["Goal_A"] = l2_distance(bot_left, bot_right) * gamma

            # 计算Goal_B - 长度
            if is_stand(keypoint):  # 判断是不是直立的
                print(f"区域{result['Goal_C']} 发现一个直立螺栓")
                if depth_img is None:
                    result["Goal_B"] = 1.0  # 测试需要
                else:
                    # # 用depth_img bbox区域内最高值减去最低值
                    # x1, y1, x2, y2 = map(int, bbox)  # 转成整数索引
                    # # 防越界
                    # x1 = max(x1, 0)
                    # y1 = max(y1, 0)
                    # x2 = min(x2, depth_img.shape[1])
                    # y2 = min(y2, depth_img.shape[0])

                    # roi_depth = depth_img[y1:y2, x1:x2]
                    # if roi_depth.size > 0:
                    #     min_depth = np.min(roi_depth)
                    #     max_depth = np.max(roi_depth)
                    result["Goal_B"] = 50.0 # 给一个定值
                    pass
            else:
                result["Goal_B"] = l2_distance((top_left + top_right) / 2, (bot_left + bot_right) / 2)  * gamma

            results.append(result)

        return results
    
    elif round == 2 : # 圆桌
        gamma = 600 / 640
        results = []
    
        for idx, keypoint in enumerate(keypoints):
            bbox = bboxes[idx]
            result = {}
            result["Goal_ID"] = 1

            # 如果一个目标检测不全， 粗略估计
            if len(keypoint) !=4:
                print("发现一个螺钉目标关键点缺失, bbox信息", bbox)
                x1, y1, x2, y2 = bbox
                result["Goal_B"] = l2_distance((x1, y1), (x2, y2)) * gamma  # 把长度设为对角线长度
                result["Goal_A"] = abs(x1 - x2) * gamma # 把宽度设为bbox宽度
                center = ((x1 + x2)/2, (y1+y2)/2)
                
                result["Goal_C"] = l2_distance(center, (320, 320)) * gamma # 第二轮goal_c为中心到圆心的距离
                result["Goal_D"] = 0

                results.append(result)

                continue

            top_left, top_right, bot_left, bot_right = keypoint

            # 计算区域
            center_x = (top_left[0] + top_right[0] + bot_left[0] + bot_right[0]) / 4
            center_y = (top_left[1] + top_right[1] + bot_left[1] + bot_right[1]) / 4
            center = (center_x, center_y)

            result["Goal_C"] = l2_distance(center, (320, 320)) * gamma # 第二轮goal_c为中心到圆心的距离
            result["Goal_D"] = 0

            # 计算Goal_A - 螺纹部分宽度
            result["Goal_A"] = l2_distance(bot_left, bot_right) * gamma

            # 计算Goal_B - 长度
            if is_stand(keypoint):  # 判断是不是直立的
                print(f"区域{result['Goal_C']} 发现一个直立螺栓")
                if depth_img is None:
                    result["Goal_B"] = 1.0  # 测试需要
                else:
                    # # 用depth_img bbox区域内最高值减去最低值
                    # x1, y1, x2, y2 = map(int, bbox)  # 转成整数索引
                    # # 防越界
                    # x1 = max(x1, 0)
                    # y1 = max(y1, 0)
                    # x2 = min(x2, depth_img.shape[1])
                    # y2 = min(y2, depth_img.shape[0])

                    # roi_depth = depth_img[y1:y2, x1:x2]
                    # if roi_depth.size > 0:
                    #     min_depth = np.min(roi_depth)
                    #     max_depth = np.max(roi_depth)
                    result["Goal_B"] = 50.0 # 给一个定值
                    pass
            else:
                result["Goal_B"] = l2_distance((top_left + top_right) / 2, (bot_left + bot_right) / 2)  * gamma

            results.append(result)

        return results
    else:
        raise ValueError("config argument 'round' should only be 1 (for desk), 2 (for circle)")

def is_stand(keypoint):
    # 判断螺钉是否直立
    top_left, top_right, bot_left, bot_right = keypoint
    top_center = ((top_left[0] + top_right[0])/2 , (top_left[1] + top_right[1])/2)
    bot_center = ((bot_left[0] + bot_right[0])/2, (bot_left[1] + bot_right[1])/2)

    if l2_distance(bot_center, top_center) < l2_distance(bot_left, bot_right) :
        return True

    return False


def calculate_shim(image, data, round=1):
    if round == 1:
        gamma = 550 / 640
        results = []
        print(data)
        
        for i in range(len(data)):
            result = {}
            result["Goal_ID"] = 2
            result["Goal_A"] = data[i][3] * 2 * gamma  # 外径
            result["Goal_B"] = data[i][1] * 2 * gamma  # 内径

            center = data[i][2] # 外圆中心
            import cv2
            # cv2.circle(image, center, 2, (0, 0, 255), 2)
            result["Goal_C"] = center[0] * gamma # 
            result["Goal_D"] = center[1] * gamma

            results.append(result)
        return results
    
    elif round == 2:
        gamma = 600 / 640
        results = []

        for i in range(len(data)):
            result = {}
            result["Goal_ID"] = 2
            result["Goal_A"] = data[i][3] * 2 * gamma  # 外径
            result["Goal_B"] = data[i][1] * 2 * gamma  # 内径

            center = data[i][2]

            result["Goal_C"] = l2_distance(center, (320, 320)) * gamma # 第二轮goal_c为中心到圆心的距离
            result["Goal_D"] = 0

            results.append(result)
        return results
    else:
        raise ValueError("config argument 'round' should only be 1 (for desk), 2 (for circle)")