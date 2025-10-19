# -*- coding:utf-8 -*-
from pyorbbecsdk import *
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image
from PIL import Image
import numpy as np
import cv2
import os
import time
from loguru import logger

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

def turnon_camera():
    try:
        print("launch camera pipline")
        pipeline = Pipeline()
        if type(pipeline)==type(None):
            logger.info("相机USB接口连接不当")
        config = Config()

        try:
            # 配置彩色传感器
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            assert  profile_list is not None
            # try:
            #     color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
            # except OBError as e:
            #     print(e)

            color_profile = profile_list.get_default_video_stream_profile()
            assert color_profile is not None
            print("color profile: ", color_profile)

            config.enable_stream(color_profile)
            # color_profile = profile_list.get_default_video_stream_profile()
            #color_profile = profile_list.get_stream_profile_by_index(3).as_video_stream_profile()

            # 配置深度传感器
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            depth_profile = profile_list.get_default_video_stream_profile()
            assert depth_profile is not None
            print("depth profile: ", depth_profile)

            # depth_profile = profile_list.get_stream_profile_by_index(0).as_video_stream_profile()

            #print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
            #                                           color_profile.get_height(),
            #                                          color_profile.get_fps(),
            #                                           color_profile.get_format()))
            #print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
            #                                           depth_profile.get_height(),
            #                                           depth_profile.get_fps(),
            #                                           depth_profile.get_format()))
            config.enable_stream(depth_profile)

        except Exception as e:
            print(e)
            return 

        # 此处之前有参数设置
        config.set_align_mode(OBAlignMode.HW_MODE)
        try:
            pipeline.enable_frame_sync()    # 这句话是对齐彩色图和深度图
        except Exception as e:
            print(e)

        pipeline.start(config)

        print("Current Parameters")
        return pipeline
    except RuntimeError:
        print("未检测到摄像头!")
    # finally:
    #     print("摄像机打开!")


def get_data(pipeline):
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        return
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return
    color_image = frame_to_bgr_image(color_frame)
    if color_image is None:
        print("failed to convert frame to image")
        return
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_data = depth_data.astype(np.float32) * scale
    # threshold = 240
    # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#########################################################
    dpt = cv2.convertScaleAbs(depth_data, alpha=0.14 )#####
#########################################################
    # dpt[dpt > threshold] = 0
    depth_image = cv2.applyColorMap(dpt, cv2.COLORMAP_JET)
    # depth_image = cv2.applyColorMap(depth_image, 2)
    # overlay color image on depth image
    # img = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
    cv2.imwrite('/home/nano/Desktop/run/color.png', color_image)
    img = Image.open('/home/nano/Desktop/run/color.png').convert('RGBA')
    imgmat = np.array(img)
    dep = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
    imgmat[:,:,3] = dep
    img = Image.fromarray(imgmat)
    # img = img.resize((1920, 1080))
    #if img is None:
      #  return
    #img.save(r'/home/nano/Desktop/run/0.png')
    return img

#################################################
class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

def get_camera_image(pipeline):
    temporal_filter = TemporalFilter(alpha=0.5)
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            print("未获取到帧数据！")
            continue

        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("未获取到彩色帧！")
            continue

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("无法将彩色帧转换为图像！")
            continue

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            print("未获取到深度帧！")
            continue

        # width = depth_frame.get_width()
        # height = depth_frame.get_height()
        # scale = depth_frame.get_depth_scale()

        # 转换为np数组
        # depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        # depth_data_s = depth_data.reshape((height, width))
        #
        # depth_data_s = depth_data_s.astype(np.float32) * scale
        # dpt = cv2.convertScaleAbs(depth_data_s, alpha=0.14)
        # depth_colormap = cv2.applyColorMap(dpt, cv2.COLORMAP_JET)
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        #
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        #
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)
        # # Apply temporal filtering
        depth_data = temporal_filter.process(depth_data)
        #
        # center_y = int(height / 2)
        # center_x = int(width / 2)
        # center_distance = depth_data[center_y, center_x]

        # 转成uint8, opencv能处理的格式
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        return color_image, depth_image


def get_data_more(pipeline, color_path, depth_path, saved_count):
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        return
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return
    color_image = frame_to_bgr_image(color_frame)
    if color_image is None:
        print("failed to convert frame to image")
        return
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.float16)
    # depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(os.path.join((color_path), "{}.png".format(saved_count)), color_image)
    # 加上那个每隔几毫秒保存
    while saved_count <= 4:
        cv2.imwrite(os.path.join((color_path), "{}.png".format(saved_count)), color_image)
        time.sleep(0.5)
        saved_count += 1


def get_data_sig(pipeline, color_path, depth_path, saved_count):
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        return
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return
    color_image = frame_to_bgr_image(color_frame)
    if color_image is None:
        print("failed to convert frame to image")
        return
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.float16)
    # depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(os.path.join((color_path), "{}.png".format(saved_count)), color_image)
    # 深度信息由采集到的float16直接保存为npy格式
    np.save(os.path.join((depth_path), "{}".format(saved_count)), depth_data)


def turnoff_camera(pipeline):
    pipeline.stop()
    print("摄像机关闭!")

if __name__ == "__main__":
    pipeline = turnon_camera()
    if pipeline is None:
        print("无法启动摄像头，请检查连接！")
        exit()

    try:
        while True:
            # 获取彩色图像和深度图像
            color_image, depth_image = get_camera_image(pipeline)
            if color_image is None or depth_image is None:
                print("未能获取图像数据，请检查摄像头状态！")
                continue

            # 显示彩色图像
            cv2.imshow("Color Image", color_image)

            # 将深度图像转换为彩色映射图并显示
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.14), cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", depth_image)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(1)
    finally:
        # 关闭摄像头
        turnoff_camera(pipeline)
        cv2.destroyAllWindows()
