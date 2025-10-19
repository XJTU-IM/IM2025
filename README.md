# Industry Measurement 2025
这里是2025年西安交通大学Robocup先进视觉-工业测量专项赛的代码仓库，我们团队来自23级人工智能、自动化专业。在2025年中国机器人大赛中，我们取得了先进视觉专项赛的冠军和全国总决赛的二等奖，本仓库为我们方案的完整实现，为学弟学妹们提供参考。

## 数据集
因数据集过大(30G), 我们放在网盘以供自取：https://pan.baidu.com/s/1MZDW8RhHfFipV-LxlUtjAA?pwd=bcag

## 使用
```
.
├── Docs # 文档，包括相机驱动安装步骤、
│   ├── imgs
│   ├── index.html
│   ├── README.md   # 方案思路
│   └── _sidebar.md
├── industry_measurement    # 主目录
│   ├── calculate.py
│   ├── cfg # 配置
│   ├── common.py
│   ├── example
│   ├── extract_desktop.py
│   ├── for_ellipse.py
│   ├── For_shim.py
│   ├── Log
│   ├── main.py # 主代码
│   ├── results
│   ├── results_process.py
│   ├── saves
│   ├── test
│   ├── ui_mainwindow.py
│   ├── utils_camera.py
│   ├── utils.py
│   ├── visualize.py
│   ├── yolo_poser.py
│   ├── yolo_segor.py
│   └── yolov5_seg
├── judgeGui-2025   # 裁判盒链接
│   └── README.md
├── README.md
└── tools   # 数据集处理工具、相机取流
    ├── clip.py
    ├── images_process.py
    ├── images_rename.py
    ├── images_save_space.py
    ├── label2mscoco.py
    ├── Log
    ├── process_dataset
    ├── __pycache__
    ├── readme.md
    ├── realsense.py
    ├── save_img_with_camera.py
    ├── split_dataset_mscoco.py
    ├── split_dataset.py
    ├── StreamForPlus.py
    └── utils.py

15 directories, 30 files
