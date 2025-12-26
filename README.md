# Industry Measurement 2025
这里是2025年西安交通大学Robocup先进视觉-工业测量专项赛的代码仓库，我们团队来自23级人工智能、自动化专业。在2025年中国机器人大赛中，我们取得了先进视觉**专项赛**的**冠军**和全国**总决赛**的**二等奖**，本仓库为我们方案的完整实现，为学弟学妹们提供参考。

##  基础

不管是为了比赛还是后续发展，我们都建议掌握以下技能

- Linux基础（Ubuntu）
- git
- python
- opencv-python
- DL(目标检测、关键点检测)

如果你参加过Robomaster/Robocon的视觉组培训，相信对以上技能不陌生；如果没有，我们建议你边做边学，这是效率最高的方式

建议队里至少有一两个学过以上内容的同学作为主力

学长们都很忙，要有自己查资料的习惯和能力

## 数据集
因数据集过大(30G), 我们放在网盘以供自取：https://pan.baidu.com/s/1MZDW8RhHfFipV-LxlUtjAA?pwd=bcag

## 目录
```
.
├── Docs # 文档，包括相机驱动安装步骤、
│   ├── imgs
│   ├── README.md   # 方案思路
│   └── orbbecsdk安装.pdf # 相机驱动配置教程
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
