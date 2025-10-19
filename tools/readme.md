| 代码                   | 功能                                                             |
|----------------------|----------------------------------------------------------------|
| process_dataset /    | rtmpose要求每个关键点要有一个唯一的group id，文件夹下分别是给我们三个关键点检测模型添加group id的代码 |
| label2mscoco         | 将标注文件转换为MSCOCO格式，注意要修改class_list对应不同的任务                        |
| clip                 | 通过角点拉伸桌面                                                       |
| images_save_space    | 拍数据集用的，可根据需求设置取流像素                                             |
| images_rename        | 数据集重命名                                                         |
| split_dataset_mscoco | 划分数据集为mscoco格式                                                 |
