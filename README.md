# 无处遁形——透明物体检测（opencv项目成果）
﹣Air-lab 鹰之队

## 前言
塑料， 玻璃制品等透明物体在日常生活中很常见。透明物体的快速检测在工业、
智能领域都有着广泛的应用。由于透明物体具有反射、折射和光投射的独特特性，一
方面，这些特性会混淆机器人的传感器,即便是再昂贵的传感器也会失效；另一方面，
很多基于视觉的目标检测方法也会受到这些特性的严重影响，导致最终的检测失败。

##环境
torch==1.7.1
scipy==1.2.2
torchvision==0.8.2
opencv-python==4.6.0.66
numpy==1.19.2

##数据准备
创建‘./datasets/transparent/Trans10K_v2 ‘
将训练/测试数据放在‘./datasets/transparent/Trans10K_v2‘
 Trans10K_v2
 ├────test
 │    ├───images
 │    └───masks_12
 ├────train
 │    └───images
      └───masks_12
 download dataset: 'https://github.com/xieenze/Trans2Seg'

##训练透明目标检测模型
我们实验基于2个GeForce GTX 1080 ，训练一小时左右。
运行./Air-lab透明目标检测/Mytrain,得到SINet_40.pth，模型保存于'./Snapshot/air-lab/'（SINet_406.pth）

参考代码'https://github.com/DengPingFan/SINet/'
**如果得到的模型为.zip格式，按照如下方法解决：
在pytorch1.6版本下运行（或打开'zip转为pth.py',填入路径进行转换）：
state_dict = torch.load("path")
torch.save(state_dict, "path", _use_new_zipfile_serialization=False
path为模型所在路径

##调用透明目标检测模型进行透明目标检测
将要检测的图片放于./Air-lab透明目标检测/motest/test/images
运行 ./Air-lab透明目标检测/main.py

##结果
测试结束后会得到结果在 ./Air-lab透明目标检测/Resulttest 中

##参考文献
[1]@article{xie2021segmenting, title={Segmenting transparent object in the wild with transformer},
 author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Sun, Peize and Xu, Hang and Liang, 
Ding and Luo, Ping}, journal={arXiv preprint arXiv:2101.08461}, year={2021} }
[2]@inproceedings{fan2020Camouflage,
title={Camouflaged Object Detection},
author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen,
 Jianbing and Shao, Ling},booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2020}
}

##项目结果图片展示
![72d7314089ad02e28d204da1b0ff04d](https://user-images.githubusercontent.com/120435702/208233610-d843787e-64ae-4e7f-a85e-2c50bfa2a685.jpg)
![46dfbe1699856879d3705ec390e71d2](https://user-images.githubusercontent.com/120435702/208233611-56f29db7-8d09-440c-aeda-7f6e936aaf8f.jpg)
![e3a4352b63e5e0b439f395630f58fd9](https://user-images.githubusercontent.com/120435702/208233619-cb7e3077-5801-41ff-afd7-9a3f27a64575.jpg)
![1c4df65458193201cc50973314f4b57](https://user-images.githubusercontent.com/120435702/208233623-a46a0141-b023-43b2-859d-372f05cffd5d.jpg)
![6d81499ddfe25ca9a12a398892b36ed](https://user-images.githubusercontent.com/120435702/208233625-0a337bb0-501c-49c1-82b5-859c6275d7fe.jpg)
![761352a97689384c823a41a8f9e1834](https://user-images.githubusercontent.com/120435702/208233630-360a547f-42fb-4a6e-bdcc-986101e1205d.jpg)

