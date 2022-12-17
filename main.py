import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import cv2
parser = argparse.ArgumentParser()
#测试集输入的图像大小
parser.add_argument('--testsize', type=int,
                    default=352, help='the snapshot input size')
#调用的模型路径
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/air-lab/SINet_406.pth')
#输出图片的保存路径
parser.add_argument('--test_save', type=str,
                    default='./motest/Results1')
opt = parser.parse_args()
model = SINet_ResNet50().cuda()
#保存模型
model.load_state_dict(torch.load(opt.model_path))
model.eval()
#for dataset in ['COD10K']:
for dataset in ['test']:#Transparent
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    # test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
    #                            gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
    #                            testsize=opt.testsize)
    test_loader = test_dataset(image_root='./motest/{}/images/'.format(dataset),
                               gt_root='./motest/{}/images/'.format(dataset),
                               testsize=opt.testsize)
    # test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
    #                            gt_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
    #                            testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1
#Notice: The above codes are  from Camouflaged Object Detection (CVPR2020-Oral)
#
# > Authors:
# > [Deng-Ping Fan](https://dengpingfan.github.io/),
# > [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en),
# > [Guolei Sun](https://github.com/GuoleiSun),
# > [Ming-Ming Cheng](https://mmcheng.net/),
# > [Jianbing Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing),
# > [Ling Shao](http://www.inceptioniai.org/).

#使用opencv优化图片，得到输出结果
#定义卷积层
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#读取待处理图片
img1 = cv2.imread("./motest/test/images/1038.jpg")
#定义窗口，设置好合适的窗口大小
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 1280, 640)
#读取通过模型处理后得到的目标图片
img = cv2.imread('./motest/Results1test/1038.png',cv2.IMREAD_GRAYSCALE)
#去噪
blur = cv2.GaussianBlur(img,(3,3),5)
#腐蚀
erode = cv2.erode(blur,kernel)
# 膨胀：把图像还原
dilate = cv2.dilate(erode, kernel, iterations=2)
# 消除内部的小块  闭运算
img = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
#将图片进行二值化处理
thresh, dst = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
#查找轮廓
contours,hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#拷贝图片，让原图片不受接下来的操作的影响
img_copy = img.copy()
#在拷贝图片上画出轮廓，检查其是否符合要求
img_copy = cv2.drawContours(img_copy,contours,-1,(0,0,255),5)
#画出所有检测出来的轮廓
for contour in contours:
    # 最大外接矩形
    (x,y,w,h) = cv2.boundingRect(contour)
    # 通过外接矩形的宽高大小来过滤掉小的矩形
    is_valid = (w>=70)&(h>=70)
    if not is_valid:
        continue
    cv2.rectangle(img1,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,255),5)
    # 添加文本，在目标透明物上写上标签transparent
    cv2.putText(img1, 'transparent',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
cv2.imshow('cam',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
