# -*- coding: utf-8 -*-
import torch
state_dict = torch.load("/home/amax/Code/SINet-master/Snapshot/air-lab2/SINet_40.pth")#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, "/home/amax/Code/SINet-master/Snapshot/air-lab/SINet_406.pth", _use_new_zipfile_serialization=False)
