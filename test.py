import filecmp
import os
from models.yolo import Model
import torch

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)
#
# txt1=open('a.txt',mode='w')
# txt2=open('a1.txt',mode='w')
#
# ck=torch.load('pretrained/yolov5x.pt')
# txt1.write(str(ck))
#
# ck1=torch.load('myweights/R_yolov5x.pt')
# ck1['epoch']-=1
# txt2.write(str(ck1))
# print(filecmp.cmp('a.txt','a1.txt'))
