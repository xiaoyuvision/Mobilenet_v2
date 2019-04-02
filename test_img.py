# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:47:01 2019

@author: leaper
"""

import numpy as np
import torch
from torch.autograd import Variable as V

from model import MobileNetV2

import os

import config
import cv2
import time

NAME=config.weights_name

SHAPE=(128,128)
weights_path=config.weights_path
solver=MobileNetV2(n_class=len(config.label_names),input_size=SHAPE[0])
solver.cuda()

if os.path.exists(os.path.join(weights_path,NAME)):
    solver.load_state_dict(torch.load(os.path.join(weights_path,NAME)),strict=False)
    print('....load weights successfully')    



def test_one_img_from_img(model, img,shape=SHAPE):
    img=cv2.resize(img,shape).transpose(2,0,1)
    
    
    img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
    img = V(torch.Tensor(img).cuda()).unsqueeze(0)
    out = model(img)
    out=out.squeeze().cpu().data.numpy()
    return np.argmax(out)




def get_results(predict_img_path,save_path):
    for i in os.listdir(predict_img_path):
        img=cv2.imread(os.path.join(predict_img_path,i))
        time1=time.time()
        out=test_one_img_from_img(solver,img)
        print('......',time.time()-time1)
        out_label=config.label_names[out]
        if not os.path.exists(os.path.join(save_path,out_label)):
            os.makedirs(os.path.join(save_path,out_label))
        cv2.imwrite(os.path.join(save_path,out_label,i),img)
if __name__=='__main__':
    get_results(r'D:\linear-classification\imgs_cut_1\yes',r'D:\linear-classification\imgs_pred')
    