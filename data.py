# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:43:00 2019

@author: leaper
"""

import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
import config
import glob








def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        

    return image

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        

    return image

def randomVerticleFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)

    return image

def randomRotate90(image,  u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        

    return image

def default_loader(img_id,resize_shape=(64,64)):

    img = cv2.imread(img_id)
    if img.shape[1]<int(resize_shape[1]*0.75) and img.shape[0]<int(resize_shape[0]*0.75):
        padding1=(int(resize_shape[0]*0.75)-img.shape[0])//2
        padding2=(int(resize_shape[1]*0.75)-img.shape[1])//2
        img = cv2.copyMakeBorder(img,padding1,padding1,padding2,padding2,cv2.BORDER_CONSTANT,value=[0,0,0])
        
    img=cv2.resize(img,resize_shape)
    
    
    
    img = randomShiftScaleRotate(img,
                                 shift_limit=(-0.1, 0.1),
                                 scale_limit=(-0.1, 0.1),
                                 aspect_limit=(-0.1, 0.1),
                                 rotate_limit=(-0, 0))
    img = randomHorizontalFlip(img )
    img= randomVerticleFlip(img)
    img = randomRotate90(img)
    
    
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
   
    
    return img

class ImageFolder(data.Dataset):

    def __init__(self,img_root,labels_dictionary):
        self.img_root=img_root
        self.labels_dictionary=labels_dictionary
        
        
        self.imgs_ids=glob.glob(os.path.join(img_root,'*','*.png'))
        
        np.random.shuffle(self.imgs_ids)
        self.loader=default_loader
        
        
        
        
        

    def __getitem__(self, index):
        idx = self.imgs_ids[index]
        #print('....idx',idx)
        img = self.loader(idx)
        img = torch.Tensor(img)
        label=self.labels_dictionary[idx.split('\\')[-2]]
       # print('.....label....',label)
        #label=torch.Tensor(label)
        
        return img,label

    def __len__(self):
        return len(self.imgs_ids)