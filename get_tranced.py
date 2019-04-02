#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:08:06 2019

@author: xiamu
"""

import cv2
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V


from module import MobileNetV2

import config
















model=MobileNetV2(n_class=28,input_size=128)

model.load_state_dict(torch.load('first.pt'),strict=False)
#model.cuda()
model.eval()

ones=torch.ones((1,3,128,128))

tranced_script_module=torch.jit.trace(model,ones)
print('load weights successfully........')

output=tranced_script_module(ones)
print(output.shape)

tranced_script_module.save('second.pt')