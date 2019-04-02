#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:16:16 2019

@author: xiamu
"""

import torch.nn as nn

import torch






class MobileNetV2(torch.jit.ScriptModule):
    __constants__=['layer1','layer2','layer3_1','layer3_2','layer4_1',
                   'layer4_2','layer4_3','layer5_1','layer5_2','layer5_3',
                   'layer5_4','layer6_1','layer6_2','layer6_3',
                   'layer7_1','layer7_2','layer7_3','layer8','layer9'
                   ,'classifier']
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        
        
        input_channel = 32
        last_channel = 1280

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.layer1=self._conv_bn(3, input_channel, 2)#(32,112,112)
        
        
        
        self.layer2=self._block1(input_channel,16,1,expand_ratio=1)#(32)
        
        #self.layer3=nn.Sequential(self._block6(16,24,2,expand_ratio=6),
        #                         self._block6(24,24,1,expand_ratio=6))
        self.layer3_1=self._block6(16,24,2,expand_ratio=6)
        self.layer3_2=self._block6(24,24,1,expand_ratio=6)
        
        #self.layer4=nn.Sequential(self._block6(24,32,2,expand_ratio=6),
          #                        self._block6(32,32,1,expand_ratio=6),
          #                        self._block6(32,32,1,expand_ratio=6))
        self.layer4_1=self._block6(24,32,2,expand_ratio=6)
        self.layer4_2=self._block6(32,32,1,expand_ratio=6)
        self.layer4_3=self._block6(32,32,1,expand_ratio=6)
        
        
        
        
        
        self.layer5_1=self._block6(32,64,2,expand_ratio=6)
        self.layer5_2=self._block6(64,64,1,expand_ratio=6)
        self.layer5_3=self._block6(64,64,1,expand_ratio=6)
        self.layer5_4=self._block6(64,64,1,expand_ratio=6)
        
        
        
        
        
        self.layer6_1=self._block6(64,96,1,expand_ratio=6)
        self.layer6_2=self._block6(96,96,1,expand_ratio=6)
        self.layer6_3=self._block6(96,96,1,expand_ratio=6)
        
    
        
        self.layer7_1=self._block6(96,160,1,expand_ratio=6)
        self.layer7_2=self._block6(160,160,1,expand_ratio=6)
        self.layer7_3=self._block6(160,160,1,expand_ratio=6)
        
        self.layer8=self._block6(160,320,1,expand_ratio=6)
        
        self.layer9=self._conv_1x1_bn(320,self.last_channel)
        
        # building inverted residual blocks

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )
       # self._initialize_weights()
        
        
    @torch.jit.script_method 
    def forward(self, x):
        x = self.layer1(x)
    
        x = self.layer2(x)
        
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.layer5_3(x)
        x = self.layer5_4(x)
        
        x = self.layer6_1(x)
        x = self.layer6_2(x)
        x = self.layer6_3(x)
        
        
        x = self.layer7_1(x)
        x = self.layer7_2(x)
        x = self.layer7_3(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        
        x = x.mean(3).mean(2)
        
        x = self.classifier(x)
        return x
    
    def _conv_bn(self,inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def _conv_1x1_bn(self,inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def _block1(self,inp, oup, stride, expand_ratio):
        hidden_dim = round(inp * expand_ratio)
        #
        
        
        #assert stride in [1, 2]
        return nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
    def _block6(self,inp, oup, stride, expand_ratio):
        hidden_dim = round(inp * expand_ratio)
        #assert stride in [1, 2]
        return nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
    

if __name__=='__main__':
    model=MobileNetV2(n_class=28,input_size=128)
    model.cuda()
    ones=torch.ones((1,3,128,128)).cuda()
    import time
    y1=model(ones)
    time1=time.time()
    y=model(ones)
    print('......',time.time()-time1)
    
    
    print('.....',y.shape)
    torch.save(model.state_dict(),'first.pt')
    

    tranced_script_module=torch.jit.trace(model,ones)
    
    
    output=tranced_script_module(ones)
    print(output.shape)
    
    tranced_script_module.save('second.pt')
    print('successfully trace model........')