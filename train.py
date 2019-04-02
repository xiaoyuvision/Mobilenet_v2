# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:59:32 2019

@author: leaper
"""

import torch

import torchvision
import os


from data import ImageFolder
from model import MobileNetV2

import config

labels_dictionary={}

for i in config.label_names:
    labels_dictionary[i]=len(labels_dictionary)
    

SHAPE=(128,128)
NAME=config.weights_name


weights_path=config.weights_path
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    

batchsize=4
img_root=config.traindata_path




solver=MobileNetV2(n_class=len(config.label_names),input_size=SHAPE[0])


solver.cuda()

lr=2e-4
optimizer=torch.optim.Adam(params=solver.parameters(), lr=lr)
criterion=torch.nn.CrossEntropyLoss().cuda()

dataset = ImageFolder(img_root,labels_dictionary)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)


def update_lr(old_lr,new_lr, mylog, factor=False):
    if factor:
        new_lr = old_lr / new_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    #print(>> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
    print('update learning rate: %f -> %f' % (old_lr, new_lr))
    old_lr = new_lr
    return old_lr

if not os.path.exists('logs'):
    os.makedirs('logs')
    
if os.path.exists(os.path.join(weights_path,NAME)):
    solver.load_state_dict(torch.load(os.path.join(weights_path,NAME)),strict=False)
    print('....load weights successfully')    

mylog = open('logs/'+NAME+'.log','w')



no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100
old_lr=lr
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, label in data_loader_iter:
        img=torch.autograd.Variable(img.cuda())
        label=torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        score=solver(img)
        train_loss=criterion(score,label)
        
        train_epoch_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
    train_epoch_loss /= len(data_loader_iter)
    print ('epoch:',epoch)
    print ('train_loss:',train_epoch_loss)
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        torch.save(solver.state_dict(),os.path.join(weights_path,NAME))

    if no_optim > 10:
        print('early stop at %d epoch' % epoch)
        break

    if no_optim > 6:
        if old_lr < 5e-7:
            break

        solver.load_state_dict(torch.load(os.path.join(weights_path,NAME)),strict=False)
        old_lr=update_lr(old_lr=old_lr,new_lr=5.0, factor = True, mylog = mylog)
    mylog.flush()


#My_module.forward(ones)
#solver.net.save('finish.pt')

print('Finish!')
mylog.close()