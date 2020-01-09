#!/usr/bin/env python
# coding: utf-8

# linux script: not working without ipython_get()
#!grep -c processor /proc/cpuinfo # 8 in cuda01, 16 in pcuda01, 24 in pcuda02
#!ls '/home/kyuchoi/AD_transfer'

# import library
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
#from torchvision.models.video import r3d_18
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchsummary import summary
from my_utils import *

# define path
home_path = '/home/kyuchoi/AD_transfer'
data_dir = os.path.join(home_path, 'data')
code_dir = os.path.join(home_path, 'code')
model_dir = os.path.join(home_path, 'model')

# set environment path: working without setting the path
#sys.path.append(code_dir)

### test transformation of images
test = np.load(os.path.join(data_dir, "AD_NL", "AD_NL_001.npy"))
# print(test.shape)
test = my_transforms(test)
#print(test.shape)
plt.imshow(test[0,:,14,:])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 7
np.random.seed(seed)

### hyperparams
# set number of workers
ncpu = 24
use_gpu = 1
use_custom_model = 1
use_class_weight = 0 # without weights are better 

batch_size = 32 # 16
epochs = 100 # 50
num_channel = test.shape[0] # 1
#print(num_channel)
num_classes = 2
learning_rate = 0.003

valid_chk = 5
lambda_class = 0.5 # weight for recon loss vs class loss
noise_variance = 0.2 # making target images for autoencoder
  
# load a train and valid dataset for AD_NL group
dataset = "AD_NL"
label_filename = os.path.join(data_dir,"label_AD_NL.npy")

partition_AD_NL, labels_AD_NL = random_split(dataset, label_filename, seed)
AD_NL_trainset = AD_NL_data(data_dir, partition_AD_NL['train'], labels_AD_NL)
AD_NL_validset = AD_NL_data(data_dir, partition_AD_NL['valid'], labels_AD_NL)
AD_NL_testset = AD_NL_data(data_dir, partition_AD_NL['test'], labels_AD_NL)
# print(AD_NL_trainset)

trainloader = DataLoader(dataset=AD_NL_trainset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True: mutually exclusive with , sampler = RandomSampler(AD_NL_trainset)
validloader = DataLoader(dataset=AD_NL_validset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True
testloader = DataLoader(dataset=AD_NL_testset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True

# check the data shape given the same index
#print(AD_NL_trainset.__getitem__(1)[0].size()) # size (x) -> size() (o)


### build model
if use_custom_model:
    ## Using custom 3D model
    model = my_model()
    # print(model)
else:
    ## Using pre-trained ResNet3D model: not good performance 
    model = r3d_18(pretrained=True)
    # fine-tune the last FC layer to meet the class to 2 rather than original 400
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features # 400
    model.fc = nn.Linear(num_ftrs, num_classes) # not 2 to regress

if use_gpu:
    model = model.cuda()

model = nn.DataParallel(model, device_ids=None)
print(model)
# print out number of parameters in the model, which requires grad_descent
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

### for torchsummary, you should do model.eval() for batchnorm layer
#model.eval() # if you have batchnorm layer in the model, then the summary function throws an error because it can't calculate batch estimates such as mean and variance with a single instance.
#summary(model, input_size = (num_channel,64,64,64)) # torch summary supports for multiple inputs (e.g. summary(model, [(1, 16, 16), (1, 28, 28)])), but not multiple outputs: https://github.com/sksq96/pytorch-summary
# Therefore, for multiple outputs, you should make the table using the main output among the outputs.

recon_criterion = nn.MSELoss()

if use_class_weight:
    # class weight 
    weights = torch.tensor([1.,2.])
    class_weights = torch.FloatTensor(weights).cuda() # for GPU
    # print(class_weights.size()) # torch.Size([2])
    class_criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
    class_criterion = nn.CrossEntropyLoss()
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=7, gamma=0.1)
# print out lr decay: x 0.1 for every 7 steps
# for epoch in range(1, 25):
#     exp_lr_scheduler.step()
#     print('Epoch {}, lr {}'.format(
#         epoch, optimizer.param_groups[0]['lr']))

list_train_loss = []
list_valid_loss = []
list_train_acc = []
list_valid_acc = []
list_epoch = []

for epoch in range(epochs):
  #### Training
  exp_lr_scheduler.step()
  model.train()
  optimizer.zero_grad()

  train_loss = 0.0
  train_total = 0
  train_correct = 0

  # for inputs, labels in trainloader:
  
  for batch_i, (inputs, labels) in enumerate(trainloader):
    target_img = add_noise(inputs, noise_variance)
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = labels.long()
    target_img = target_img.to(device)
    
    z_pred, y_pred = model(inputs) 
    recon_loss = recon_criterion(y_pred, target_img) * 100
    class_loss = class_criterion(z_pred, labels) * 10
#     print(recon_loss.item(), class_loss.item())
    loss = recon_loss * (1 - lambda_class) + class_loss * lambda_class
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    
    _ , pred_class = torch.max(z_pred, 1)
    train_total += labels.size(0)
    train_correct += (pred_class == labels).sum().item()
    
    #### Validation
    if (batch_i+1) % valid_chk == 0:
        model.eval()

        with torch.no_grad():
            valid_loss = 0.0
            valid_total = 0
            valid_correct = 0
        
            for val_inputs, val_labels in validloader:
                val_target_img = add_noise(val_inputs, noise_variance)
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_labels = val_labels.long()
                val_target_img = val_target_img.to(device)
                
                val_z_pred, val_y_pred = model(val_inputs)
                val_recon_loss = recon_criterion(val_y_pred, val_target_img) * 100
                val_class_loss = class_criterion(val_z_pred, val_labels) * 10
#                 print(val_recon_loss.item(), val_class_loss.item())
                val_loss = val_recon_loss * (1 - lambda_class) + val_class_loss * lambda_class
                valid_loss += val_loss.item()
               
                _ , val_pred_class = torch.max(val_z_pred, 1)
                valid_total += val_labels.size(0)
                valid_correct += (val_pred_class == val_labels).sum().item()
            
            print("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | valid loss: {:.4f}| train acc: {:.4f} | valid acc: {:.4f}".format(
                epoch+1, epochs, batch_i+1, len(trainloader), train_loss / len(trainloader), valid_loss / len(validloader), train_correct * 100 / train_total, valid_correct * 100 / valid_total
                ))
            list_train_loss.append(train_loss / len(trainloader))
            list_valid_loss.append(valid_loss / len(validloader))
            list_train_acc.append(train_correct * 100 / train_total)
            list_valid_acc.append(valid_correct * 100 / valid_total)
            
            train_loss = 0.0
            train_total = 0
            train_correct = 0


# plot loss and acc

fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
list_epoch = list(np.arange(0,epochs*2))
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_valid_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_epoch, list_train_acc, marker='x', label='train_acc')
ax2.plot(list_epoch, list_valid_acc, marker='x', label='val_loss')
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs Accuracy')

plt.show()

# model evaluation with test set
model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for data in testloader:
        test_images, test_labels = data
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()
        test_labels = test_labels.long()

        class_outputs, recon_images = model(test_images)
        _, predicted = torch.max(class_outputs.data, 1)
        # print(class_outputs.data)
        test_total += test_labels.size(0)
        test_correct += (predicted == test_labels).sum().item()
        # print(predicted, labels)
    test_acc = 100 * test_correct / test_total
print(test_acc)

#%% save the optimized model
model_filename = 'ae_test_6590.ckpt'

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
#opt_model.load_state_dict(torch.load(os.path.join(model_dir, model_filename)))
#print(opt_model)
