#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from my_utils import *

# define path
home_path = '/home/kyuchoi/AD_transfer'
data_dir = os.path.join(home_path, 'data')
code_dir = os.path.join(home_path, 'code')
model_dir = os.path.join(home_path, 'model')

### test transformation of images
test = np.load(os.path.join(data_dir, "AD_NL", "AD_NL_001.npy"))
test = my_transforms(test)
#plt.imshow(test[0,:,14,:])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 777 # BEST for AD_NL: 77777, initial: 7 # BEST for AD_non_AD: 7777777
np.random.seed(seed)
torch.manual_seed(seed)

### hyperparams
ncpu = 0 # set number of workers
torch.backends.cudnn.enabled = False
use_gpu = 1
use_custom_model = 1
use_CEL = 1 # without weights are better 

batch_size = 32 # BEST: 32
epochs = 50 # 50, 100
num_channel = test.shape[0] # 1
num_classes = 2
learning_rate = 0.003 # BEST: 0.003

valid_chk = 5
# weight for recon loss vs class loss
lambda_class = 0.7 # BEST: 0.8 with seed 77777, initial: 0.5
noise_variance = 0.2 # making target images for autoencoder
  
# load a train and valid dataset for AD_NL group
dataset = "AD_NL"
label_filename = os.path.join(data_dir,f"label_{dataset}.npy")

partition_AD_NL, labels_AD_NL = random_split(dataset, label_filename, seed)
AD_NL_trainset = AD_NL_data(data_dir, partition_AD_NL['train'], labels_AD_NL)
AD_NL_validset = AD_NL_data(data_dir, partition_AD_NL['valid'], labels_AD_NL)
AD_NL_testset = AD_NL_data(data_dir, partition_AD_NL['test'], labels_AD_NL)

trainloader = DataLoader(dataset=AD_NL_trainset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True: mutually exclusive with , sampler = RandomSampler(AD_NL_trainset)
validloader = DataLoader(dataset=AD_NL_validset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True
testloader = DataLoader(dataset=AD_NL_testset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True

### build model
if use_custom_model:
    ## Using custom 3D model
    model = my_model()
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

recon_criterion = nn.MSELoss()

if use_CEL:
    # no class weight
#    class_criterion = nn.CrossEntropyLoss() # BEST
    # class weight 
    weights = torch.tensor([1.,2.]) # pos_weight in BCEloss:  a weight of positive examples. Must be a vector with length equal to the number of classes.
    class_weights = torch.FloatTensor(weights).cuda() # for GPU
    class_criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
#    class_criterion = nn.BCEWithLogitsLoss() 
#    pos_weight = torch.tensor([(len(label_AD_non_AD)-sum(label_AD_non_AD))/sum(label_AD_non_AD)])
#    pos_weight_cuda = torch.FloatTensor(pos_weight).to(device)
    class_criterion = nn.BCEWithLogitsLoss(weight = torch.tensor(2))#pos_weight_cuda)
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=7, gamma=0.1)

red_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min') # init_lr is critical: BEST = 0.003 > 0.01 (worse)

list_train_loss = []
list_valid_loss = []
list_train_acc = []
list_valid_acc = []
list_epoch = []

for epoch in range(epochs):
  #### Training
#  exp_lr_scheduler.step()
  model.train()
  optimizer.zero_grad()

  train_loss = 0.0
  train_total = 0
  train_correct = 0
  
  for batch_i, (inputs, labels) in enumerate(trainloader):
    target_img = add_noise(inputs, noise_variance)
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels = labels.long() # for CELoss
    target_img = target_img.to(device)
    
    z_pred, y_pred = model(inputs) 
    recon_loss = recon_criterion(y_pred, target_img) * 100
#    print("recon_loss:", recon_loss)
    class_loss = class_criterion(z_pred, labels) * 10 # MUST get both z_pred and labels as tensor.FloatTensor to calculate class_criterion
    # which is the reason that we converted the self.x_data and self.y_data as tensor.FloatTensor in the Dataset function.
#    print("class_loss:", class_loss)
    
    loss = recon_loss * (1 - lambda_class) + class_loss * lambda_class
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    
    _ , pred_class = torch.max(z_pred, 1) # pred_class.dtype: torch.int64 (long)
#    print("z_pred:", z_pred)
    # to calculate accuracy, convert labels from float tensor into long tensor (int64) again
#    labels = labels # labels has 1-dim for output: [[0],[1],[0]]
    train_total += labels.size(0) # labels.dtype: torch.float32 (float)
#    print("pred_class:", pred_class.dtype, "labels:", labels.dtype)
#    print("pred_class:", pred_class, "labels:", labels)
#    print(pred_class == labels) # return 0, and 1, if pred_class is same as labels (False, and True)
    # e.g.) pred_class: tensor([1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
#        1, 1, 0, 0, 0, 0, 0, 0], device='cuda:0') labels: tensor([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#        1, 1, 1, 0, 0, 1, 0, 0], device='cuda:0')
#    print((pred_class == labels).sum())
#    train_correct += (pred_class == labels.squeeze().long()).sum().item()
#    train_correct += (pred_class == labels).sum().item()
    train_correct += pred_class.eq(labels.view_as(pred_class)).sum().item()
    
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
                val_labels = val_labels.long() # for CELoss
                val_target_img = val_target_img.to(device)
                
                val_z_pred, val_y_pred = model(val_inputs)
                val_recon_loss = recon_criterion(val_y_pred, val_target_img) * 100
                val_class_loss = class_criterion(val_z_pred, val_labels) * 10

                val_loss = val_recon_loss * (1 - lambda_class) + val_class_loss * lambda_class
                valid_loss += val_loss.item()
               
                _ , val_pred_class = torch.max(val_z_pred, 1)
#                val_labels = val_labels
                valid_total += val_labels.size(0)
#                print(val_labels.size(0))
#                print((val_pred_class == val_labels.squeeze().long()).sum())
#                valid_correct += (val_pred_class == val_labels.squeeze().long()).sum().item()
                valid_correct += (val_pred_class == val_labels).sum().item()
#            print(valid_total, valid_correct)
            
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
            
            valid_loss_for_lr = valid_loss / len(validloader)
            red_lr_scheduler.step(valid_loss_for_lr)

#%% plot loss and acc

fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
list_epoch = list(np.arange(0,len(list_train_loss)))
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
ax2.plot(list_epoch, list_valid_acc, marker='x', label='val_acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs Accuracy')

plt.show()
#%%

from sklearn.metrics import confusion_matrix
# model evaluation with test set
model.eval()

tns, fps, fns, tps = 0,0,0,0
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
#        print(predicted, test_labels)
        predicted_np = predicted.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
#        print(predicted_np, test_labels_np)
        tn, fp, fn, tp = confusion_matrix(test_labels_np, predicted_np).ravel()
#        print("tn, fp, fn, tp:", tn, fp, fn, tp)
        test_total += test_labels.size(0)
#        test_labels = test_labels.squeeze().long()
        test_correct += (predicted == test_labels).sum().item()
        
        tns += tn
        fps += fp
        fns += fn
        tps += tp
    sens = tps / (tps+fns)
    spec = tns / (tns+fps)
    acc = (tps+tns) / (tps+tns+fps+fns)
    print("tp,fn,fp,tn:",tps, fns, fps, tns)
    print("sens:",sens, "spec:", spec, "acc:",acc)    
    test_acc = 100 * test_correct / test_total
print(test_acc)
#%%
# save the optimized model
model_filename = 'ae_test_7727.pt'

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
#torch.save(model.state_dict(), os.path.join(model_dir, model_filename))