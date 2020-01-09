import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18
import torch.optim as optim
import matplotlib.pyplot as plt
from my_utils import *

# define path
home_path = '/home/kyuchoi/AD_transfer'
data_dir = os.path.join(home_path, 'data')
code_dir = os.path.join(home_path, 'code')
model_dir = os.path.join(home_path, 'model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 2 # BEST for AD_NL: 77777, initial: 7 # BEST for AD_non_AD: 7777777
np.random.seed(seed)
torch.manual_seed(seed)

### hyperparams
ncpu = 0 # set number of workers
torch.backends.cudnn.enabled = False

use_finetune = 1
use_class_weight = 0 # without weights are better 
use_red_lr = 1

batch_size = 64 # BEST: 32
epochs = 100 # 50, 100
num_classes = 2
learning_rate = 0.003 # BEST: 0.003
valid_chk = 5
  
# load a train and valid dataset for AD_normal group
dataset = "AD_normal"
label_filename = os.path.join(data_dir,f"label_{dataset}.npy")

partition_AD_normal, labels_AD_normal = random_split_ratio(dataset, label_filename, seed)
AD_normal_trainset = AD_normal_data(data_dir, partition_AD_normal['train'], labels_AD_normal)
AD_normal_validset = AD_normal_data(data_dir, partition_AD_normal['valid'], labels_AD_normal)
AD_normal_testset = AD_normal_data(data_dir, partition_AD_normal['test'], labels_AD_normal)

trainloader = DataLoader(dataset=AD_normal_trainset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True: mutually exclusive with , sampler = RandomSampler(AD_normal_trainset)
validloader = DataLoader(dataset=AD_normal_validset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True
testloader = DataLoader(dataset=AD_normal_testset, batch_size = batch_size, shuffle=False, num_workers=ncpu) # , shuffle=True

## Using pre-trained ResNet3D model: not good performance 
model = r3d_18(pretrained=True)

### build model
total_param = 0
if use_finetune:
    for name, layer in model.named_children():
        layer_param = sum(p.numel() for p in layer.parameters())
        print('{0} layer: {1:09d} parameters'.format(name, layer_param))
        for param in layer.parameters():
            param.requires_grad = False
        total_param += layer_param
        
        if name == 'layer4':
            for param in layer.parameters():
                param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = False
    
# fine-tune the last FC layer to meet the class to 2 rather than original 400
num_ftrs = model.fc.in_features # 400
model.fc = nn.Linear(num_ftrs, num_classes) # not 2 to regress
    
# print out number of parameters in the model, which requires grad_descent
print('trainable/total parameters: {}/{}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad), total_param))
model = nn.DataParallel(model.cuda())

#%% loss function
if use_class_weight:
    # class weight 
    weights = torch.tensor([1.,2.]) # pos_weight in BCEloss:  a weight of positive examples. Must be a vector with length equal to the number of classes.
    class_weights = torch.FloatTensor(weights).cuda() # for GPU
    class_criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
    # no class weight
    class_criterion = nn.CrossEntropyLoss() # BEST
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if use_red_lr:
    red_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min') # init_lr is critical: BEST = 0.003 > 0.01 (worse)
else:
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=7, gamma=0.1)

#%% training
list_train_loss = []
list_valid_loss = []
list_train_acc = []
list_valid_acc = []
list_epoch = []

for epoch in range(epochs):
  #### Training
  model.train()
  optimizer.zero_grad()

  train_loss = 0.0
  train_total = 0
  train_correct = 0
  
  for batch_i, (inputs, labels) in enumerate(trainloader):    
    inputs = inputs.cuda()
    labels = labels.cuda()
    labels = labels.long() # for CELoss
    
    y_pred = model(inputs) 
    loss = class_criterion(y_pred, labels) 
    
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    
    _ , pred_class = torch.max(y_pred, 1) # pred_class.dtype: torch.int64 (long)
    train_total += labels.size(0) # labels.dtype: torch.float32 (float)
    train_correct += pred_class.eq(labels.view_as(pred_class)).sum().item()
    
    #### Validation
    if (batch_i+1) % valid_chk == 0:
        model.eval()

        with torch.no_grad():
            valid_loss = 0.0
            valid_total = 0
            valid_correct = 0
        
            for val_inputs, val_labels in validloader:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
                val_labels = val_labels.long() # for CELoss
                
                val_y_pred = model(val_inputs)
                val_loss = class_criterion(val_y_pred, val_labels)
                valid_loss += val_loss.item()
               
                _ , val_pred_class = torch.max(val_y_pred, 1)
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
            
            if use_red_lr:
                valid_loss_for_lr = valid_loss / len(validloader)
                red_lr_scheduler.step(valid_loss_for_lr)
            else:
                exp_lr_scheduler.step        

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

        class_outputs = model(test_images)
        _, predicted = torch.max(class_outputs.data, 1)

        predicted_np = predicted.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
#        print(predicted_np, test_labels_np)
        tn, fp, fn, tp = confusion_matrix(test_labels_np, predicted_np).ravel()
#        print("tn, fp, fn, tp:", tn, fp, fn, tp)
        test_total += test_labels.size(0)
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
model_filename = 'res_test.pt'

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
#torch.save(model.state_dict(), os.path.join(model_dir, model_filename))