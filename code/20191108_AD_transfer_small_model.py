#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('grep -c processor /proc/cpuinfo')

# set number of workers
ncpu = 24

get_ipython().system("ls '/home/kyuchoi/AD_transfer/data'")
data_dir = '/home/kyuchoi/AD_transfer/data'
code_dir = '/home/kyuchoi/AD_transfer/code'


# In[2]:


import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
# from torchvision.models.video import r3d_18
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from skimage.transform import resize
# from torchsummary import summary
# from skimage.util import crop
# from skimage.io import *
sys.path.append(code_dir)
# from my_utils import *
# import my_utils.py


# In[3]:


# create train/valid/test set using 8:1:1 ratio 
def random_split(dataset, label_filename, seed):
  np.random.seed(seed)
  label_dataset = np.load(label_filename)
  all_ID_list = range(label_dataset.shape[0])

  # pick valid and test index randomly without replacement
  valid_test_ID_list = np.random.choice(label_dataset.shape[0], (label_dataset.shape[0]//10, 2), replace=False)
  valid_ID_list = valid_test_ID_list[:,0]
  test_ID_list = valid_test_ID_list[:,1]
  train_ID_list = set(all_ID_list) - set(valid_ID_list) - set(test_ID_list) # use datatype 'set' to remove list A from list B

  # make id from 1 to 001
  train_ID_list = [("%03d" % i) for i in train_ID_list] # MUST change datatype via list
  valid_ID_list = [("%03d" % i) for i in valid_ID_list] 
  test_ID_list = [("%03d" % i) for i in test_ID_list] 

  # return partition and corresponding labels
  partition = {'train': train_ID_list, 'valid': valid_ID_list, 'test': test_ID_list}
  labels = {}
  for i in range(label_dataset.shape[0]):
    labels['%03d'% i] = label_dataset[i]
  return partition, labels

def my_transforms(x):
  # Gaussian normalize
  x = (x - np.mean(x)) / np.std(x)
  # resize
#   x = resize(x, (input_size, input_size, input_size))
#   print(x.shape)
  # random crop to downscale half
#   crop_size = 64
#   x1 = np.random.choice(range(x.shape[0]-crop_size),3)[0]
#   y1 = np.random.choice(range(x.shape[1]-crop_size),3)[1]
#   z1 = np.random.choice(range(x.shape[2]-crop_size),3)[2]
#   x2 = x1 + crop_size
#   y2 = y1 + crop_size
#   z2 = z1 + crop_size
#   ## using skimage.util.crop: working but different from expected
#   #   x = crop(x, ((x1, x2), (y1, y2), (z1, z2)), copy = False)
#   #   x = crop(x, ((2, crop_size), (crop_size, crop_size), (crop_size, crop_size)), copy = False)
#   x = x[x1:x2, y1:y2, z1:z2]
#   print(x.shape)
  # dimension change
  x = x[:,:,:,np.newaxis]
  x = np.concatenate((x, x, x), axis = -1)
  x = np.transpose(x, (3,2,0,1)) # (H,W,D,C) -> (C,D,H,W) for 3d tensor in torch
  return x


# In[4]:


# input_size = 128

test = np.load(os.path.join(data_dir, "AD_NL", "AD_NL_001.npy"))
print(test.shape)
test = my_transforms(test)


# In[5]:


print(test.shape)
plt.imshow(test[0,:,14,:])
print(np.min(test), np.max(test))
# plt.hist(test[0,:,14,:])


# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

seed = 7
np.random.seed(seed)

batch_size = 32 # 16
epochs = 200 # 50
input_size = 128

class AD_NL_data(Dataset):
  def __init__(self, data_dir, list_IDs, labels):  
    self.data_dir = data_dir
    self.labels = labels
    self.list_IDs = list_IDs

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    ID = self.list_IDs[index] # len(self.labels) will give you an error.

    # load AD_NL npy and add channel dimension and change it from 1 to 3 for resnet_3d_18
    x_data = np.load(os.path.join(data_dir, "AD_NL", "AD_NL_{}.npy".format(ID))) # (79, 95, 79)
    x_data = my_transforms(x_data)
    # print(np.min(x_data), np.max(x_data))
    # get y from labels
    y_data = np.array(self.labels[ID], dtype = np.uint8) # removing 'np.array' will get 'TypeError: expected np.ndarray (got numpy.int64)'

    # change datatype from npy to torch
    self.x_data = torch.from_numpy(x_data)
    self.y_data = torch.from_numpy(y_data)
   
    # change tensor type from double to float for resnet_3d_18
    self.x_data = self.x_data.type(torch.FloatTensor)
    self.y_data = self.y_data.type(torch.FloatTensor)
  
    return self.x_data, self.y_data
  
# load a train and valid dataset for AD_NL group
dataset = "AD_NL"
label_filename = os.path.join(data_dir,"label_AD_NL.npy")

partition_AD_NL, labels_AD_NL = random_split(dataset, label_filename, seed)
AD_NL_trainset = AD_NL_data(data_dir, partition_AD_NL['train'], labels_AD_NL)
AD_NL_validset = AD_NL_data(data_dir, partition_AD_NL['valid'], labels_AD_NL)
AD_NL_testset = AD_NL_data(data_dir, partition_AD_NL['test'], labels_AD_NL)
print(AD_NL_trainset)

trainloader = DataLoader(dataset=AD_NL_trainset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True: mutually exclusive with , sampler = RandomSampler(AD_NL_trainset)
validloader = DataLoader(dataset=AD_NL_validset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True
testloader = DataLoader(dataset=AD_NL_testset, batch_size = batch_size, shuffle=True, num_workers=ncpu) # , shuffle=True

# check the data shape given the same index
print(AD_NL_trainset.__getitem__(1)[0].size()) # size (x) -> size() (o)
print(np.load(os.path.join(data_dir, "AD_NL", "AD_NL_001.npy")).shape)


# In[7]:


### build model
use_custom_model = 1
if use_custom_model:
    ## Using custom 3D model
    class my_model(nn.Module):
        def __init__(self):
            super(my_model, self).__init__()

            # kernel size
            k = 5
            # number of conv filters
            nf = 8
            dropout = 0.3
            num_classes = 2

#             act = nn.Tanh()
            act = nn.PReLU()

            self.CNN = nn.Sequential(
                nn.Conv3d(3, nf, k, padding = 1),
                nn.BatchNorm3d(nf),
                act,
                nn.MaxPool3d(2),
                nn.Conv3d(nf, nf * 2, k, padding = 1),
                nn.BatchNorm3d(nf * 2),
                act,
                nn.MaxPool3d(2),
                nn.Conv3d(nf * 2, nf * 4, k, padding = 1),
                nn.BatchNorm3d(nf * 4),
                act,
                nn.MaxPool3d(2),
                nn.Conv3d(nf * 4, nf * 8, k, padding = 1),
                nn.BatchNorm3d(nf * 8),
                act,
                nn.MaxPool3d(2)
            )

#             self.fc1 = nn.Linear(nf*8*2*2*2,64*2) # refer to output size of torch tensor from below: print(x1.size()) e.g. 64*6*6*6 (o) but, 128, 256 (x) -> it's nothing but (# of filters in the last conv layer) * (last conv size * last conv size * last conv size)
            self.fc1 = nn.Linear(nf*8*3*3*4,64*2) 
            self.fc2 = nn.Linear(64*2,64)
            self.fc3 = nn.Linear(64,num_classes)
            self.dropout = nn.Dropout(dropout)
            self.prelu = nn.PReLU()
            self.lrelu = nn.LeakyReLU()

        def forward(self, x):
            x = self.CNN(x)
#             print(x.size()) # to check the output size of torch tensor: if (8,128,4,4,5) -> 8 is batch_size so read the others and multiply: 128*4*4*5 for nn.Linear
            x = x.view(x.size(0), -1)
            x = self.lrelu(self.fc1(x))
            x = self.dropout(x)
            x = self.lrelu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    model = my_model()
    # print(model)
else:
    ## Using pre-trained ResNet3D model: not good performance 
    # adjust the last FC layer to meet the class to 2 rather than original 400
    num_classes = 2
    model = r3d_18(pretrained=True)
    # print(model) # same as model.summary() in Keras
    num_ftrs = model.fc.in_features # 400
    model.fc = nn.Linear(num_ftrs, num_classes) # not 2 to regress
    # print(model)
use_gpu = 1
if use_gpu:
    model = model.cuda()

model = nn.DataParallel(model, device_ids=None)
# summary(model, input_size = (3,64,64,64)) # same as model.summary() in Keras
# print(model)
# print out number of parameters in the model, which requires grad_descent
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# criterion = nn.BCELoss()
use_class_weight = 0
if use_class_weight:
    # class weight 
    weights = torch.tensor([1.,2.])
    class_weights = torch.FloatTensor(weights).cuda() # for GPU
    # print(class_weights.size()) # torch.Size([2])
    criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# In[8]:


list_train_loss = []
list_valid_loss = []
list_train_acc = []
list_valid_acc = []
list_epoch = []

valid_chk = 5

for epoch in range(epochs):
  #### Training
  model.train()
  optimizer.zero_grad()

  train_loss = 0.0
  train_total = 0
  train_correct = 0

  # for inputs, labels in trainloader:
  
  for batch_i, (inputs, labels) in enumerate(trainloader):
    # ts = time.time()
    inputs, labels = inputs.to(device), labels.to(device)
    labels = labels.long() # convert y_true as integer using .long() to use CrossEntropyLoss() as loss function
    y_pred = model(inputs) # y_pred.size() = torch.size([16,2]), labels.size() = torch.size([16])
    # print(y_pred.size(), y_pred)
    # loss = criterion(torch.sigmoid(y_pred), labels) # not working for CEL
    loss = criterion(y_pred, labels)
    # print(loss)
    
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    # _ , pred_class = torch.max(torch.sigmoid(y_pred), 1)
    _ , pred_class = torch.max(y_pred, 1) # torch.max returns [values, indices], so you need to indexing by using [1], or _ , pred_class would do the same.
    # print(pred_class,labels)
    train_total += labels.size(0)
    # print(pred_class.size(), labels.size()) # pred_class.size(), labels.size() = 16, 16
    train_correct += (pred_class == labels).sum().item()
    # te = time.time()
    # print("batch_calculating time:%2.2f" % (te - ts))
    
    #### Validation
    if (batch_i+1) % valid_chk == 0:
        model.eval()

        with torch.no_grad():
            valid_loss = 0.0
            valid_total = 0
            valid_correct = 0

            for val_inputs, val_labels in validloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_labels = val_labels.long()
                val_y_pred = model(val_inputs)

                val_loss = criterion(val_y_pred, val_labels)
                valid_loss += val_loss.item()
                # _ , val_pred_class = torch.max(torch.sigmoid(val_y_pred), 1)
                _ , val_pred_class = torch.max(val_y_pred, 1)
                # print(val_pred_class, val_labels)
                valid_total += val_labels.size(0)
                # print(val_labels.size(0)) # 16
                # valid_correct += (val_pred_class == val_labels).double().sum().item()
                valid_correct += (val_pred_class == val_labels).sum().item()
            # print(valid_correct, valid_total)
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


# In[9]:


fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
list_epoch = list(np.arange(0,epochs*4))
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


# In[ ]:


# save the optimized model
model_dir = '/home/kyuchoi/AD_transfer/model'
model_filename = 'test.pt'

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
# torch.save(model.state_dict(), os.path.join(model_dir, model_filename))

