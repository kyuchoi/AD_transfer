# -*- coding: utf-8 -*-
"""my_utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sa5RL-X6v2QJtiJZqbMp9ZT5haJVnk2V
"""

import numpy as np
import os
#from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
    x_data = np.load(os.path.join(self.data_dir, "AD_NL", "AD_NL_{}.npy".format(ID))) # (79, 95, 79)
    x_data = my_transforms(x_data)
    # get y from labels
    y_data = np.array(self.labels[ID], dtype = np.uint8) # removing 'np.array' will get 'TypeError: expected np.ndarray (got numpy.int64)'

    # change datatype from npy to torch
    self.x_data = torch.from_numpy(x_data)
    self.y_data = torch.from_numpy(y_data)
   
    # change tensor type from double to float for resnet_3d_18
    self.x_data = self.x_data.type(torch.FloatTensor)
    self.y_data = self.y_data.type(torch.FloatTensor)
  
    return self.x_data, self.y_data

class AD_non_AD_data(Dataset):
  def __init__(self, data_dir, list_IDs, labels):  
    self.data_dir = data_dir
    self.labels = labels
    self.list_IDs = list_IDs

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    ID = self.list_IDs[index] # len(self.labels) will give you an error.

    # load AD_non_AD npy and add channel dimension and change it from 1 to 3 for resnet_3d_18
    x_data = np.load(os.path.join(self.data_dir, "AD_non_AD", "AD_non_AD_{}.npy".format(ID))) # (79, 95, 79)
    x_data = my_transforms(x_data, use_resnet = False) # use_resnet enables to convert a grayscale image into a 3-channeld image
    # get y from labels
    y_data = np.array(self.labels[ID], dtype = np.uint8) # removing 'np.array' will get 'TypeError: expected np.ndarray (got numpy.int64)'
    
    # change datatype from npy to torch
    self.x_data = torch.from_numpy(x_data)
    self.y_data = torch.from_numpy(y_data)#.unsqueeze(0) # unsqueeze for BCELoss
    
    # change tensor type from double to float for resnet_3d_18
    self.x_data = self.x_data.type(torch.FloatTensor)
    self.y_data = self.y_data.type(torch.FloatTensor)
  
    return self.x_data, self.y_data

class AD_normal_data(Dataset):
  def __init__(self, data_dir, list_IDs, labels):  
    self.data_dir = data_dir
    self.labels = labels
    self.list_IDs = list_IDs

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    ID = self.list_IDs[index] # len(self.labels) will give you an error.

    # load AD_normal npy and add channel dimension and change it from 1 to 3 for resnet_3d_18
    x_data = np.load(os.path.join(self.data_dir, "AD_normal", "AD_normal_{}.npy".format(ID))) # (79, 95, 79)
    x_data = my_transforms(x_data, use_resnet = True) # use_resnet enables to convert a grayscale image into a 3-channeld image
    # get y from labels
    y_data = np.array(self.labels[ID], dtype = np.uint8) # removing 'np.array' will get 'TypeError: expected np.ndarray (got numpy.int64)'
    
    # change datatype from npy to torch
    self.x_data = torch.from_numpy(x_data)
    self.y_data = torch.from_numpy(y_data)#.unsqueeze(0) # unsqueeze for BCELoss
    
    # change tensor type from double to float for resnet_3d_18
    self.x_data = self.x_data.type(torch.FloatTensor)
    self.y_data = self.y_data.type(torch.FloatTensor)
  
    return self.x_data, self.y_data

class my_model(nn.Module):
        def __init__(self):
            super(my_model, self).__init__()

            self.kernel_size = 3 # kernel size
            self.nf = 2 # number of filters: BEST = 2
            self.leaky_rate = 0.2
            self.num_channel = 1 # you can also fix num_channel as 1, not variable (i.e. self.num_channel = num_channel) in class __init__
            self.num_classes = 2 # for BCEWithlogitLoss, you should specify the output shape as 1, but 2 for CELoss/ fixed num_classes in class __init__
            
            def encoder_block(in_filters, out_filters, normalization = 'batchnorm', activation = 'prelu'): # BEST: batchnorm + lrelu
                
                normalizations = nn.ModuleDict([
                        ['batchnorm', nn.BatchNorm3d(out_filters)],
                        ['instancenorm',nn.InstanceNorm3d(out_filters)],
                        ['groupnorm', nn.GroupNorm(num_groups=self.nf, num_channels=out_filters)]
                        ])
                        
                activations = nn.ModuleDict([
                        ['relu', nn.ReLU()],
                        ['prelu', nn.PReLU()],
                        ['lrelu', nn.LeakyReLU(self.leaky_rate)]
                        ])
    
                block = [nn.Conv3d(in_filters, out_filters, kernel_size = self.kernel_size, stride = 2, padding = 1), 
                        normalizations[normalization],
                        activations[activation]
                        ]
                    
                return block

            self.encoder = nn.Sequential(
                            *encoder_block(self.num_channel, self.nf), # DO NOT miss self when defining in class
                            *encoder_block(self.nf, self.nf * 2),
                            *encoder_block(self.nf * 2, self.nf * 4),
                            *encoder_block(self.nf * 4, self.nf * 8),
                            *encoder_block(self.nf * 8, self.nf * 16),
                            *encoder_block(self.nf * 16, self.nf * 32)
            )
            
            def decoder_block(in_filters, out_filters, normalization = 'batchnorm', activation = 'prelu', last_layer = False): # BEST: batchnorm + lrelu
                
                normalizations = nn.ModuleDict([
                        ['batchnorm', nn.BatchNorm3d(out_filters)],
                        ['instancenorm',nn.InstanceNorm3d(out_filters)],
                        ['groupnorm', nn.GroupNorm(num_groups=self.nf, num_channels=out_filters)]
                        ])
                        
                activations = nn.ModuleDict([
                        ['relu', nn.ReLU()],
                        ['prelu', nn.PReLU()],
                        ['lrelu', nn.LeakyReLU(self.leaky_rate)]
                        ])
                
                if not last_layer:
                    block = [nn.ConvTranspose3d(in_filters, out_filters, kernel_size = self.kernel_size, stride = 2, padding = 1, output_padding = 1), 
                             normalizations[normalization],
                             activations[activation]
                             ]
                elif last_layer:
                    block = [nn.ConvTranspose3d(in_filters, out_filters, kernel_size = self.kernel_size, stride = 2, padding = 1, output_padding = 1), 
                             normalizations[normalization],
                             nn.Sigmoid()]
                return block

            self.decoder = nn.Sequential(
                            *decoder_block(self.nf * 32, self.nf * 16),
                            *decoder_block(self.nf * 16, self.nf * 8),            
                            *decoder_block(self.nf * 8, self.nf * 4),
                            *decoder_block(self.nf * 4, self.nf * 2),
                            *decoder_block(self.nf * 2, self.nf),
                            *decoder_block(self.nf, self.num_channel, last_layer = True)
            )
            
            self.fc = nn.Sequential(
                    nn.Linear(self.nf * 32, self.nf * 4),
                    nn.Linear(self.nf * 4, self.num_classes)
                    ) # not better than original
#            self.fc = nn.Linear(self.nf * 32, self.num_classes) # original
                        
        def forward(self, x):
            z = self.encoder(x)
            x = self.decoder(z)
#            print(x.size()) # torch.Size([16, 1, 64, 64, 64]): To check the dimension for last FC layer with nn.Linear layer
            # 16 is the half of the batch_size=32, d/t nn.DataParallel
            z_flatten = z.view(z.size(0),-1)
            z_out = self.fc(z_flatten)
#            print(z_out.shape) # (16,1)
            
            return z_out, x # change this to return z_out, or return x, for using torchsummary
        
# to check the model
#from torchsummary import summary
#model = my_model().cuda() # MUST need .cuda()
#summary(model, input_size = (1,64,64,64))

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

# random split considering class imbalanced ratio
def random_split_ratio(dataset, label_filename, seed):
    np.random.seed(seed)
    label_dataset = np.load(label_filename)
#    print("pos num:", sum(label_dataset))
    all_ID_list = range(label_dataset.shape[0])
    pos_valid_test_ID_list = np.random.choice(sum(label_dataset), (sum(label_dataset)//10, 2), replace=False)
    pos_valid_ID_list = pos_valid_test_ID_list[:,0]
    pos_test_ID_list = pos_valid_test_ID_list[:,1]
    
    # pick valid and test index randomly without replacement
    neg_valid_test_ID_list = sum(label_dataset) + np.random.choice(label_dataset.shape[0] - sum(label_dataset), ((label_dataset.shape[0] - sum(label_dataset))//10, 2), replace=False)
    neg_valid_ID_list = neg_valid_test_ID_list[:,0]
    neg_test_ID_list = neg_valid_test_ID_list[:,1]
    
#    train_ID_list = set(all_ID_list) - set(neg_valid_ID_list) - set(pos_valid_ID_list) - set(neg_test_ID_list) - set(pos_test_ID_list) # use datatype 'set' to remove list A from list B
    valid_ID_list = list(neg_valid_ID_list) + list(pos_valid_ID_list)
    test_ID_list = list(neg_test_ID_list) + list(pos_test_ID_list)
    train_ID_list = set(all_ID_list) - set(valid_ID_list) - set(test_ID_list) # use datatype 'set' to remove list A from list B    
#    print("pos_valid_ID_list:", pos_valid_ID_list)#.shape[0])
#    print("pos_test_ID_list:", pos_test_ID_list)#.shape[0])
    
    # make id from 1 to 001
    train_ID_list = [("%03d" % i) for i in train_ID_list] # MUST change datatype via list
    valid_ID_list = [("%03d" % i) for i in valid_ID_list] 
    test_ID_list = [("%03d" % i) for i in test_ID_list] 

    # return partition and corresponding labels
    partition = {'train': train_ID_list, 'valid': valid_ID_list, 'test': test_ID_list}
#    print('valid:', valid_ID_list, 'test:', test_ID_list)
#    print('valid_size:', len(valid_ID_list), 'test_size:', len(test_ID_list))
    labels = {}
    for i in range(label_dataset.shape[0]):
      labels['%03d'% i] = label_dataset[i]
    return partition, labels

def my_transforms(x, input_size = 64, Gaussian = False, use_resnet = False):
    if Gaussian:
        x = (x - np.mean(x)) / np.std(x)
    else:
        # Min-max normalize
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
  # resized done already for server
#   x = resize(x, (input_size, input_size, input_size))
    if use_resnet:
        x = np.stack((x,) * 3, axis = -1)
#        print(x.shape) # (64, 64, 64, 3)
    else:
        x = x[:,:,:,None] # (64, 64, 64, 1)/ np.newaxis
    x = np.transpose(x, (3,2,0,1)) # (H,W,D,C) -> (C,D,H,W) for 3d tensor in torch
#    print(x.shape) # (1, 64, 64, 64)/ (3, 64, 64, 64) for resnet
    
    return x

def add_noise(img, noise_variance):
    noise = torch.randn(img.size()) * noise_variance
    noisy_img = img + noise
    return noisy_img

#def save_model()