#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("ls '/home/kyuchoi/AD_transfer/data'")

import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

#%% making npy datasets for each group
data_dir = '/home/kyuchoi/AD_transfer/data'
group_name = ['AD','MCI_neg','MCI_pos','YC','PD','NL']
#%%
len_group_total = {}
for (j, group) in enumerate(group_name):
  print(j, group)
  group_4D = np.load(os.path.join(data_dir,"{}_4D_64.npy".format(group)))
  len_group_total[group] = group_4D.shape[0]
  for i in range(group_4D.shape[0]):
    if not os.path.isdir(os.path.join(data_dir, "{}".format(group))):
      os.mkdir(os.path.join(data_dir, "{}".format(group)))
    print(i, group_4D[i].shape)
#    np.save(os.path.join(data_dir,"{}".format(group),"%s_%03d.npy"% (group, i)), group_4D[i])
# save number of data for each group
print(len_group_total)


# In[2]:


### copying AD and NL npy files into a single folder, "AD_NL"
get_ipython().system("mkdir '/home/kyuchoi/AD_transfer/data/AD_NL'")
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/AD/*.npy /home/kyuchoi/AD_transfer/data/AD_NL')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/NL/*.npy /home/kyuchoi/AD_transfer/data/AD_NL')


# In[3]:


AD_NL_list = os.listdir(os.path.join(data_dir, "AD_NL"))
for (i, AD_NL_filename) in enumerate(AD_NL_list):
  # print(AD_NL_filename)
  os.rename(os.path.join(data_dir, "AD_NL", AD_NL_filename), os.path.join(data_dir, "AD_NL", "AD_NL_%03d.npy" % (i))) # MUST make filename starts as AD_000.npy, instead of AD_001.npy, not to get an error with dataloader


# In[4]:


label_AD_NL = np.array([1] * len_group_total["AD"] + [0] * len_group_total["NL"])
# print(label_AD_NL)
np.save(os.path.join(data_dir, "label_AD_NL.npy"), label_AD_NL)


# In[5]:


### copying MCI_pos and MCI_neg npy files into a single folder, "MCI_pos_neg"
get_ipython().system("mkdir '/home/kyuchoi/AD_transfer/data/MCI_pos_neg'")
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/MCI_pos/*.npy /home/kyuchoi/AD_transfer/data/MCI_pos_neg')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/MCI_neg/*.npy /home/kyuchoi/AD_transfer/data/MCI_pos_neg')


# In[6]:


MCI_pos_neg_list = os.listdir(os.path.join(data_dir, "MCI_pos_neg"))
for (i, MCI_pos_neg_filename) in enumerate(MCI_pos_neg_list):
  # print(MCI_pos_neg_filename)
  os.rename(os.path.join(data_dir, "MCI_pos_neg", MCI_pos_neg_filename), os.path.join(data_dir, "MCI_pos_neg", "MCI_pos_neg_%03d.npy" % (i)))


# In[7]:


label_MCI_pos_neg = np.array([1] * len_group_total["MCI_pos"] + [0] * len_group_total["MCI_neg"])
print(label_MCI_pos_neg.shape)
np.save(os.path.join(data_dir, "label_MCI_pos_neg.npy"), label_MCI_pos_neg)

#%% making non_AD group with PD, YC, and NL

### copying PD and YC npy files into a single folder, "AD_non_AD"
get_ipython().system("mkdir '/home/kyuchoi/AD_transfer/data/AD_non_AD'")
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/AD/*.npy /home/kyuchoi/AD_transfer/data/AD_non_AD')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/PD/*.npy /home/kyuchoi/AD_transfer/data/AD_non_AD')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/YC/*.npy /home/kyuchoi/AD_transfer/data/AD_non_AD')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/NL/*.npy /home/kyuchoi/AD_transfer/data/AD_non_AD')

# In[6]:

AD_non_AD_list = sorted(os.listdir(os.path.join(data_dir, "AD_non_AD")))
for (i, AD_non_AD_filename) in enumerate(AD_non_AD_list):
#    print(AD_non_AD_filename)
    os.rename(os.path.join(data_dir, "AD_non_AD", AD_non_AD_filename), os.path.join(data_dir, "AD_non_AD", "AD_non_AD_%03d.npy" % (i)))
# In[7]:

label_AD_non_AD = np.array([1] * len_group_total["AD"] + [0] * (len_group_total["PD"]+len_group_total["YC"]+len_group_total["NL"]))
print(label_AD_non_AD.shape) # (960,)
np.save(os.path.join(data_dir, "label_AD_non_AD.npy"), label_AD_non_AD)


#%% making AD_normal group with YC, and NL

### copying PD and YC npy files into a single folder, "AD_normal"
get_ipython().system("mkdir '/home/kyuchoi/AD_transfer/data/AD_normal'")
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/AD/*.npy /home/kyuchoi/AD_transfer/data/AD_normal')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/YC/*.npy /home/kyuchoi/AD_transfer/data/AD_normal')
get_ipython().system('cp /home/kyuchoi/AD_transfer/data/NL/*.npy /home/kyuchoi/AD_transfer/data/AD_normal')

# In[6]:

AD_normal_list = sorted(os.listdir(os.path.join(data_dir, "AD_normal")))
for (i, AD_normal_filename) in enumerate(AD_normal_list):
    print(AD_normal_filename)
    os.rename(os.path.join(data_dir, "AD_normal", AD_normal_filename), os.path.join(data_dir, "AD_normal", "AD_normal_%03d.npy" % (i)))
# In[7]:

label_AD_normal = np.array([1] * len_group_total["AD"] + [0] * (len_group_total["YC"]+len_group_total["NL"]))
print(label_AD_normal.shape) # (610,)
np.save(os.path.join(data_dir, "label_AD_normal.npy"), label_AD_normal)
