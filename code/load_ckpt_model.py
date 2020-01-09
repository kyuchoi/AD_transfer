#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:48:13 2019

@author: kyuchoi
"""

import torch
import os
from my_utils import my_model

home_path = '/home/kyuchoi/AD_transfer'
model_dir = os.path.join(home_path, 'model')
model_filename = 'ae_test_6590.ckpt' # load the checkpoint model

model = my_model()
# MUST wrap the called model with nn.DataParallel, if you used the nn.DataParallel when you define the model
model = nn.DataParallel(model, device_ids=None)

model.load_state_dict(torch.load(os.path.join(model_dir, model_filename)))
model.eval() # you should use the model with evaluation mode to simply test the model

# print out the state_dict of the model
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print out the state_dict of the optimizer
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

