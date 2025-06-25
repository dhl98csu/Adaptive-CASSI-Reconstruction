import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
np.random.seed(0)

def generate_masks(mask_path, batch_size):

    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    #mask = np.random.randint(2, size=(256, 256))
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2          # meas scale
    y_temp = shift_back(meas)

    PhiTy = torch.mul(y_temp, mask3d_batch)

    return PhiTy,meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output

def shift_back(inputs,step=2):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output