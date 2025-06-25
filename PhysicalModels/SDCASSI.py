import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path)
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 26))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W])
    return mask3d_batch

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def gen_meas_torch(data_batch, mask3d_batch):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1,keepdim=True)
    meas = meas / nC * 2

    return meas