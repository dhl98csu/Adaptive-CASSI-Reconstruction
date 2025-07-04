import torch
import torch.nn as nn
import torch.optim as optim
from intitt import *
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from PhysicalModels.SDCASSI2 import generate_masks, gen_meas_torch, shift

from PhysicalModels.loss import *
from dataloder import read_mat_files, add_noise_and_crop_resize
import cv2
import numpy as np
import scipy.io
import math
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import os
from adjust_channels import model_channelchange
from torchvision.transforms.functional import center_crop


def create_scheduler(optimizer, T, eta_min):
    initial_lr = optimizer.param_groups[0]['lr']
    return CosineAnnealingLR(optimizer, T_max=T, eta_min=eta_min)


def test(model, GT, measurement):
    model.eval()

    with torch.no_grad():
        outputs = model(measurement)
        avg_psnr, avg_psnr_loss = batch_psnrloss(GT, outputs)
        ssim, ssim_loss = compute_batch_ssim(GT, outputs)

    print(f' Test PNSR: {avg_psnr}')
    print(f' Test SSIM: {ssim}')

    return 0


torch.manual_seed(1)
torch.cuda.set_device(1)
device = torch.device('cuda')

num_epochs = 2400
batch_size = 1


data_path = r"./data/Harvard_test"
test_data_path = r".\data\Harvard_test"
mask_path = r"./mask"
lossoutput_dir = r".\loss_cuve"
save_dir = r"./result"
# mask_data= scipy.io.loadmat(mask_path)
# mask = mask_data['mask']


mask = generate_masks(mask_path, batch_size)
mask = mask.to(device)
mask_shift = shift(mask, step=2)

criterion = nn.MSELoss()
loss_fn1 = PSNRLoss()
loss_fn2 = GradientLoss()
index = 0




former_pnsr = 0
best_pnsr = 0
intial_pnsr = 0
data_times = []
epochs = []
losses = []
psnrs = []
max_stagnant_epochs = 2000  
max_outer_epochs = 1  
max_data_files = 50  
stagnant_epochs = 0  


data_initial_psnrs = [[] for _ in range(max_data_files)]
data_best_psnrs = [[] for _ in range(max_data_files)]
data_improve_psnrs = [[] for _ in range(max_data_files)]
data_end_epochs = [[] for _ in range(max_data_files)]

data_initial_ssims = [[] for _ in range(max_data_files)]
data_best_ssims = [[] for _ in range(max_data_files)]
data_improve_ssims = [[] for _ in range(max_data_files)]
data_end_ssims = [[] for _ in range(max_data_files)]

mask3d_batch, input_mask = init_mask(mask_path, 'Phi_PhiPhiT', 1)

for data_index in range(0, max_data_files):  
    start_data_time = time.time()
 
    ground_truth = read_mat_files(data_path, data_index).to(device)
    data_batch, measurement = gen_meas_torch(ground_truth, mask, is_training=True)

    for outer_epoch in range(max_outer_epochs):  
        model = model_channelchange(new_in_channels=31, new_out_channels=32, dit='')
        model.to(device)
        T_max = 800
        initial_lr = 0.00008
        optimizer = optim.Adam(model.parameters(), lr=0.00008)
        scheduler = create_scheduler(optimizer, T=T_max, eta_min=0.000001)
        best_loss = float('inf')  
        best_model_path = 'best_model.pth'  
        best_pnsr = float('-inf')
        best_ssim = float('-inf')
        intial_pnsr = None
        former_pnsr = None
        improve_pnsr = None
        intial_ssim = None
        former_ssim = None
        improve_ssim = None

        for epoch in range(num_epochs):

            outputs = model(data_batch)
            augmented_outputs = augment_batch_data(outputs)

            data_nextepoch, model_measurement = gen_meas_torch(outputs, mask)
            augmented_y, augmented_measurement = gen_meas_torch(augmented_outputs, mask)
            augmented_y_outputs = model(augmented_y)

   
            avg_psnr = psnr(ground_truth, outputs)
            avg_ssim = ssimloss(ground_truth, outputs).detach().cpu().numpy()

            if epoch == 0:
                intial_pnsr = avg_psnr
                intial_ssim = avg_ssim

  
            loss1 = loss_fn1(model_measurement, measurement)
            mseloss2 = criterion(augmented_y_outputs, augmented_outputs)
            loss = loss1 + 200 * mseloss2

 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % T_max == 0:
                
                initial_lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr
                    # 重置调度器以应用新的初始学习率
                    scheduler = create_scheduler(optimizer, T_max, eta_min=0.000001)


            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)


            if avg_psnr > best_pnsr:
                best_pnsr = avg_psnr
                stagnant_epochs = 0  
                torch_tensor = outputs
                result_numpy = torch_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                result = {'img': result_numpy}
                output_file = os.path.join(save_dir, f'Harvard_test{data_index}.mat')
                sio.savemat(output_file, result)
            else:
                stagnant_epochs += 1

            if avg_ssim > best_ssim:
                best_ssim = avg_ssim

            # former_pnsr = avg_psnr
            improve_pnsr = best_pnsr - intial_pnsr

            improve_ssim = best_ssim - intial_ssim

            print(f'Data_num [{data_index}]')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
            print(f' PSNR: {avg_psnr}')
            print(f' Initial PSNR: {intial_pnsr}')
            print(f' Best PSNR: {best_pnsr}')
            print(f' PSNR improvement: {improve_pnsr}')

            print(f' SSIM: {avg_ssim}')
            print(f' Initial ssim: {intial_ssim}')
            print(f' Best ssim: {best_ssim}')
            print(f' ssim improvement: {improve_ssim}')
            epochs.append(epoch)
            losses.append(loss.clone().detach().cpu().numpy())
            psnrs.append(avg_psnr)

   
            if stagnant_epochs >= max_stagnant_epochs:
                break

        end_data_time = time.time()
        data_time = end_data_time - start_data_time
        data_times.append(data_time)

        data_initial_psnrs[data_index].append(intial_pnsr)
        data_best_psnrs[data_index].append(best_pnsr)
        data_improve_psnrs[data_index].append(improve_pnsr)

        data_initial_ssims[data_index].append(intial_ssim)
        data_best_ssims[data_index].append(best_ssim)
        data_improve_ssims[data_index].append(improve_ssim)


        data_end_epochs[data_index].append(epoch + 1)

        print(f'End of outer loop epoch {outer_epoch + 1}')
        print(f' Initial PSNR: {intial_pnsr}')
        print(f' Best PSNR: {best_pnsr}')
        print(f' PSNR improvement: {improve_pnsr}')

        print(f' Initial ssim: {intial_ssim}')
        print(f' Best ssim: {best_ssim}')
        print(f' ssim improvement: {improve_ssim}')
        print(f' End epoch of outer loop: {epoch + 1}')




avg_best_psnr = np.mean(data_best_psnrs)
avg_best_ssim = np.mean(data_best_ssims)


total_time = sum(data_times)
average_time = total_time / len(data_times)
print(f"Average processing time per data file: {average_time:.2f} seconds")
print(f'Average Best PSNR: {avg_best_psnr}')
print(f'Average Best SSIM: {avg_best_ssim}')

print('Training finished.')

print('Training finished.')
