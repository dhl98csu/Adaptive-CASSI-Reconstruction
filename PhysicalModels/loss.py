import torch
import torch.nn.functional as F
import torch.nn as nn

def ssimloss(img1, img2, window_size=11, sigma=1.5, size_average=True):
    # Data normalization
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local luminance variance and covariance
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    # Compute SSIM
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).cpu().detach().numpy()


def mssimloss(img1, img2, device,window_size=11, sigma=1.5, size_average=True):
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32).to(device)
    levels = weights.size(0)
    mssim_val = []
    mcs_val = []

    for _ in range(levels):
        ssim_map = ssimloss(img1, img2, window_size, sigma, size_average=False)
        mssim_val.append(ssim_map.mean())
        mcs_val.append(((img1.mean() - img2.mean()) ** 2).mean())
        img1 = F.avg_pool2d(img1, 2)
        img2 = F.avg_pool2d(img2, 2)

    mssim_val = torch.stack(mssim_val)
    mcs_val = torch.stack(mcs_val)

    return (torch.prod(mcs_val[0:levels - 1] ** weights[0:levels - 1]) *
                (mssim_val[levels - 1] ** weights[levels - 1]))


def gaussian_kernel(size, sigma):
    kernel = torch.linspace(-(size // 2), size // 2, size)
    kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
    kernel = kernel / kernel.sum()
    return kernel


def create_gaussian_filter(kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel_2d = kernel.view(1, -1) @ kernel.view(-1, 1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kernel_size, kernel_size]
    return kernel_2d


def ssim(img1, img2, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    def filter2D(img, kernel):
        padding = kernel.size(2) // 2
        return F.conv2d(img, kernel, padding=padding)

    kernel = create_gaussian_filter(window_size, sigma).to(img1.device)

    mu1 = filter2D(img1, kernel)
    mu2 = filter2D(img2, kernel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2D(img1 * img1, kernel) - mu1_sq
    sigma2_sq = filter2D(img2 * img2, kernel) - mu2_sq
    sigma12 = filter2D(img1 * img2, kernel) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_n / ssim_d
    return ssim_map.cpu().numpy().mean()


def psnr(original, compressed):
    # 将张量转换为浮点数
    original = original.float()
    compressed = compressed.float()

    # 计算像素值的最大可能值
    max_val = torch.max(original)

    # 计算均方误差(MSE)
    mse = F.mse_loss(original, compressed)

    # 计算PSNR
    psnr = 10 * torch.log10((max_val ** 2) / mse)

    return psnr.item()  # 将张量转换为标量值并返回


def batch_psnr(original_batch, compressed_batch):
    # 将张量转换为浮点数
    original_batch = original_batch.float()
    compressed_batch = compressed_batch.float()

    # 计算每个图像的PSNR
    psnr_values = []
    for original, compressed in zip(original_batch, compressed_batch):
        max_min = torch.max(original)-torch.min(original)
        mse = F.mse_loss(original, compressed)
        psnr = 10 * torch.log10((max_min ** 2) / mse)
        psnr_values.append(psnr.item())

    # 计算整个批次的平均PSNR
    avg_psnr = sum(psnr_values) / len(psnr_values)

    return avg_psnr

def batch_psnrloss(original_batch, compressed_batch):
    # 将张量转换为浮点数
    original_batch = original_batch.float()
    compressed_batch = compressed_batch.float()

    # 计算每个图像的PSNR
    psnr_values = []
    for original, compressed in zip(original_batch, compressed_batch):
        max_val = torch.max(original)
        mse = F.mse_loss(original, compressed)
        psnr = 10 * torch.log10((max_val ** 2) / mse)
        psnr_values.append(psnr.item())

    # 计算整个批次的平均PSNR
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_psnr_loss = 100-avg_psnr

    return avg_psnr,avg_psnr_loss

def compute_batch_ssim(imgs1, imgs2):
    batch_ssim = []
    for i in range(imgs1.shape[0]):
        ssim_val = ssimloss(imgs1[i], imgs2[i])
        batch_ssim.append(ssim_val.item())
    avg_ssim = sum(batch_ssim) / len(batch_ssim)
    ssim_loss = 1-avg_ssim

    return avg_ssim,ssim_loss

# Compute MSSIM for a batch of images
def compute_batch_mssim(imgs1, imgs2):
    batch_mssim = []
    for i in range(imgs1.shape[0]):
        mssim_val = mssimloss(imgs1[i], imgs2[i])
        batch_mssim.append(mssim_val.item())
    avg_mssim = sum(batch_mssim) / len(batch_mssim)
    mssim_loss = 1-avg_mssim
    return avg_mssim,mssim_loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, pred, target):
        # 计算 MSE
        mse = F.mse_loss(pred, target)

        # 计算最大像素值
        max_val = torch.max(target) # 对整个 batch 的 target 取最大值

        # 如果 MSE 非常接近 0，则 PSNR 为无穷大
        if mse == 0:
            return float('inf')

        # 计算 PSNR
        #psnr = 20 * torch.log10(max_val/ torch.sqrt(mse))
        psnr = 10 * torch.log10(max_val**2 / mse)
        # PSNR 越大，损失越小，因此取负号作为损失
        loss = 100-psnr
        #loss = 1/psnr

        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, hsi_image):
        # Compute mean across channels to synthesize grayscale image
        gray_image = torch.mean(hsi_image, dim=1, keepdim=True)  # shape: (batch_size, 1, height, width)

        # Compute gradients of hsi_image along height and width dimensions
        hsi_grad_x = torch.abs(hsi_image[:, :, :, :-1] - hsi_image[:, :, :, 1:])
        hsi_grad_y = torch.abs(hsi_image[:, :, :-1, :] - hsi_image[:, :, 1:, :])

        # Compute gradients of gray_image along height and width dimensions
        gray_grad_x = torch.abs(gray_image[:, :, :, :-1] - gray_image[:, :, :, 1:])
        gray_grad_y = torch.abs(gray_image[:, :, :-1, :] - gray_image[:, :, 1:, :])

        # Calculate average gradient loss
        loss_x = torch.mean(torch.abs(hsi_grad_x - gray_grad_x))
        loss_y = torch.mean(torch.abs(hsi_grad_y - gray_grad_y))
        loss = (loss_x + loss_y) / 2.0

        return 1/loss