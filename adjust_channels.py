from architecture.LKMSFRNet import LKMSFRNet
from PhysicalModels.loss import *


def model_channelchange(new_in_channels, new_out_channels, dit):
    model = LKMSFRNet()

    # 获取原始卷积层的权重
    pretrained_dict = torch.load(dit, map_location=torch.device('cpu'))
    original_conv1_weight = pretrained_dict['patch_embed.proj.weight']
    original_conv2_weight = pretrained_dict['patch_unembed.proj.0.weight']

    # 修改输入卷积层
    new_conv1_weight = torch.zeros((48, new_in_channels, 3, 3))
    new_conv1_weight[:, :28, :, :] = original_conv1_weight
    model.patch_embed.proj = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=48,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode='reflect'
    )
    model.patch_embed.proj.weight = nn.Parameter(new_conv1_weight)

    # 修改输出卷积层
    new_conv2_weight = torch.zeros((new_out_channels, 48, 3, 3))
    new_conv2_weight[:29, :, :, :] = original_conv2_weight
    model.patch_unembed.proj = nn.Conv2d(
        in_channels=48,
        out_channels=new_out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode='reflect'
    )
    model.patch_unembed.proj.weight = nn.Parameter(new_conv2_weight)

    # 加载其他预训练权重
    model_dict = model.state_dict()
    transfer_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and not k.startswith(('patch_embed.proj', 'patch_unembed.proj'))
    }
    model.load_state_dict(transfer_dict, strict=False)

    return model




