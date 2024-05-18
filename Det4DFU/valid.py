import os
import numpy as np
import cv2
import torch
import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
import utils.transforms as extended_transforms
import dfu_load
from utils.metrics import diceCoeffv2
from tqdm import tqdm
from networks.Unet import Baseline
from utils import helpers
import scipy

# 超参数设置
crop_size = 512
val_path = 'media/Datasets/dfu/raw_data'
center_crop = joint_transforms.CenterCrop(crop_size)
val_input_transform = extended_transforms.NpyToTensor()
target_transform = extended_transforms.MaskToTensor()

val_set = dfu_load.Dataset(val_path, 'val', 5,
                              joint_transform=None, transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette =  dfu_load.palette
num_classes =  dfu_load.num_classes

# 创建模型实例并加载权重
# depth为2的
net = Baseline(num_classes=dfu_load.num_classes, depth=8)
#net.load_state_dict(torch.load("checkpoint/unet_depth=2_fold_1_dice_502252.pth", map_location=torch.device('cpu')))
net.load_state_dict(torch.load("checkpoint/unet_depth=8_fold_5_dice_943839.pth", map_location=torch.device('cpu')))

net.eval()

def validate_single_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    input_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1)  # Change channel order

    # 进行预测
    with torch.no_grad():
        pred = net(input_tensor.unsqueeze(0))  # Add batch dimension
        pred = torch.sigmoid(pred)

    pred = pred.squeeze().detach().numpy()

    # 处理预测结果
    save_pred = helpers.onehot_to_mask(np.array(pred).transpose([1, 2, 0]), palette)
    save_pred_png = helpers.array_to_img(save_pred)

    # 保存预测结果
    output_image_path = "predicted_image/2.png"
    save_pred_png.save(output_image_path)
    print("Predicted image saved:", output_image_path)

if __name__ == '__main__':
    image_path = "predicted_image/1-1.png"
    validate_single_image(image_path)