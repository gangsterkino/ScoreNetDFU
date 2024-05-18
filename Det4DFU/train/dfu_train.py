import time
import os
import random
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import dfu_single_load
from networks.DiceBce import DiceBCELoss
import utils.image_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils.loss import *
from utils.metrics import diceCoeffv2, iou
from utils import misc
from utils.pytorchtools import EarlyStopping
from utils.LRScheduler import PolyLR

# 超参设置
crop_size = 256  # 输入裁剪大小
batch_size = 2  # batch size
n_epoch = 300  # 训练的最大epoch
early_stop__eps = 1e-3  # 早停的指标阈值
early_stop_patience = 15  # 早停的epoch阈值
initial_lr = 1e-5  # 初始学习率
threshold_lr = 1e-7  # 早停的学习率阈值
weight_decay = 5e-7  # 学习率衰减率
optimizer_type = 'adam'  # adam, sgd
scheduler_type = 'ReduceLR'  # ReduceLR, StepLR, poly
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, 1e6)

model_type = "unet"

if model_type == "unet":
    from networks.Unet import Baseline

root_path = '../'
fold = 5  # 训练集k-fold, 可设置1, 2, 3, 4, 5
depth = 10  # unet编码器的卷积层数
loss_name = 'dice_bce'  # dice, bce, wbce, dual, wdual
reduction = ''  # aug
model_name = '1_{}_depth={}_fold_{}_{}_{}{}_single'.format(model_type, depth, fold, loss_name, reduction, model_number)

# 训练集路径
# train_path = os.path.join(root_path, 'media/Datasets/bladder/Augdata_5folds', 'train{}'.format(fold), 'npy')
train_path = os.path.join(root_path, 'media2/Datasets/dfu')
val_path = os.path.join(root_path, 'media2/Datasets/dfu')


def main():
    # 定义网络
    net = Baseline(num_classes=dfu_single_load.num_classes, depth=depth).cuda()

    # 数据预处理
    center_crop = joint_transforms.CenterCrop(crop_size)
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()

    # 训练集加载
    train_set = dfu_single_load.Dataset(train_path, 'train', fold, joint_transform=None, center_crop=center_crop,
                                        transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # 验证集加载
    val_set = dfu_single_load.Dataset(val_path, 'val', fold,
                                      joint_transform=None, transform=input_transform, center_crop=center_crop,
                                      target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义损失函数
    if loss_name == 'dice_bce':
        criterion = DiceBCELoss(dfu_single_load.num_classes).cuda()

    # 定义早停机
    early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=early_stop__eps,
                                   path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))

    # 定义优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # 定义学习率衰减策略
    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    elif scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_type == 'poly':
        scheduler = PolyLR(optimizer, max_iter=n_epoch, power=0.9)
    else:
        scheduler = None

    train(train_loader, val_loader, net, criterion, optimizer, early_stopping, n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, early_stopping, num_epoches,
          iters):
    output_txt = open('train_log1-2.txt', 'w')
    miou_values = []  # 用于存储每个验证批次的mIoU

    for epoch in range(1, num_epoches + 1):
        st = time.time()
        train_class_dices = np.array([0] * (dfu_single_load.num_classes - 1), dtype=float)
        val_class_dices = np.array([0] * (dfu_single_load.num_classes - 1), dtype=float)
        val_dice_arr = []
        train_losses = []
        val_losses = []

        # 训练模型
        net.train()
        for batch, ((input, mask), file_name) in enumerate(train_loader, 1):
            X = input.cuda()
            y = mask.cuda()
            optimizer.zero_grad()

            output = net(X)
            output = torch.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())

            class_dice = []
            for i in range(1, dfu_single_load.num_classes):
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)
            string_print = 'epoch: {} - iters: {} - loss: {:.4} - mean_dice: {:.4} - dfu_dice: {:.4}  - time: {:.2}\n' \
                .format(epoch, iters, loss.data.cpu(), mean_dice, class_dice[0], time.time() - st)
            misc.log(string_print)
            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size

        print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_dfu: {:.4}\n'.format(
            epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0]))
        output_txt.write('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_dfu: {:.4}'.format(
            epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0]))
        output_txt.flush()

        # 验证模型
        net.eval()
        for val_batch, ((input, mask), file_name) in tqdm(enumerate(val_loader, 1)):
            val_X = input.cuda()
            val_y = mask.cuda()

            pred = net(val_X)
            pred = torch.sigmoid(pred)
            val_loss = criterion(pred, val_y)

            val_losses.append(val_loss.item())
            pred = pred.cpu().detach()
            val_class_dice = []
            for i in range(1, dfu_single_load.num_classes):
                val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

            val_dice_arr.append(val_class_dice)
            val_class_dices += np.array(val_class_dice)

            # 计算并记录当前批次的mIoU
            iou_batch = []  # 用于存储当前批次的IoU
            for i in range(1, dfu_single_load.num_classes):
                iou_batch.append(iou(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

            miou_batch = np.nanmean(iou_batch)  # 计算当前批次的mIoU，忽略NaN值
            miou_values.append(miou_batch)

        val_loss = np.average(val_losses)

        val_dice_arr = np.array(val_dice_arr)
        val_class_dices = val_class_dices / val_batch

        val_mean_dice = val_class_dices.sum() / val_class_dices.size

        # 计算并记录整个验证集的平均mIoU
        miou = np.nanmean(miou_values)  # 计算平均mIoU，忽略NaN值

        print('val_loss: {:.4} - val_mean_dice: {:.4} - mIoU: {:.4} - dfu: {:.4} '
              .format(val_loss, val_mean_dice, miou, val_class_dices[0]))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        output_txt.write('val_loss: {:.4} - val_mean_dice: {:.4} - mIoU: {:.4} - dfu: {:.4} \n'
                          .format(val_loss, val_mean_dice, miou, val_class_dices[0]))
        output_txt.flush()

        early_stopping(val_mean_dice, net, epoch)
        if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
            print("Early stopping")
            # 结束模型训练
            break
    output_txt.close()

    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stopping.save_epoch))
    print('stopped epoch {}'.format(epoch))
    print('----------------------------------------------------------')


if __name__ == '__main__':
    main()
