import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from utils import helpers

'''
    0: 'Background',
    1: 'ulcer1',
    2: 'ulcer2',
    3: 'infection1',
    4: 'infection2',
    5: 'gangnere1',
    6: 'gangnere2'
'''
palette = [[0], [1], [2],[3],[4],[5],[6]]  # one-hot的颜色表
num_classes = 7  # 分类数
def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'labels')

        if 'Augdata' in root:  # 当使用增广后的训练集
            data_list = os.listdir(os.path.join(root, 'labels'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'labels')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        img_path = os.path.join(root, 'images')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, 'c0', it))
            items.append(item)
    return items


class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        file_name = mask_path.split('/')[-1]
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)

        img = np.array(img)
        mask = np.array(mask)
        
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, self.palette)


        #img = np.expand_dims(img, axis=2)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return (img, mask), file_name

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    from torch.utils.data import DataLoader
    import utils.image_transforms as joint_transforms
    import utils.transforms as extended_transforms

    def demo():
        train_path = r'media/Datasets/dfu/raw_data'
        val_path = r'media/Datasets/dfu/raw_data'
        test_path = r'media/Datase ts/dfu/test'
        # 设立数据处理
        center_crop = joint_transforms.CenterCrop(512)
        test_center_crop = joint_transforms.SingleCenterCrop(512)
        train_input_transform = extended_transforms.NpyToTensor()
        target_transform = extended_transforms.MaskToTensor()

        train_set = Dataset(train_path, 'train', 1,
                              joint_transform=None, center_crop=center_crop,
                              transform=train_input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

        for (input, mask), file_name in train_loader:
            batch_input = input[-1]  # Select the first sample from each batch

            print(batch_input.shape)
            transposed_input = batch_input.transpose(0, 2)
            transposed_input = transposed_input.transpose(0,1)
            # Transpose dimensions 0 and 2
            print(transposed_input.shape)

            #np.expand_dims(batch_input.squeeze(), 2)
            print(batch_input.shape)

            img = helpers.array_to_img(transposed_input)
            label_mask = np.argmax(mask[0], axis=0)
            label_mask_rgb = np.zeros_like(transposed_input)
            for idx, color in enumerate(palette):
                    label_mask_rgb[label_mask == idx] = color
            label_mask_img = helpers.array_to_img(label_mask_rgb)


            cv2.imshow('Input and Mask', np.hstack([img, label_mask_img]))
            cv2.waitKey(1000)


    demo()

