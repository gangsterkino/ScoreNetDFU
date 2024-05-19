
import os, shutil
from sklearn.model_selection import KFold
# K折交叉验证划分数据集
def dataset_kfold(dataset_dir, save_path):
    data_list = os.listdir(dataset_dir)

    kf = KFold(n_splits=5, shuffle=True, random_state=12345)

    for i, (tr, val) in enumerate(kf.split(data_list), 1):
        print(len(tr), len(val))
        if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):

            print('清空原始数据中...')
            os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
            os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))
            print('原始数据已清空。')

        for item in tr:
            file_name = data_list[item]
            with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')

        for item in val:
            file_name = data_list[item]
            with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')
"""
# 多类别
if __name__ == '__main__':
    # 首次划分数据集或者重新划分数据集时运行
    dataset_kfold(os.path.join('media/Datasets', 'dfu/labels'),
                  os.path.join('media/Datasets', 'dfu'))
"""
# 单类别
if __name__ == '__main__':

    # 首次划分数据集或者重新划分数据集时运行
    dataset_kfold(os.path.join('media3', 'dfu/labels'),
                  os.path.join('media3', 'dfu'))
