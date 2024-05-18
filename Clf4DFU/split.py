import os
import shutil
from sklearn.model_selection import train_test_split
# 数据集路径
main_folder = 'Data_new_2'

# 设置用于存储划分后数据的文件夹路径
train_folder = 'split2/train'
validation_folder = 'split2/val'
test_folder = 'split2/test'

class_folders = os.listdir(main_folder)

for class_folder in class_folders:
    if class_folder.startswith('.'):
        continue
    class_path = os.path.join(main_folder, class_folder)

    # 获取当前类别的所有文件，并排除隐藏文件
    file_list = [file_name for file_name in os.listdir(class_path) if not file_name.startswith('.')]

    # 划分数据
    train_data, temp_data = train_test_split(file_list, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.2, random_state=42)

    # 创建存储文件夹
    os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(validation_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_folder, class_folder), exist_ok=True)

    # 复制文件到相应的文件夹
    for file_name in train_data:
        source = os.path.join(class_path, file_name)
        destination = os.path.join(train_folder, class_folder, file_name)
        shutil.copy(source, destination)

    for file_name in validation_data:
        source = os.path.join(class_path, file_name)
        destination = os.path.join(validation_folder, class_folder, file_name)
        shutil.copy(source, destination)

    for file_name in test_data:
        source = os.path.join(class_path, file_name)
        destination = os.path.join(test_folder, class_folder, file_name)
        shutil.copy(source, destination)
