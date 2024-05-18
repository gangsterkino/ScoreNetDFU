import cv2
import numpy as np
import os
from PIL import Image
import glob

def extract_color_from_mask(mask_image, target_colors_rgb, threshold_range):
    """
    提取mask图像中指定颜色范围内的像素，并返回提取的图像列表。

    参数：
    - mask_image: 输入的mask图像，使用OpenCV加载。
    - target_colors_rgb: 目标颜色的RGB值列表，形式为[[R1, G1, B1], [R2, G2, B2], ...]。
    - threshold_range: RGB颜色阈值范围，用于提取目标颜色范围内的像素。

    返回：
    返回提取的图像列表，每个图像都是一个NumPy数组。
    """
    extracted_images = []
    for target_color in target_colors_rgb:
        lower_bound = np.array([max(0, x - threshold_range) for x in target_color])
        upper_bound = np.array([min(255, x + threshold_range) for x in target_color])
        color_mask = cv2.inRange(mask_image, lower_bound, upper_bound)
        extracted_images.append(color_mask)
    return extracted_images

def resize_images_in_directory(root_directory, target_size=(224, 224)):
    """
    将指定目录下的所有图像调整为目标大小并保存。

    参数：
    - root_directory: 包含子文件夹的根目录的路径。
    - target_size: 目标图像大小，形式为元组 (width, height)。默认为 (224, 224)。

    返回：
    无返回值，但会将调整大小后的图像保存在原始文件的位置。
    """
    for subdir, _, _ in os.walk(root_directory):
        for filename in os.listdir(subdir):
            image_path = os.path.join(subdir, filename)
            try:
                with Image.open(image_path) as img:
                    resized_img = img.resize(target_size, Image.ANTIALIAS)
                    resized_img.save(image_path)
                    print(f"Resized: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def delete_hidden_files(folder_path):
    """
    删除指定文件夹中的隐藏文件。

    参数：
    - folder_path: 要删除隐藏文件的文件夹路径。

    返回：
    无返回值，但会删除文件夹中的隐藏文件。
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted hidden file: {file_path}")

if __name__ == "__main__":
    # 示例使用 extract_color_from_mask 函数
    mask_image = cv2.imread('mask.jpg')
    target_colors_rgb = [
        [70, 70, 70],
        [90, 90, 90],
        [140, 140, 140],
        [110, 110, 110],
        [120, 120, 120],
        [200, 200, 200],
        [220, 220, 220]
    ]
    threshold_range = 10
    extracted_images = extract_color_from_mask(mask_image, target_colors_rgb, threshold_range)

    # 保存提取的图像（示例）
    for i, extracted_image in enumerate(extracted_images):
        save_path = f'extracted_color_{i}.jpg'
        cv2.imwrite(save_path, extracted_image)
        print(f"Saved extracted color image {i} to {save_path}")

    # 示例使用 resize_images_in_directory 函数
    root_directory = 'dataset1_handle'
    target_size = (224, 224)
    resize_images_in_directory(root_directory, target_size)

    # 示例使用 delete_hidden_files 函数
    folder_path = "split_data"  # 将此处替换为要遍历的文件夹路径
    delete_hidden_files(folder_path)
