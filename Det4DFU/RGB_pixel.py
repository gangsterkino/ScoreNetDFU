import os
import cv2
import numpy as np

# RGB转化像素值
color_ranges = {
    (70, 70, 70): 'ulcer1',
    (90, 90, 90): 'ulcer2',
    (120, 120, 120): 'infection1',
    (140, 140, 140): 'infection2',
    (200, 200, 200): 'gangnere1',
    (220, 220, 220): 'gangnere2'
}

category_types = ['Background', 'ulcer1', 'ulcer2', 'infection1', 'infection2', 'gangnere1', 'gangnere2']
pixel_to_category = {
    0: 'Background',
    1: 'ulcer1',
    2: 'ulcer2',
    3: 'infection1',
    4: 'infection2',
    5: 'gangnere1',
    6: 'gangnere2'
}

def convert_color_to_pixel(color, color_ranges, category_types):
    if color in color_ranges:
        category = color_ranges[color]
        return category_types.index(category)
    return 0  # 默认为 Background

def process_image(input_path, output_folder, color_ranges, category_types):
    img = cv2.imread(input_path)
    h, w, _ = img.shape
    mask = np.zeros([h, w, 1], np.uint8)

    for y in range(h):
        for x in range(w):
            pixel_color = tuple(img[y, x])
            pixel_value = convert_color_to_pixel(pixel_color, color_ranges, category_types)
            mask[y, x] = pixel_value

    return mask

def main():
    # 输入和输出文件夹路径
    input_folder = "mask2"
    output_folder = "media/Datasets/dfu/raw_data/labels"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            mask = process_image(img_path, output_folder, color_ranges, category_types)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, mask)

    #通过像素值获取对应的类别
    pixel_value = 3
    category = pixel_to_category[pixel_value]
    print(f"Pixel Value: {pixel_value}, Corresponding Category: {category}")

    # 输出每个类别的像素值
    for idx, category in enumerate(category_types):
        pixel_value = idx
        print(f"Category: {category}, Pixel Value: {pixel_value}")

if __name__ == "__main__":
    main()
"""
# 单类别
import os
import cv2
import numpy as np

# RGB转化像素值
color_ranges = {
    (0, 0, 0): 'Background',     # 黑色
    (255, 255, 255): 'DFU'       # 白色
}

# 定义类别标签和像素值的映射关系
category_to_pixel = {
    'Background': 0,  # 黑色映射为0
    'DFU': 255         # 白色映射为255
}

def convert_color_to_pixel(color, color_ranges, category_to_pixel):
    if color in color_ranges:
        category = color_ranges[color]
        pixel_value = category_to_pixel.get(category, 0)  # 默认为0（背景）
        return pixel_value
    return 0  # 默认为0（背景）

def process_image(input_path, output_folder, color_ranges, category_to_pixel):
    img = cv2.imread(input_path)
    h, w, _ = img.shape
    mask = np.zeros([h, w], dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            pixel_color = tuple(img[y, x])
            pixel_value = convert_color_to_pixel(pixel_color, color_ranges, category_to_pixel)
            mask[y, x] = pixel_value

    return mask

def main():
    # 输入和输出文件夹路径
    input_folder = "mask3"
    output_folder = "media2/Datasets/dfu/raw_data/labels"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            mask = process_image(img_path, output_folder, color_ranges, category_to_pixel)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    main()
"""
