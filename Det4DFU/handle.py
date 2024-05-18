"""
import cv2
import os

# 原始图像文件夹路径
original_image_folder = 'media2/Datasets/dfu/raw_data/images'
# 掩码图像文件夹路径
mask_image_folder = 'media2/Datasets/dfu/raw_data/labels'
# 输出文件夹路径
output_folder = 'ori2'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历原始图像文件夹
for filename in os.listdir(original_image_folder):
    if filename.endswith('.png'):
        original_image_path = os.path.join(original_image_folder, filename)
        mask_image_path = os.path.join(mask_image_folder, filename)

        # 加载原始图像和掩码图像
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        # 将掩码图像应用到原始图像，其他部分变为黑色
        masked_image = cv2.bitwise_and(original_image, original_image, mask=mask_image)

        # 保存新的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, masked_image)

        print(f"Processed and saved: {output_path}")

"""
import cv2
import os

# 原始图像路径
original_image_path = 'predicted_image/2.png'
# 掩码图像路径
mask_image_path = 'predicted_image/1.png'
# 输出路径
output_path = 'predicted_image/2-1.png'

# 加载原始图像和掩码图像
original_image = cv2.imread(original_image_path)
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# 创建一个副本以保留原始图像
result_image = original_image.copy()

# 将掩码应用于原始图像，其他部分变为黑色
result_image[mask_image == 0] = [0, 0, 0]  # 设置掩码外的像素为黑色

# 保存新的图像
cv2.imwrite(output_path, result_image)

print(f"Processed and saved: {output_path}")

