import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import csv
from _internal_logic import _internal_calculate_bias
import os
import numpy as np
# 类别与对应颜色映射
color_ranges = {
    'ulcer1': [70,70,70],
    'ulcer2': [90,90,90],
    'infection2': [140,140,140],
    'infection1': [120,120,120],
    'genere1': [200,200,200],
    'genere2': [220,220,220]
}
# 特征权重值
model1_weights = [9.0, 7.0, 0.0, 5.0]
model2_weights = [3.0, 4.0, 2.2, 2.9, 2.4, 2.5]

# 获取类别特征概率
def feature(image_path):
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    pretrained_model.fc = torch.nn.Linear(2048, 4)
    model_weights_path = 'model/model3-2.pt'
    pretrained_model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    pretrained_model.eval()
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = pretrained_model(input_tensor)
    # 获取预测结果
    probabilities = F.softmax(output, dim=1)[0]
    predicted_classes = torch.argmax(output, dim=1)

    # 获取特征向量
    features = pretrained_model.conv1(input_tensor)
    features = pretrained_model.bn1(features)
    features = pretrained_model.relu(features)
    features = pretrained_model.maxpool(features)
    features = pretrained_model.layer1(features)
    features = pretrained_model.layer2(features)
    features = pretrained_model.layer3(features)
    features = pretrained_model.layer4(features)
    features = pretrained_model.avgpool(features)
    features = torch.flatten(features, 1)
    return probabilities, features

def extract_color(image, color_range):
    lower_range = np.array(color_range[0], dtype=np.uint8)
    upper_range = np.array(color_range[1], dtype=np.uint8)
    mask = cv2.inRange(image, lower_range, upper_range)
    return mask

def calculate_color_areas(image):
    total_color_area = 0
    color_areas = {}
    for color, color_range in color_ranges.items():
        mask = extract_color(image, color_range)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            color_area += area
        color_areas[color] = color_area
        total_color_area += color_area
        """
        for color, area in color_areas.items():
            ratio = area / total_color_area if total_color_area != 0 else 0
            print(f"{color} 面积：{area} 像素  比例：{ratio:.4f}")
        """

    return total_color_area, color_areas

def process_single_image(image_path):
    image = cv2.imread(image_path)
    total_area, color_areas = calculate_color_areas(image)
    return total_area, color_areas

# 计算DFU得分
def calculate_score(image_path, mask_path, model1_weights, model2_weights):
    probabilities, features = feature(image_path)
    total_area, color_areas = process_single_image(mask_path)
    # 计算特征得分
    xa = probabilities[0] * model1_weights[0] * ((color_areas['genere1'] * model2_weights[0] + color_areas['genere2'] * model2_weights[1]) / total_area)
    xb = probabilities[1] * model1_weights[1] * ((color_areas['infection1'] * model2_weights[2] + color_areas['infection2'] * model2_weights[3]) / total_area)
    xc = probabilities[3] * model1_weights[3] * ((color_areas['ulcer1'] * model2_weights[4] + color_areas['ulcer2'] * model2_weights[5]) / total_area)
    c=2
    x1 = c*xa
    x2 = c*xb
    x3 = c*xc
    score = x1 + x2 + x3
    score,bias = _internal_calculate_bias(score, total_area, x1, x2, x3)
    score = min(max(0, score + bias), 100)
    return x1, x2, x3, score
"""

# 保存结果
csv_filename = 'results2.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Image Filename', 'Score']  # CSV文件的列名
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()


    image_folder = "/Users/tanxinyu/DFU/医生/"  # 图像文件夹路径

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") and not filename.endswith("_det.jpg"):
            image_path = os.path.join(image_folder, filename)

            mask_filename = filename.replace(".jpg", "_det.jpg")
            mask_path = os.path.join(image_folder, mask_filename)
            if not os.path.exists(mask_path):
                mask_path = None
            if image_path and mask_path:
                x1, x2, x3, score = calculate_score(image_path, mask_path, model1_weights, model2_weights)
                print(f"Image: {filename}, Score: {score/10}")
                # 写入CSV文件
                writer.writerow({'Image Filename': filename, 'Score': score/10})
"""
# 样例测试
image_path = "dataset1/infection/1100-open-0010.jpg"
mask_path = "/Users/tanxinyu/seg_GPU/mask_origin/1+2_1100-open-0010_det.jpg"
x1, x2, x3, score = calculate_score(image_path, mask_path, model1_weights, model2_weights)
print(f"得分:{score/10}")

