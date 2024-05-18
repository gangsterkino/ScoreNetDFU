import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torchvision

# 加载resnet50模型
pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# pretrained_model =models.resnet50(pretrained =True)

pretrained_model.fc = torch.nn.Linear(2048, 4)

# 设定保存的模型一的参数权重
model_weights_path = 'models/model1.pt'
pretrained_model.eval()

# 定义转换操作，用于将图像转换为模型所需的输入格式
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)


# 模型训练
# 定义训练数据和标签
train_data = torchvision.datasets.ImageFolder('split/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)

# 循环训练
num_epochs = 40
losses=[]

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = pretrained_model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss=running_loss / len(train_loader)
        losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# 保存训练完的模型的参数权重
torch.save(pretrained_model.state_dict(), model_weights_path)

# 模型的分类预测
def classify_and_locate(image_path):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).unsqueeze(0)
    # 分类预测
    with torch.no_grad():
        outputs = pretrained_model(transformed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        label = ["gangrene", "infection","normal", "ulcer"][predicted.item()]
        print(f"分类概率：{probabilities}")
    return label, probabilities

# 测试用例
image_path = 'validset1/infection/1100-open-0267.jpg'  # 图片路径
predicted_label, probabilities = classify_and_locate(image_path)
print(f"预测的类别：{predicted_label}")
