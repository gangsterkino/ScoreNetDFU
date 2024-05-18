# Smart Diabetic Foot Ulcer Scoring System

[![License](https://img.shields.io/github/license/用户名/仓库名)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/用户名/仓库名)](https://github.com/用户名/仓库名/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/用户名/仓库名)](https://github.com/用户名/仓库名/network)

本项目开发了一个用于糖尿病足溃疡（DFU）风险分层的评分系统。

## 目录

- [介绍](#介绍)
- [模型训练](#模型训练)
- [安装](#安装)
- [使用方法](#使用方法)
- [贡献](#贡献)
- [许可证](#许可证)
- [联系信息](#联系信息)

## 介绍

为评估数据集，我们邀请了三位不同经验水平的皮肤科医生，包括一位初级医生（3年以上经验）、一位中级医生（5年以上经验）和一位高级医生（10年以上经验）。在模型训练阶段（如图1所示），我们采用了 U-Net 结构进行多类别分割任务，以分析糖尿病足溃疡，精准捕捉不同特征并提高分割准确性。我们使用包含医生标注的掩码的医院数据集进行训练。同时，我们选择了 ResNet50 作为基础网络，用于分类糖尿病足溃疡的病变特征。

## 模型训练

### 分割模型

在模型训练阶段，我们采用了 U-Net 结构进行多类别分割任务，以分析糖尿病足溃疡，精准捕捉不同特征并提高分割准确性。

```bash
# 训练 U-Net 模型
python train_unet.py
```
### 分类模型
我们选择了 ResNet50 作为基础网络，用于分类糖尿病足溃疡的病变特征。
```sh
# 训练 ResNet50 模型
python train_resnet50.py
```
## 安装
前提条件
Python 3.x
TensorFlow
PyTorch
其他必要的依赖项可以在 requirements.txt 中找到
