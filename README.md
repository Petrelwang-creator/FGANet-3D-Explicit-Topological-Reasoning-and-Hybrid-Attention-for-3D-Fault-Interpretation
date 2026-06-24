# 3D Seismic Fault Segmentation 🌍

# FGANet-3D-Explicit-Topological-Reasoning-and-Hybrid-Attention-for-3D-Fault-Interpretation
To overcome the connectivity disruptions and topological distortions of traditional 3D CNNs when processing noisy and highly imbalanced seismic data,here  proposes FGANet3D—a hybrid deep learning framework integrating high-frequency CNN convolutions, hybrid self-attention, and a multi-objective physical constraint joint loss function.



基于增强型 3D U-Net 的地震数据断层分割深度学习框架。本项目针对 3D 地震体数据中“断层极度不平衡”及“细长结构难提取”的痛点，集成了多种注意力机制、前沿损失函数以及高效的训练流。

## ✨ 核心特性

* **增强型模型架构**：采用 3D U-Net 为骨干，可选配 **CBAM (通道与空间注意力)** 及 **Non-local (全局自注意力)** 模块，精准聚焦断层特征。
* **针对不平衡数据的损失函数**：集成 `Focal-Tversky Loss`、`Boundary Loss` (基于距离变换) 与 `BCE Loss`，有效克服断层像素占比极小的训练难点。
* **断层优先数据管道**：自定义 3D 随机裁剪策略（优先采样包含断层目标的 Patch），并内置 3D 翻转、旋转及高斯模糊等数据增强。
* **企业级训练引擎**：支持 **AMP (自动混合精度)**、梯度累加、动态学习率衰减 (ReduceLROnPlateau) 以及早停机制 (Early Stopping)。

## 📁 项目结构

```text
├── Unet3_torch18.py      # 网络架构：3D U-Net + CBAM + Non-local
├── faultseg_torch13.py   # 数据加载器：3D Volume 切割与增强
├── loss_utils13.py       # 损失函数：Focal-Tversky + Boundary + BCE
├── train_torch18.py      # 训练主干：前向传播、AMP 加速、断点续训
├── requirements.txt      # 环境依赖清单
└── README.md             # 项目说明文档

