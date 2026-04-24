# CNN 手写汉字分类 — 代码说明文档

> 对应文件：`models/part2/cnn.py`
> 任务：12 类手写汉字分类（PyTorch 自实现 CNN）

---

## 目录

1. [整体结构概览](#1-整体结构概览)
2. [数据集模块](#2-数据集模块)
3. [数据增强与预处理](#3-数据增强与预处理)
4. [DataLoader 构建](#4-dataloader-构建)
5. [模型结构](#5-模型结构)
6. [训练与评估](#6-训练与评估)
7. [主流程 run()](#7-主流程-run)
8. [超参数说明](#8-超参数说明)
9. [防过拟合策略（Bonus）](#9-防过拟合策略bonus)
10. [输出文件](#10-输出文件)

---

## 1. 整体结构概览

```
cnn.py
├── 数据集
│   ├── ChineseCharDataset          # 自定义 Dataset
│   ├── stratified_split_indices    # 按类别分层划分训练/验证集
│   ├── build_transforms            # 构建数据增强 transform
│   ├── build_dataloaders           # 构建 DataLoader（完整参数版）
│   └── get_dataloaders             # 简化封装入口
├── 模型
│   ├── ConvBlock                   # 基础卷积块（Conv+BN+ReLU+可选Pool）
│   └── SimpleCNN                   # 完整 CNN 模型
├── 训练
│   ├── train_one_epoch             # 单 epoch 训练
│   └── evaluate                   # 验证集评估
└── run()                           # 主入口：超参数设置、训练循环、绘图
```

---

## 2. 数据集模块

### `ChineseCharDataset`（第 22–48 行）

继承 `torch.utils.data.Dataset`，负责从磁盘加载图片。

**目录结构要求：**
```
datasets/train_data/train/
    ├── 1/   ← 第 1 类，内含 *.bmp 图片
    ├── 2/
    ...
    └── 12/
```

**关键实现：**

| 方法 | 作用 |
|------|------|
| `__init__` | 遍历所有类目录，将 `(图片路径, 标签)` 存入 `self.samples` 列表 |
| `__len__` | 返回样本总数，供 DataLoader 使用 |
| `__getitem__` | 按索引读取图片，转为灰度图（`"L"` 模式），应用 transform 后返回 `(img, label)` |

**注意：**
- 类目录按文件夹名（整数）升序排列，保证标签与类别的映射稳定。
- 图片统一转为单通道灰度（`Image.convert("L")`），对应模型输入通道数为 1。

---

### `stratified_split_indices`（第 51–61 行）

**分层划分**训练集和验证集，确保每个类别在验证集中都有代表性样本。

```
对每个类别 c:
    取出该类所有样本索引 → 随机打乱
    前 15% → 验证集
    后 85% → 训练集
```

- `val_ratio=0.15`：验证集占 15%
- `seed=42`：固定随机种子，保证可复现性

---

## 3. 数据增强与预处理

### `build_transforms`（第 64–80 行）

返回两套 transform：训练集（含增强）和验证集（仅标准化）。

**训练集 transform（`augment=True`）：**

```
RandomHorizontalFlip(p=0.5)              # 随机水平翻转
RandomAffine(degrees=10,                 # 随机旋转 ±10°
             translate=(0.1, 0.1),       # 随机平移 ±10%
             scale=(0.9, 1.1))           # 随机缩放 90%~110%
ToTensor()                               # 转为 [0,1] 的 Tensor
Normalize(mean=[0.5], std=[0.5])         # 归一化到 [-1, 1]
```

**验证集 transform：**
```
ToTensor()
Normalize(mean=[0.5], std=[0.5])         # 仅归一化，无随机操作
```

**数据增强的意义：**
- 水平翻转、仿射变换模拟手写时的书写差异
- 增强模型对位置、角度、大小变化的鲁棒性
- 相当于增大了有效训练集规模，缓解过拟合

---

## 4. DataLoader 构建

### `build_dataloaders`（第 83–114 行）

```python
train_ds = Subset(ChineseCharDataset(data_dir, transform=train_tf), train_idx)
val_ds   = Subset(ChineseCharDataset(data_dir, transform=eval_tf),  val_idx)
```

- 使用 `Subset` 对同一数据集按索引切分，**训练集和验证集使用不同的 transform**。
- `pin_memory=True`（有 GPU 时）：将数据预锁定在 CPU 内存，加速 CPU→GPU 传输。
- `shuffle=True`（训练集）：每 epoch 随机打乱顺序，防止模型记住样本顺序。
- `num_workers=4`：4 个子进程并行读取数据，避免 IO 成为训练瓶颈。

---

## 5. 模型结构

### `ConvBlock`（第 129–143 行）

最小复用单元，封装了标准的 **Conv → BN → ReLU → (可选 MaxPool)** 组合。

```python
ConvBlock(in_c, out_c, pool=True)
```

| 层 | 说明 |
|----|------|
| `Conv2d(kernel=3, padding=1)` | 保持空间尺寸不变（same padding），`bias=False` 因为后面有 BN |
| `BatchNorm2d` | 对每个通道做归一化，加速收敛，提升稳定性 |
| `ReLU(inplace=True)` | 非线性激活，`inplace` 节省内存 |
| `MaxPool2d(2,2)`（可选） | 步长为 2 的最大池化，空间尺寸减半，扩大感受野 |

---

### `SimpleCNN`（第 146–188 行）

**输入：** `(B, 1, 28, 28)`  —— Batch × 1通道 × 28×28 像素

**网络结构与特征图尺寸变化：**

```
输入              (B, 1,   28, 28)
ConvBlock(1→32,   pool=True)   → (B, 32,  14, 14)   # 通道×32，尺寸减半
ConvBlock(32→64,  pool=True)   → (B, 64,   7,  7)   # 通道×64，尺寸再减半
ConvBlock(64→128, pool=False)  → (B, 128,  7,  7)   # 通道×128，尺寸不变
AdaptiveAvgPool2d(1)           → (B, 128,  1,  1)   # 全局平均池化
Flatten                        → (B, 128)
Linear(128→256) + ReLU         → (B, 256)
Dropout(0.5)                   → (B, 256)            # 随机丢弃50%神经元
Linear(256→12)                 → (B, 12)             # 12类输出 logits
```

**权重初始化（`_init_weights`）：**

| 层类型 | 初始化方法 | 原因 |
|--------|-----------|------|
| `Conv2d` | Kaiming Normal | 专为 ReLU 设计，防止梯度消失/爆炸 |
| `Linear` | Xavier Normal | 适合线性层，保持前向/反向信号方差 |
| `Linear.bias` | 全零 | 偏置无需特殊初始化 |

**为什么用全局平均池化（GAP）而非 Flatten？**
- 将 `7×7` 特征图直接平均成一个标量，大幅减少参数量
- 相比直接 Flatten（`128×7×7=6272` 维），GAP 输出仅 128 维
- 具有一定空间不变性，泛化能力更强

---

## 6. 训练与评估

### `train_one_epoch`（第 192–205 行）

标准的单 epoch 训练循环：

```
for 每个 batch (X, y):
    1. 数据移至 device（GPU/CPU）
    2. 梯度清零（zero_grad）
    3. 前向传播：out = model(X)
    4. 计算损失：loss = criterion(out, y)
    5. 反向传播：loss.backward()
    6. 参数更新：optimizer.step()
    7. 累计 loss 和正确预测数
返回：平均损失, 准确率
```

**损失累计方式：** `total_loss += loss.item() * len(y)` 用样本数加权，最后除以总样本数，得到真实平均损失（避免最后一个 batch 样本数不足导致的偏差）。

---

### `evaluate`（第 208–219 行）

使用 `@torch.no_grad()` 装饰器，禁用梯度计算，节省内存和计算。

`model.eval()` 的作用：
- 关闭 Dropout（推理时不随机丢弃神经元）
- 让 BatchNorm 使用训练时统计的全局均值/方差，而非当前 batch 统计值

---

## 7. 主流程 `run()`

### 训练配置

```python
BATCH   = 64      # batch 大小
EPOCHS  = 100     # 训练轮数
LR      = 1e-3    # 学习率
WD      = 1e-4    # 权重衰减（L2 正则）
DROPOUT = 0.5     # Dropout 比例
```

### 损失函数

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

**标签平滑（Label Smoothing）：** 将硬标签（one-hot）软化，目标概率从 1.0 变为 0.9，其余 11 类各分配约 0.009。防止模型过于自信，提升泛化能力。

### 优化器

```python
optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
```

- **AdamW**：Adam 的改进版，权重衰减（L2 正则）与梯度更新解耦，正则化效果更纯净。

### 学习率调度

```python
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

余弦退火：学习率从初始值 `LR` 按余弦曲线逐渐降至接近 0。避免后期学习率过大导致在最优解附近震荡。

```
LR
│╲
│  ╲
│    ╲___
│        ╲___
│             ╲____
└──────────────────→ Epoch
0                100
```

### 模型保存

每个 epoch 结束后，若验证准确率创历史最高，保存模型权重：
```python
torch.save(model.state_dict(), RESULTS_DIR / "cnn_best.pth")
```

### 可视化

训练结束后生成两张图：

1. **`cnn_training.png`**：训练/验证的 Loss 和 Accuracy 曲线
2. **`cnn_confusion.png`**：验证集混淆矩阵（12×12），直观展示各类别预测情况

---

## 8. 超参数说明

| 超参数 | 值 | 选择依据 |
|--------|-----|---------|
| `BATCH` | 64 | 显存与泛化性的平衡点 |
| `EPOCHS` | 100 | 配合余弦退火完整衰减一个周期 |
| `LR` | 1e-3 | 消融实验中与 2e-3 并列最优，leaderboard 排名更高 |
| `WD` | 1e-4 | 消融实验默认选择（full 配置） |
| `DROPOUT` | 0.5 | 标准值，防止 FC 层过拟合 |
| `val_ratio` | 0.15 | 训练:验证 ≈ 85:15 |

---

## 9. 防过拟合策略（Bonus）

代码实现了多种正则化手段，共同防止过拟合：

| 策略 | 位置 | 作用 |
|------|------|------|
| **BatchNorm** | 每个 ConvBlock | 归一化激活值，减少内部协变量偏移 |
| **Dropout(0.5)** | FC 层之间 | 随机屏蔽 50% 神经元，防止共适应 |
| **数据增强** | 训练 transform | 翻转+仿射变换，扩充有效样本多样性 |
| **Weight Decay(1e-4)** | AdamW | L2 正则，抑制权重过大 |
| **Label Smoothing(0.1)** | CrossEntropyLoss | 防止模型过度自信 |
| **最优权重保存** | 每个 epoch 比较 val_acc | 训练跑满 100 epoch，但最终使用过程中 val_acc 最高的权重，而非末轮权重 |

---

## 10. 输出文件

训练完成后，所有结果保存至 `models/part2/results/`：

| 文件 | 内容 |
|------|------|
| `cnn_best.pth` | 验证准确率最高时的模型权重 |
| `cnn_training.png` | Loss / Accuracy 训练曲线图 |
| `cnn_confusion.png` | 12×12 混淆矩阵热力图 |
