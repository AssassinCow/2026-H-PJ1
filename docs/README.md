# Project 1 - 人工智能 (H) 26春

## 项目结构

```
2026-H-PJ1/
├── datasets/
│   └── train_data/train/{1..12}/*.bmp   # 训练数据（解压后）
├── models/
│   ├── part1/
│   │   ├── nn.py              # 纯 numpy BP 神经网络核心实现
│   │   ├── regression.py      # 回归任务：拟合 sin(x)
│   │   ├── classification.py  # 分类任务：12类汉字（BP）
│   │   └── results/           # part1 主训练产物
│   └── part2/
│       ├── cnn.py             # 卷积神经网络（PyTorch）
│       └── results/           # part2 主训练产物
├── train/
│   ├── run_all.py             # 主训练统一入口
│   └── train_log/             # 主训练日志
├── ablation/
│   ├── ablation_part1/        # part1 消融脚本与结果
│   ├── ablation_part2/        # part2 消融脚本与结果
│   └── logs/                  # 消融日志
├── contrast/
│   ├── contrast_part1/        # part1 对比脚本与结果
│   ├── contrast_part2/        # part2 对比脚本与结果
│   ├── logs/                  # 对比实验日志
│   └── run_contrast_all.py    # 对比实验总入口
├── docs/
│   ├── README.md
│   └── lab_report.md
├── PJ1.pdf                    # 课程项目说明
└── requirements*.txt          # 环境依赖
```

---

## 环境配置

实验使用 Python 3.11 + PyTorch（2.6+）。`requirements-cu128.txt` 提供的是 CUDA 12.8 通道上的 PyTorch wheel，对新架构 GPU 兼容性更好；若使用较旧 CUDA 驱动，可以改用 PyTorch 官方对应 CUDA 版本的索引。

### 1. 创建 conda 虚拟环境

```bash
conda create -n pj1 python=3.11 -y
conda activate pj1
```

### 2. 安装依赖（按设备选择）

```bash
# CPU 环境
pip install -r requirements-cpu.txt

# GPU 环境（CUDA 12.8 通道的 PyTorch）
pip install -r requirements-cu128.txt
```

### 3. 验证 GPU 可用

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 数据准备

在项目根目录（即 `2026-H-PJ1/`）下执行：

```bash
mkdir -p datasets
unzip train_data.zip -d datasets
mv datasets/train_data_update_03-31_v2 datasets/train_data
```

---

## 运行命令

以下命令均在项目根目录下执行：

```bash
# 推荐实验顺序：先消融确定组件，再做参数对比，最后跑主训练
python ablation/ablation_part1/run_ablation_part1.py
python contrast/contrast_part1/run_contrast_part1.py
python ablation/ablation_part2/run_ablation_part2.py
python contrast/contrast_part2/run_contrast_part2.py
python train/run_all.py

# 也可使用总入口统一跑参数对比
python contrast/run_contrast_all.py --task all

# 如果只想直接运行主训练
python train/run_all.py --part 1
python train/run_all.py --part 2
python train/run_all.py

# 也可单独运行各脚本
python models/part1/regression.py
python models/part1/classification.py
python models/part2/cnn.py

# 单独运行参数对比
python contrast/contrast_part1/run_contrast_part1.py
python contrast/contrast_part2/run_contrast_part2.py
```

说明：

- 本仓库采用“先消融、后调参、再最终训练”的实验组织逻辑；主训练脚本中的默认配置来自前述实验结论
- `ablation/*` 与 `contrast/*` 都采用同构输出：每个 variant 子目录下含 `metrics.csv` 与 `summary.json`，根目录含 `leaderboard.csv` 与 `summary_all.json`
- 运行 `train/run_all.py` 时，会自动生成 `train/train_log/train_part1.log`、`train/train_log/train_part2.log`
- 运行 `ablation/ablation_part1/run_ablation_part1.py`、`ablation/ablation_part2/run_ablation_part2.py` 时，会自动生成 `ablation/logs/ablation_part1.log`、`ablation/logs/ablation_part2.log`
- 运行 `contrast/run_contrast_all.py` 或对应 part 脚本时，会自动生成 `contrast/logs/contrast_part1.log`、`contrast/logs/contrast_part2.log`

结果（训练曲线、混淆矩阵、模型权重）保存在各目录的 `results/` 文件夹下。

---

## 关键设计说明

### 第一部分：反向传播神经网络（`models/part1/`）

**不使用任何深度学习框架，仅依赖 numpy。**

#### 回归任务（`regression.py`）

- 目标：拟合 $y = \sin(x)$，$x \in [-\pi, \pi]$，平均误差 $< 0.01$
- 数据：在区间内随机采样 2000 个训练点、500 个测试点
- 网络结构：`[1, 128, 64, 64, 1]`
- 激活函数：`tanh`（对周期函数拟合效果优于 ReLU）
- 损失函数：MSE
- 输入预处理：$x / \pi$ 归一化到 $[-1, 1]$

#### 分类任务（`classification.py`）

- 目标：12 类手写汉字分类
- 网络结构：`[784, 512, 256, 128, 12]`
- 激活函数：LeakyReLU（隐藏层）+ Softmax（输出层）
- 损失函数：Cross-Entropy
- 正则化：L2 weight decay
- 学习率调度：每 30 轮 × 0.7
- 数据增强：默认仅做 ±2px 随机平移（按最新 BP 消融结果关闭主模型水平翻转）

#### BP 网络超参数（`nn.py`）

| 超参数 | 说明 |
|---|---|
| `layer_sizes` | 各层神经元数，如 `[784, 256, 128, 12]` |
| `learning_rate` | 初始学习率 |
| `activation` | 隐藏层激活：`relu` / `leaky_relu` / `tanh` / `sigmoid` |
| `output_activation` | 输出层激活：`linear` / `softmax` |
| `loss` | 损失函数：`mse` / `cross_entropy` |
| `weight_decay` | L2 正则化系数 |
| `lr_decay_step` | 学习率衰减间隔（轮数，0 表示不衰减） |
| `lr_decay_gamma` | 每次衰减比例，如 `0.5` 表示减半 |

---

### 第二部分：卷积神经网络（`models/part2/cnn.py`）

使用 PyTorch 自行实现 CNN，不调用预训练模型。

#### 网络结构（`SimpleCNN`）

```
输入: (B, 1, 28, 28)
Conv(1→32, 3×3) + BN + ReLU + MaxPool(2×2)  → (B, 32, 14, 14)
Conv(32→64, 3×3) + BN + ReLU + MaxPool(2×2) → (B, 64,  7,  7)
Conv(64→128, 3×3) + BN + ReLU               → (B, 128, 7,  7)
GlobalAvgPool                                 → (B, 128)
FC(128→256) + ReLU + Dropout(0.5)
FC(256→12)
```

#### 训练策略

| 项目 | 设置 |
|---|---|
| 优化器 | AdamW |
| 学习率调度 | Cosine Annealing |
| 正则化 | Dropout(0.5) + BatchNorm（Bonus：防过拟合）；默认 `weight_decay=1e-4` |
| 损失函数 | Cross-Entropy + 标签平滑（label_smoothing=0.1） |
| 数据增强 | 随机水平翻转 + 随机仿射变换（旋转±10°、平移±10%、缩放0.9~1.1） |
| 训练轮数 | 100 epochs |
