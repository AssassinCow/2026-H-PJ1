# 2026-H-PJ1

项目当前按“代码 / 数据 / 文档 / 实验产物”四类组织：

```text
2026-H-PJ1/
├── models/                    # 训练代码
│   ├── part1/                 # numpy BP：回归 + 分类
│   └── part2/                 # PyTorch CNN
├── train/                     # 主训练统一入口与训练日志
│   ├── run_all.py
│   └── train_log/
├── ablation/                  # 消融脚本、结果、日志
│   ├── ablation_part1/
│   ├── ablation_part2/
│   └── logs/
├── contrast/                  # 参数对比脚本与结果
│   ├── contrast_part1/
│   ├── contrast_part2/
│   ├── logs/
│   └── run_contrast_all.py
├── datasets/                  # 原始训练数据
├── docs/                      # 说明文档与实验报告
├── PJ1.pdf                    # 课程项目说明
└── requirements*.txt          # 环境依赖
```

建议阅读顺序：

1. [PJ1.pdf](PJ1.pdf)
2. [docs/README.md](docs/README.md)
3. [docs/lab_report.md](docs/lab_report.md)
4. [train/run_all.py](train/run_all.py)
5. [ablation/ablation_part1/run_ablation_part1.py](ablation/ablation_part1/run_ablation_part1.py)、[ablation/ablation_part2/run_ablation_part2.py](ablation/ablation_part2/run_ablation_part2.py)
6. [contrast/run_contrast_all.py](contrast/run_contrast_all.py)

常用入口：

```bash
# 推荐实验顺序：先消融，再参数对比，最后主训练
python ablation/ablation_part1/run_ablation_part1.py
python contrast/contrast_part1/run_contrast_part1.py
python ablation/ablation_part2/run_ablation_part2.py
python contrast/contrast_part2/run_contrast_part2.py

# 最后再跑主训练
python train/run_all.py

# 也可使用总入口串行跑完参数对比
python contrast/run_contrast_all.py --task all

# 如果只想直接跑主训练
python train/run_all.py
```

数据准备（当前仓库）：

```bash
mkdir -p datasets
unzip train_data.zip -d datasets
mv datasets/train_data_update_03-31_v2 datasets/train_data
```

`train_data.zip` 内部自带顶层目录 `train_data_update_03-31_v2/`，整理后目录应为：

```text
datasets/train_data/train/1/
datasets/train_data/train/2/
...
datasets/train_data/train/12/
```

说明：

- `models/*/results/` 保存主训练图表、权重等直接产物
- `train/train_log/` 保存 `train/run_all.py` 自动生成的主训练日志
- `ablation/ablation_*/*results*/` 保存消融结果
- `contrast/contrast_*/*results*/` 保存参数对比结果
- `ablation/logs/` 保存消融脚本自动生成的控制台日志
- `contrast/logs/` 保存对比脚本自动生成的控制台日志
