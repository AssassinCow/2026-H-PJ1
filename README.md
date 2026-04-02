# 2026 PJ1 Release

项目当前按“代码 / 数据 / 文档 / 实验产物”四类组织：

```text
2026_pj1_release/
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

1. [PJ1.pdf](/home/lzx/my_workspace/ai_lab/2026_pj1_release/PJ1.pdf)
2. [docs/README.md](/home/lzx/my_workspace/ai_lab/2026_pj1_release/docs/README.md)
3. [docs/lab_report.md](/home/lzx/my_workspace/ai_lab/2026_pj1_release/docs/lab_report.md)
4. [train/run_all.py](/home/lzx/my_workspace/ai_lab/2026_pj1_release/train/run_all.py)
5. [ablation/ablation_part1/run_ablation_part1.py](/home/lzx/my_workspace/ai_lab/2026_pj1_release/ablation/ablation_part1/run_ablation_part1.py)、[ablation/ablation_part2/run_ablation_part2.py](/home/lzx/my_workspace/ai_lab/2026_pj1_release/ablation/ablation_part2/run_ablation_part2.py)
6. [contrast/run_contrast_all.py](/home/lzx/my_workspace/ai_lab/2026_pj1_release/contrast/run_contrast_all.py)

常用入口：

```bash
# 主训练
python train/run_all.py

# 消融实验
python ablation/ablation_part1/run_ablation_part1.py
python ablation/ablation_part2/run_ablation_part2.py

# 参数对比
python contrast/run_contrast_all.py --task all
```

说明：

- `models/*/results/` 保存主训练图表、权重等直接产物
- `train/train_log/` 保存 `train/run_all.py` 自动生成的主训练日志
- `ablation/ablation_*/*results*/` 保存消融结果
- `contrast/contrast_*/*results*/` 保存参数对比结果
- `ablation/logs/` 保存消融脚本自动生成的控制台日志
- `contrast/logs/` 保存对比脚本自动生成的控制台日志
