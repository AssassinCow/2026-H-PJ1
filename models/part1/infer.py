"""
第一部分：BP 分类测试集推理脚本

示例：
    python models/part1/infer.py \
        --test-dir datasets/test_data \
        --weights models/part1/results/bp_cls_model.npz \
        --output models/part1/results/test_predictions.csv

    # 若测试集带真值标签，可额外计算准确率：
    python models/part1/infer.py \
        --test-dir datasets/test_data \
        --labels-csv datasets/test_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn import NeuralNetwork


IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def collect_images(test_dir: str) -> list[Path]:
    root = Path(test_dir)
    if not root.exists():
        raise FileNotFoundError(f"测试目录不存在: {root}")
    files = sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: str(p.relative_to(root)),
    )
    if not files:
        raise FileNotFoundError(f"未在 {root} 下找到测试图片")
    return files


def load_images(files: list[Path]) -> np.ndarray:
    arrs = []
    for p in files:
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        arrs.append(arr.flatten())
    return np.stack(arrs).astype(np.float32)


def load_labels_csv(label_csv: str, files: list[Path], root: Path) -> np.ndarray:
    label_path = Path(label_csv)
    if not label_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    with label_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise KeyError("标签 csv 必须包含 image,label 两列")

        label_map: dict[str, int] = {}
        for row in reader:
            rel = Path(row["image"]).as_posix()
            if rel in label_map:
                raise ValueError(f"标签 csv 中存在重复图片项: {rel}")
            label = int(row["label"])
            if not 1 <= label <= 12:
                raise ValueError(f"标签超出范围 [1,12]: {label}")
            label_map[rel] = label - 1

    labels = []
    missing = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel not in label_map:
            missing.append(rel)
            continue
        labels.append(label_map[rel])

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"标签 csv 缺少 {len(missing)} 张图片的真值，例如: {preview}")

    return np.asarray(labels, dtype=np.int64)


def infer_labels_from_dirs(files: list[Path], root: Path) -> np.ndarray | None:
    labels = []
    for p in files:
        parts = p.relative_to(root).parts
        if len(parts) < 2 or not parts[0].isdigit():
            return None
        label = int(parts[0])
        if not 1 <= label <= 12:
            return None
        labels.append(label - 1)
    return np.asarray(labels, dtype=np.int64)


def resolve_true_labels(label_csv: str | None, files: list[Path], root: Path) -> np.ndarray | None:
    if label_csv is not None:
        return load_labels_csv(label_csv, files, root)
    return infer_labels_from_dirs(files, root)


def predict(test_dir: str, weights: str, output: str, label_csv: str | None = None) -> None:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    # 从权重文件读取 layer_sizes，确保网络结构与训练一致
    saved = np.load(weights_path, allow_pickle=True)
    if "layer_sizes" not in saved:
        raise KeyError(f"权重文件缺少 layer_sizes: {weights_path}")
    layer_sizes = [int(x) for x in saved["layer_sizes"].tolist()]

    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation="leaky_relu",
        output_activation="softmax",
        loss="cross_entropy",
    )
    model.load(str(weights_path))

    root = Path(test_dir)
    files = collect_images(test_dir)
    X = load_images(files)
    pred = model.predict_class(X)  # 0-indexed
    true_labels = resolve_true_labels(label_csv, files, root)

    rows = []
    for p, label_idx in zip(files, pred):
        rel = p.relative_to(root).as_posix()
        rows.append((rel, int(label_idx) + 1))  # 与训练目录 1..12 对齐

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(rows)

    print(f"测试图片数: {len(rows)}")
    if true_labels is not None:
        acc = float((pred == true_labels).mean())
        print(f"准确率: {acc * 100:.2f}%")
    else:
        print("未提供真值标签，跳过准确率计算")
    print(f"预测结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Part1 BP 测试集推理脚本")
    parser.add_argument("--test-dir", type=str, required=True, help="测试图片目录")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(RESULTS_DIR / "bp_cls_model.npz"),
        help="模型权重路径 (.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "test_predictions.csv"),
        help="预测结果 csv 输出路径",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="可选真值标签 csv，需包含 image,label 两列；若不提供则尝试从测试目录一级子目录名解析类别",
    )
    args = parser.parse_args()

    predict(
        test_dir=args.test_dir,
        weights=args.weights,
        output=args.output,
        label_csv=args.labels_csv,
    )


if __name__ == "__main__":
    main()
