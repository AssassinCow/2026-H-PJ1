"""
第二部分：CNN 测试集推理脚本

示例：
    python models/part2/infer.py \
        --test-dir datasets/test_data \
        --weights models/part2/results/cnn_best.pth \
        --output models/part2/results/test_predictions.csv

    # 若测试集带真值标签，可额外计算准确率：
    python models/part2/infer.py \
        --test-dir datasets/test_data \
        --labels-csv datasets/test_labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cnn import RESULTS_DIR, SimpleCNN, build_transforms


IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


def load_labels_csv(label_csv: str, rel_paths: list[str]) -> list[int]:
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
    for rel in rel_paths:
        rel = Path(rel).as_posix()
        if rel not in label_map:
            missing.append(rel)
            continue
        labels.append(label_map[rel])

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"标签 csv 缺少 {len(missing)} 张图片的真值，例如: {preview}")

    return labels


def infer_labels_from_dirs(rel_paths: list[str]) -> list[int] | None:
    labels = []
    for rel in rel_paths:
        parts = Path(rel).parts
        if len(parts) < 2 or not parts[0].isdigit():
            return None
        label = int(parts[0])
        if not 1 <= label <= 12:
            return None
        labels.append(label - 1)
    return labels


class TestImageDataset(Dataset):
    """
    递归读取测试目录下的所有图片文件。
    输出：
        image_tensor, relative_path
    """

    def __init__(self, test_dir: str, transform=None):
        self.root = Path(test_dir)
        self.transform = transform
        self.samples = sorted(
            [
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ],
            key=lambda p: str(p.relative_to(self.root)),
        )
        if not self.samples:
            raise FileNotFoundError(f"未在 {self.root} 下找到测试图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, str(path.relative_to(self.root))


@torch.no_grad()
def predict(
    test_dir: str,
    weights: str,
    output: str,
    batch_size: int = 256,
    workers: int = 4,
    label_csv: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = build_transforms(augment=False)

    ds = TestImageDataset(test_dir, transform=eval_tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleCNN(num_classes=12, dropout=0.5).to(device)
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    pred_labels = []
    for images, rel_paths in loader:
        logits = model(images.to(device))
        pred = logits.argmax(dim=1).cpu().tolist()
        for rel_path, label_idx in zip(rel_paths, pred):
            rows.append((Path(rel_path).as_posix(), label_idx + 1))
            pred_labels.append(label_idx)

    true_labels = (
        load_labels_csv(label_csv, [rel for rel, _ in rows])
        if label_csv is not None else
        infer_labels_from_dirs([rel for rel, _ in rows])
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(rows)

    print(f"测试图片数: {len(rows)}")
    if true_labels is not None:
        correct = sum(int(p == y) for p, y in zip(pred_labels, true_labels))
        acc = correct / len(pred_labels)
        print(f"准确率: {acc * 100:.2f}%")
    else:
        print("未提供真值标签，跳过准确率计算")
    print(f"预测结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Part2 CNN 测试集推理脚本")
    parser.add_argument("--test-dir", type=str, required=True, help="测试图片目录")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(RESULTS_DIR / "cnn_best.pth"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "test_predictions.csv"),
        help="预测结果 csv 输出路径",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
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
        batch_size=args.batch_size,
        workers=args.workers,
        label_csv=args.labels_csv,
    )


if __name__ == "__main__":
    main()
