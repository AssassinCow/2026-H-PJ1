#!/usr/bin/env python3
"""
Part1(BP, numpy) 参数对比实验脚本。

默认对比项（learning rate）：
- lr_5e-3
- lr_1e-2
- lr_2e-2

输出结构与消融实验保持一致：
- 每个 variant: metrics.csv + summary.json + best.npz
- 根目录: leaderboard.csv + summary_all.json

示例：
    python contrast/contrast_part1/run_contrast_part1.py
    python contrast/contrast_part1/run_contrast_part1.py --epochs 150 --batch-size 128
    python contrast/contrast_part1/run_contrast_part1.py --variants lr_1e-2,lr_2e-2
"""
import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART1_DIR = PROJECT_ROOT / "models" / "part1"
LOG_PATH = PROJECT_ROOT / "contrast" / "logs" / "contrast_part1.log"
sys.path.insert(0, str(PART1_DIR))

from nn import NeuralNetwork
from classification import augment_batch, load_dataset, train_val_split, one_hot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee = Tee(old_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def get_variant_bank() -> dict:
    return {
        "lr_5e-3": {"learning_rate": 5e-3},
        "lr_1e-2": {"learning_rate": 1e-2},
        "lr_2e-2": {"learning_rate": 2e-2},
    }


def run_variant(name: str, cfg: dict, args, root_out: Path, X_tr, y_tr, X_va, y_va, num_classes: int) -> dict:
    out_dir = root_out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("\n" + "=" * 72)
    print(f"[Part1 Contrast] {name}")
    print("=" * 72)
    print(f"learning_rate={cfg['learning_rate']} epochs={args.epochs} batch_size={args.batch_size}")

    model = NeuralNetwork(
        layer_sizes=[784, 512, 256, 128, num_classes],
        learning_rate=cfg["learning_rate"],
        activation="relu",
        output_activation="softmax",
        loss="cross_entropy",
        weight_decay=1e-4,
        momentum=args.momentum,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    y_tr_oh = one_hot(y_tr, num_classes)
    best_val_acc = 0.0
    best_epoch = 0
    best_W = [w.copy() for w in model.W]
    best_b = [b.copy() for b in model.b]

    metrics_rows = []
    csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"
    ckpt_path = out_dir / "best.npz"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        Xa = augment_batch(X_tr, img_size=args.img_size)
        perm = np.random.permutation(len(Xa))
        Xa = Xa[perm]
        ya_oh = y_tr_oh[perm]

        total_loss, n_bat = 0.0, 0
        for s in range(0, len(Xa), args.batch_size):
            Xb = Xa[s:s + args.batch_size]
            yb = ya_oh[s:s + args.batch_size]
            yp = model.forward(Xb)
            total_loss += model._loss(yp, yb)
            n_bat += 1
            model.backward(yb)

        train_loss = total_loss / max(1, n_bat)
        train_acc = model.accuracy(X_tr, y_tr)
        val_acc = model.accuracy(X_va, y_va)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_W = [w.copy() for w in model.W]
            best_b = [b.copy() for b in model.b]

        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "lr": float(model.lr),
            "sec": float(time.time() - t0),
        }
        metrics_rows.append(row)

        if model.lr_decay_step > 0 and ep % model.lr_decay_step == 0:
            model.lr *= model.lr_decay_gamma

        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            print(
                f"Ep {ep:3d}/{args.epochs} "
                f"loss={train_loss:.4f} train={train_acc*100:.2f}% "
                f"val={val_acc*100:.2f}% [best={best_val_acc*100:.2f}%@{best_epoch}] "
                f"{row['sec']:.2f}s"
            )

    model.W = best_W
    model.b = best_b
    model.save(str(ckpt_path.with_suffix("")))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_acc", "lr", "sec"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "variant": name,
        "config": cfg,
        "epochs": args.epochs,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_val_acc": float(metrics_rows[-1]["val_acc"]),
        "final_train_acc": float(metrics_rows[-1]["train_acc"]),
        "seed": args.seed,
        "data_dir": relpath(Path(args.data)),
        "outputs": {
            "metrics_csv": relpath(csv_path),
            "summary_json": relpath(summary_path),
            "best_ckpt": relpath(ckpt_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Part1(BP) 参数对比实验脚本")
    default_data = str(PROJECT_ROOT / "datasets" / "train_data")
    default_outdir = str(Path(__file__).resolve().parent / "contrast_results_part1")

    p.add_argument("--data", type=str, default=default_data, help="数据根目录（含 train/）")
    p.add_argument("--img-size", type=int, default=28)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--lr-decay-step", type=int, default=30)
    p.add_argument("--lr-decay-gamma", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--outdir", type=str, default=default_outdir)
    p.add_argument(
        "--variants",
        type=str,
        default="lr_5e-3,lr_1e-2,lr_2e-2",
        help="逗号分隔，例如 lr_1e-2,lr_2e-2",
    )
    return p.parse_args()


def main():
    with tee_output(LOG_PATH):
        args = parse_args()
        if not Path(args.data).is_absolute():
            args.data = str((PROJECT_ROOT / args.data).resolve())
        if not Path(args.outdir).is_absolute():
            args.outdir = str((PROJECT_ROOT / args.outdir).resolve())

        bank = get_variant_bank()
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in selected if v not in bank]
        if unknown:
            raise ValueError(f"未知 variants: {unknown}. 可选: {list(bank.keys())}")

        print(f"Data: {args.data}")
        print(f"Variants: {selected}")

        X, y = load_dataset(args.data, img_size=args.img_size)
        X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_ratio=args.val_ratio, seed=args.split_seed)
        num_classes = int(len(np.unique(y)))
        print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Classes: {num_classes}")

        root_out = Path(args.outdir)
        root_out.mkdir(parents=True, exist_ok=True)

        all_summaries = []
        t_start = time.time()
        for name in selected:
            all_summaries.append(
                run_variant(name, bank[name], args, root_out, X_tr, y_tr, X_va, y_va, num_classes)
            )

        total_sec = time.time() - t_start
        rank = sorted(all_summaries, key=lambda x: x["best_val_acc"], reverse=True)
        summary_all_path = root_out / "summary_all.json"
        with open(summary_all_path, "w", encoding="utf-8") as f:
            json.dump({"total_seconds": total_sec, "variants": rank}, f, ensure_ascii=False, indent=2)

        table_path = root_out / "leaderboard.csv"
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "variant", "best_val_acc", "best_epoch", "final_val_acc"])
            for i, s in enumerate(rank, 1):
                writer.writerow([i, s["variant"], s["best_val_acc"], s["best_epoch"], s["final_val_acc"]])

        print("\n" + "=" * 72)
        print("Part1 contrast finished")
        print(f"Total time: {total_sec/60:.2f} min")
        print(f"Leaderboard: {relpath(table_path)}")
        print(f"All summary: {relpath(summary_all_path)}")


if __name__ == "__main__":
    main()
