#!/usr/bin/env python3
"""
完整消融实验一键脚本（默认全轮数跑完）

示例:
    python ablation/ablation_part2/run_ablation_part2.py
    python ablation/ablation_part2/run_ablation_part2.py --epochs 100 --batch-size 64
    python ablation/ablation_part2/run_ablation_part2.py --variants full,no_aug,no_dropout
"""
import argparse
import csv
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART2_DIR = PROJECT_ROOT / "models" / "part2"
LOG_PATH = PROJECT_ROOT / "ablation" / "logs" / "ablation_part2.log"
sys.path.insert(0, str(PART2_DIR))

# 复用已有实现，避免重复造轮子
from cnn import SimpleCNN, build_dataloaders, train_one_epoch, evaluate


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_variant(name: str, cfg: dict, args, root_out: Path, device: torch.device) -> dict:
    out_dir = root_out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("\n" + "=" * 72)
    print(f"[Ablation] {name}")
    print("=" * 72)
    print(
        f"augment={cfg['augment']} dropout={cfg['dropout']} "
        f"label_smoothing={cfg['label_smoothing']} weight_decay={cfg['weight_decay']} "
        f"scheduler={cfg['scheduler']} epochs={args.epochs}"
    )

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        augment=cfg["augment"],
        workers=args.workers,
        seed=args.seed,
    )

    model = SimpleCNN(num_classes=12, dropout=cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if cfg["scheduler"] else None
    )

    best_val_acc = 0.0
    best_epoch = 0
    metrics_rows = []

    ckpt_path = out_dir / "best.pth"
    csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), ckpt_path)

        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        metrics_rows.append(row)

        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            print(
                f"Ep {ep:3d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% "
                f"[best={best_val_acc*100:.1f}%@{best_epoch}] {row['sec']:.1f}s"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "sec"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "variant": name,
        "config": cfg,
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": metrics_rows[-1]["val_acc"],
        "final_train_acc": metrics_rows[-1]["train_acc"],
        "device": str(device),
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


def get_variant_bank() -> dict:
    return {
        # 全特性版本
        "full": {
            "augment": True,
            "dropout": 0.5,
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "scheduler": True,
        },
        # 各组件消融
        "no_aug": {
            "augment": False,
            "dropout": 0.5,
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "scheduler": True,
        },
        "no_dropout": {
            "augment": True,
            "dropout": 0.0,
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "scheduler": True,
        },
        "no_label_smoothing": {
            "augment": True,
            "dropout": 0.5,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "scheduler": True,
        },
        "no_weight_decay": {
            "augment": True,
            "dropout": 0.5,
            "label_smoothing": 0.1,
            "weight_decay": 0.0,
            "scheduler": True,
        },
        "no_scheduler": {
            "augment": True,
            "dropout": 0.5,
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "scheduler": False,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="完整消融实验脚本（默认全轮数）")
    default_data = str(PROJECT_ROOT / "datasets" / "train_data")
    default_outdir = str(Path(__file__).resolve().parent / "ablation_results_part2")
    p.add_argument("--data", type=str, default=default_data, help="数据根目录（含 train/）")
    p.add_argument("--epochs", type=int, default=100, help="每组实验训练轮数（完整跑完）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--outdir", type=str, default=default_outdir)
    p.add_argument(
        "--variants",
        type=str,
        default="full,no_aug,no_dropout,no_label_smoothing,no_weight_decay,no_scheduler",
        help="逗号分隔，例如 full,no_aug,no_dropout",
    )
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
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

        device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Device: {device}")
        print(f"Data: {args.data}")
        print(f"Variants: {selected}")

        root_out = Path(args.outdir)
        root_out.mkdir(parents=True, exist_ok=True)

        all_summaries = []
        t_start = time.time()
        for name in selected:
            summary = run_variant(name, bank[name], args, root_out, device)
            all_summaries.append(summary)

        total_sec = time.time() - t_start
        rank = sorted(all_summaries, key=lambda x: x["best_val_acc"], reverse=True)
        summary_all_path = root_out / "summary_all.json"
        with open(summary_all_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_seconds": total_sec,
                    "variants": rank,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        table_path = root_out / "leaderboard.csv"
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "variant", "best_val_acc", "best_epoch", "final_val_acc"])
            for i, s in enumerate(rank, 1):
                writer.writerow([i, s["variant"], s["best_val_acc"], s["best_epoch"], s["final_val_acc"]])

        print("\n" + "=" * 72)
        print("Ablation finished")
        print(f"Total time: {total_sec/60:.1f} min")
        print(f"Leaderboard: {relpath(table_path)}")
        print(f"All summary: {relpath(summary_all_path)}")


if __name__ == "__main__":
    main()
