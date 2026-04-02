#!/usr/bin/env python3
"""
对比实验总入口：按需串行运行 Part1/Part2 对比实验。

示例：
    python contrast/run_contrast_all.py --task all
    python contrast/run_contrast_all.py --task part1 --epochs-part1 150
    python contrast/run_contrast_all.py --task part2 --epochs-part2 100 --cpu
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "contrast" / "logs"


def run_cmd(cmd: list[str], log_path: Path) -> None:
    print("[Exec]", " ".join(cmd))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def main():
    p = argparse.ArgumentParser(description="对比实验总入口")
    p.add_argument("--task", choices=["part1", "part2", "all"], default="all")

    p.add_argument("--data", type=str, default="datasets/train_data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--epochs-part1", type=int, default=150)
    p.add_argument("--batch-size-part1", type=int, default=128)
    p.add_argument("--part1-variants", type=str, default="lr_5e-3,lr_1e-2,lr_2e-2")

    p.add_argument("--epochs-part2", type=int, default=100)
    p.add_argument("--batch-size-part2", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--part2-variants", type=str, default="lr_5e-4,lr_1e-3,lr_2e-3")

    args = p.parse_args()

    py = sys.executable

    if args.task in {"part1", "all"}:
        cmd1 = [
            py,
            "contrast/contrast_part1/run_contrast_part1.py",
            "--data", args.data,
            "--epochs", str(args.epochs_part1),
            "--batch-size", str(args.batch_size_part1),
            "--seed", str(args.seed),
            "--log-every", str(args.log_every),
            "--variants", args.part1_variants,
        ]
        run_cmd(cmd1, LOG_DIR / "contrast_part1.log")

    if args.task in {"part2", "all"}:
        cmd2 = [
            py,
            "contrast/contrast_part2/run_contrast_part2.py",
            "--data", args.data,
            "--epochs", str(args.epochs_part2),
            "--batch-size", str(args.batch_size_part2),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--log-every", str(args.log_every),
            "--variants", args.part2_variants,
        ]
        if args.cpu:
            cmd2.append("--cpu")
        run_cmd(cmd2, LOG_DIR / "contrast_part2.log")


if __name__ == "__main__":
    main()
