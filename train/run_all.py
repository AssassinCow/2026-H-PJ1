"""
一键运行两部分
Usage:
    python train/run_all.py              # 跑全部
    python train/run_all.py --part 1     # 只跑第一部分（BP）
    python train/run_all.py --part 2     # 只跑第二部分（CNN）
"""
import argparse, os, sys
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
LOG_DIR = os.path.join(ROOT, "train_log")


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
def tee_output(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee = Tee(old_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=[1, 2], default=0,
                    help="0=全部, 1=BP, 2=CNN")
parser.add_argument("--data", type=str, default="datasets/train_data",
                    help="训练数据根目录（含 train/ 子目录）")
args = parser.parse_args()

DATA = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)

if args.part in (0, 1):
    with tee_output(os.path.join(LOG_DIR, "train_part1.log")):
        print("\n" + "="*60)
        print("【第一部分 A】回归任务：sin(x)")
        print("="*60)
        part1_dir = os.path.join(PROJECT_ROOT, "models", "part1")
        sys.path.insert(0, part1_dir)
        os.chdir(part1_dir)
        from regression import run as run_reg
        run_reg()

        print("\n" + "="*60)
        print("【第一部分 B】分类任务：手写汉字（BP）")
        print("="*60)
        from classification import run as run_cls
        run_cls(data_dir=DATA)

if args.part in (0, 2):
    with tee_output(os.path.join(LOG_DIR, "train_part2.log")):
        print("\n" + "="*60)
        print("【第二部分】CNN 手写汉字分类")
        print("="*60)
        part2_dir = os.path.join(PROJECT_ROOT, "models", "part2")
        sys.path.insert(0, part2_dir)
        os.chdir(part2_dir)
        from cnn import run as run_cnn
        run_cnn(data_dir=DATA)
