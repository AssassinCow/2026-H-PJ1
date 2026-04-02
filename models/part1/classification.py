"""
分类任务：12 类手写汉字识别（纯 numpy 反向传播，不使用 PyTorch）
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn import NeuralNetwork

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "datasets" / "train_data")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ------------------------------------------------------------------ 数据加载
def load_dataset(data_dir: str, img_size: int = 28):
    """
    目录结构：data_dir/train/<class_id>/<img>.bmp
    返回 X: (N, 784) float32，y: (N,) int
    """
    train_dir = Path(data_dir) / "train"
    class_dirs = sorted(
        [d for d in train_dir.iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )
    print(f"类别数: {len(class_dirs)}")
    X_list, y_list = [], []
    for label, cdir in enumerate(class_dirs):
        for p in sorted(cdir.glob("*.bmp")):
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            X_list.append(arr.flatten())
            y_list.append(label)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"数据: X={X.shape}  y={y.shape}")
    return X, y


def train_val_split(X, y, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nv = max(1, int(len(idx) * val_ratio))
        va_idx.extend(idx[:nv]); tr_idx.extend(idx[nv:])
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def one_hot(y, n):
    oh = np.zeros((len(y), n), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


# ------------------------------------------------------------------ 数据增强
def augment_batch(X: np.ndarray, img_size: int = 28) -> np.ndarray:
    """最新消融默认: 不翻转，仅做 ±2 像素平移"""
    imgs = X.reshape(-1, img_size, img_size)
    out  = []
    for img in imgs:
        dy, dx = np.random.randint(-2, 3), np.random.randint(-2, 3)
        img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        out.append(img.flatten())
    return np.array(out, dtype=np.float32)


# ------------------------------------------------------------------ 主流程
def run(data_dir: str = DEFAULT_DATA_DIR):
    print("=" * 55)
    print("分类任务：12 类手写汉字（BP 神经网络）")
    print("=" * 55)
    np.random.seed(42)

    X, y = load_dataset(data_dir)
    num_classes = 12
    X_tr, y_tr, X_va, y_va = train_val_split(X, y)
    print(f"训练: {len(X_tr)}  验证: {len(X_va)}")

    y_tr_oh = one_hot(y_tr, num_classes)

    # 结构：784 -> 512 -> 256 -> 128 -> 12
    model = NeuralNetwork(
        layer_sizes       = [784, 512, 256, 128, num_classes],
        learning_rate     = 0.01,
        activation        = "leaky_relu",  # 最新消融默认选择 bp_leaky_relu
        output_activation = "softmax",
        loss              = "cross_entropy",
        weight_decay      = 1e-4,
        momentum          = 0.9,
        max_grad_norm     = 5.0,
        lr_decay_step     = 30,
        lr_decay_gamma    = 0.7,
    )
    print(f"结构: {model.layer_sizes}  激活: {model.activation}")

    epochs, batch_size = 150, 128
    best_val_acc = 0.0
    best_W = [w.copy() for w in model.W]   # 用初始权重兜底，避免 None
    best_b = [b.copy() for b in model.b]
    tr_losses, tr_accs, va_accs = [], [], []

    for ep in range(1, epochs + 1):
        # 增强 + 打乱
        Xa = augment_batch(X_tr)
        perm = np.random.permutation(len(Xa))
        Xa, ya_oh = Xa[perm], y_tr_oh[perm]

        tot_loss, n_bat = 0.0, 0
        for s in range(0, len(Xa), batch_size):
            Xb = Xa[s:s+batch_size]; yb = ya_oh[s:s+batch_size]
            yp = model.forward(Xb)
            tot_loss += model._loss(yp, yb); n_bat += 1
            model.backward(yb)

        tl = tot_loss / n_bat
        ta = model.accuracy(X_tr, y_tr)
        va = model.accuracy(X_va, y_va)
        tr_losses.append(tl); tr_accs.append(ta); va_accs.append(va)

        if va > best_val_acc:
            best_val_acc = va
            best_W = [w.copy() for w in model.W]
            best_b = [b.copy() for b in model.b]

        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{epochs}  loss={tl:.4f}  "
                  f"train={ta*100:.1f}%  val={va*100:.1f}%  "
                  f"[best={best_val_acc*100:.1f}%]")

    # 恢复最优权重
    model.W = best_W; model.b = best_b
    print(f"\n最终验证准确率: {best_val_acc*100:.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(RESULTS_DIR / "bp_cls_model"))

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(tr_losses); axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True)

    axes[1].plot([a*100 for a in tr_accs], label="train")
    axes[1].plot([a*100 for a in va_accs], label="val")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    curve_path = RESULTS_DIR / "bp_classification.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"图表 → {curve_path}")

    # 混淆矩阵
    y_pred = model.predict_class(X_va)
    cm = np.zeros((num_classes, num_classes), int)
    for t, p in zip(y_va, y_pred): cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ticks = range(num_classes)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(i+1) for i in ticks])
    ax.set_yticklabels([str(i+1) for i in ticks])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (val)")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()*0.5 else "black", fontsize=8)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "bp_confusion.png"
    plt.savefig(cm_path, dpi=150)
    print(f"混淆矩阵 → {cm_path}")

    return model, best_val_acc


if __name__ == "__main__":
    run()
