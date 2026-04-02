"""
第二部分：卷积神经网络（PyTorch）
12 类手写汉字分类
包含：自实现 CNN、数据增强、Dropout/BN 防过拟合（Bonus）
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "datasets" / "train_data")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ================================================================== 数据集
class ChineseCharDataset(Dataset):
    """
    手写汉字数据集
    目录结构：data_dir/train/<class_id>/<img>.bmp
    """
    def __init__(self, data_dir: str, transform=None):
        self.samples = []
        self.transform = transform
        train_dir = Path(data_dir) / "train"
        class_dirs = sorted(
            [d for d in train_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name),
        )
        for label, cdir in enumerate(class_dirs):
            for p in sorted(cdir.glob("*.bmp")):
                self.samples.append((str(p), label))
        print(f"[Dataset] {len(self.samples)} 张图片，{len(class_dirs)} 类")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")   # 灰度图
        if self.transform:
            img = self.transform(img)
        return img, label


def stratified_split_indices(labels: list[int], val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    labels_np = np.asarray(labels, dtype=np.int64)
    for c in np.unique(labels_np):
        idx = np.where(labels_np == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return train_idx, val_idx


def build_transforms(augment: bool = True):
    train_ops = []
    if augment:
        train_ops.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    train_ops.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_tf = T.Compose(train_ops)
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, eval_tf


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    augment: bool = True,
    workers: int = 4,
    seed: int = 42,
):
    train_tf, eval_tf = build_transforms(augment=augment)
    full_ds = ChineseCharDataset(data_dir, transform=None)
    labels = [label for _, label in full_ds.samples]
    train_idx, val_idx = stratified_split_indices(labels, val_ratio=val_ratio, seed=seed)

    train_ds = Subset(ChineseCharDataset(data_dir, transform=train_tf), train_idx)
    val_ds = Subset(ChineseCharDataset(data_dir, transform=eval_tf), val_idx)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_dataloaders(data_dir: str, batch_size: int = 64, val_ratio: float = 0.15):
    return build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        augment=True,
        workers=4,
        seed=42,
    )


# ================================================================== 模型
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> (可选 MaxPool)"""
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    自实现 CNN（不调用预训练模型）
    输入：(B, 1, 28, 28)
    输出：(B, 12)

    结构：
        Conv(1->32, 3x3) + BN + ReLU + Pool(2x2) → 14x14
        Conv(32->64, 3x3) + BN + ReLU + Pool(2x2) → 7x7
        Conv(64->128, 3x3) + BN + ReLU             → 7x7
        GAP → 128
        FC(128->256) + ReLU + Dropout(0.5)
        FC(256->12)
    """
    def __init__(self, num_classes: int = 12, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32, pool=True),   # → (B,32,14,14)
            ConvBlock(32,  64, pool=True),   # → (B,64, 7, 7)
            ConvBlock(64, 128, pool=False),  # → (B,128,7, 7)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B,128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ================================================================== 训练
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def run(data_dir: str = DEFAULT_DATA_DIR):
    print("=" * 55)
    print("第二部分：CNN 手写汉字分类（PyTorch）")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 超参数 ----
    BATCH     = 64
    EPOCHS    = 100
    LR        = 1e-3      # 最新对比结果: 与 2e-3 并列最优，leaderboard 排名更高
    WD        = 1e-4      # 最新消融默认选择 full
    DROPOUT   = 0.5       # Bonus：防过拟合

    train_loader, val_loader = get_dataloaders(data_dir, batch_size=BATCH)

    model     = SimpleCNN(num_classes=12, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0.0
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = RESULTS_DIR / "cnn_best.pth"

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), best_path)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:3d}/{EPOCHS}  "
                  f"train_loss={tl:.4f}  train_acc={ta*100:.1f}%  "
                  f"val_loss={vl:.4f}  val_acc={va*100:.1f}%  "
                  f"[best={best_val_acc*100:.1f}%]  {time.time()-t0:.1f}s")

    print(f"\n最优验证准确率: {best_val_acc*100:.2f}%")
    print(f"模型 → {best_path}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"],   label="val")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot([a*100 for a in history["train_acc"]], label="train")
    ax.plot([a*100 for a in history["val_acc"]],   label="val")
    ax.set_title("Accuracy (%)"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    train_fig_path = RESULTS_DIR / "cnn_training.png"
    plt.savefig(train_fig_path, dpi=150)
    plt.close()
    print(f"图表 → {train_fig_path}")

    # ---- 混淆矩阵 ----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in val_loader:
            all_pred.extend(model(X.to(device)).argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    cm = np.zeros((12, 12), int)
    for t, p in zip(all_true, all_pred): cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ticks = range(12)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(i+1) for i in ticks])
    ax.set_yticklabels([str(i+1) for i in ticks])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"CNN Confusion Matrix  val_acc={best_val_acc*100:.1f}%")
    for i in range(12):
        for j in range(12):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()*0.5 else "black", fontsize=8)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "cnn_confusion.png"
    plt.savefig(cm_path, dpi=150)
    print(f"混淆矩阵 → {cm_path}")

    return model, best_val_acc


if __name__ == "__main__":
    run()
