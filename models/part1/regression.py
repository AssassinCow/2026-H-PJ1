"""
回归任务：拟合 y = sin(x)，x ∈ [-π, π]，要求平均误差 < 0.01
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn import NeuralNetwork

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate_data(n_train=2000, n_test=500, seed=42):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-np.pi, np.pi, (n_train, 1))
    X_test  = rng.uniform(-np.pi, np.pi, (n_test,  1))
    return X_train, np.sin(X_train), X_test, np.sin(X_test)


def run():
    print("=" * 55)
    print("回归任务：拟合 y = sin(x)，x ∈ [-π, π]")
    print("=" * 55)
    np.random.seed(0)

    X_tr, y_tr, X_te, y_te = generate_data()

    # 输入归一化到 [-1, 1]
    X_tr_n = X_tr / np.pi
    X_te_n = X_te / np.pi

    # 网络：1 -> 128 -> 64 -> 64 -> 1，tanh 对周期函数效果好
    model = NeuralNetwork(
        layer_sizes    = [1, 128, 64, 64, 1],
        learning_rate  = 0.02,
        activation     = "tanh",
        output_activation = "linear",
        loss           = "mse",
        weight_decay   = 1e-5,
        lr_decay_step  = 1000,
        lr_decay_gamma = 0.5,
    )
    print(f"结构: {model.layer_sizes}  激活: {model.activation}")

    history = model.train(
        X_tr_n, y_tr,
        epochs=3000, batch_size=64,
        verbose=True, print_every=500,
        X_val=X_te_n, y_val=y_te,
    )

    y_pred = model.predict(X_te_n)
    mae = float(np.mean(np.abs(y_pred - y_te)))
    mse = float(np.mean((y_pred - y_te) ** 2))
    print(f"\n测试 MAE = {mae:.6f}  ({'达标 ✓' if mae < 0.01 else '未达标 ✗'})")
    print(f"测试 MSE = {mse:.6f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 拟合曲线 ----
    x_vis = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
    y_vis = model.predict(x_vis / np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(x_vis, np.sin(x_vis), "b-", lw=2, label="true: sin(x)")
    ax.plot(x_vis, y_vis, "r--", lw=2, label="prediction")
    ax.scatter(X_te.flatten(), y_te.flatten(), s=4, c="gray", alpha=0.3, label="test pts")
    ax.set_title(f"sin(x) Regression  MAE={mae:.5f}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.semilogy(history["train_loss"], label="train")
    ax.semilogy(history["val_loss"],   label="val")
    ax.set_title("Loss Curve"); ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log)"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    out_path = RESULTS_DIR / "regression.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图表 → {out_path}")

    return model, mae


if __name__ == "__main__":
    run()
