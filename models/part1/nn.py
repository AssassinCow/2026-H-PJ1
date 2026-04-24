"""
反向传播神经网络
仅依赖 numpy，不使用任何深度学习框架
"""
import numpy as np
from typing import List, Optional


# ---------- 激活函数 ----------
def relu(x):         return np.maximum(0.0, x)
def relu_d(x):       return (x > 0.0).astype(x.dtype)
def leaky_relu(x):   return np.where(x > 0, x, 0.01 * x)
def leaky_relu_d(x): return np.where(x > 0, 1.0, 0.01)
def tanh_(x):        return np.tanh(x)
def tanh_d(x):       return 1.0 - np.tanh(x) ** 2
def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))
def sigmoid_d(x):
    s = sigmoid(x); return s * (1.0 - s)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    "relu":       (relu,       relu_d),
    "leaky_relu": (leaky_relu, leaky_relu_d),
    "tanh":       (tanh_,      tanh_d),
    "sigmoid":    (sigmoid,    sigmoid_d),
}


class NeuralNetwork:
    """
    全连接神经网络，支持灵活配置层数、神经元数、激活函数、学习率等超参数。

    Parameters
    ----------
    layer_sizes       : 各层维度，如 [1,64,64,1] 或 [784,256,128,12]
    learning_rate     : 初始学习率
    activation        : 隐藏层激活 ('relu'|'leaky_relu'|'tanh'|'sigmoid')
    output_activation : 输出层激活 ('linear'|'softmax')
    loss              : 损失函数 ('mse'|'cross_entropy')
    weight_decay      : L2 正则化系数
    momentum          : SGD 动量系数（0 = 普通 SGD）
    max_grad_norm     : 全局梯度范数裁剪阈值（0 = 不裁剪）
    lr_decay_step     : 学习率衰减间隔轮数（0 = 不衰减）
    lr_decay_gamma    : 每次衰减比例
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.001,
        activation: str = "relu",
        output_activation: str = "linear",
        loss: str = "mse",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        max_grad_norm: float = 5.0,
        lr_decay_step: int = 0,
        lr_decay_gamma: float = 0.5,
    ):
        assert activation in ACTIVATIONS, f"未知激活函数: {activation}"
        self.layer_sizes     = layer_sizes
        self.lr              = learning_rate
        self.init_lr         = learning_rate
        self.output_act_name = output_activation
        self.loss_name       = loss
        self.weight_decay    = weight_decay
        self.momentum        = momentum
        self.max_grad_norm   = max_grad_norm
        self.lr_decay_step   = lr_decay_step
        self.lr_decay_gamma  = lr_decay_gamma
        self.num_layers      = len(layer_sizes) - 1

        self.activation = activation
        self._act_fn, self._act_d = ACTIVATIONS[activation]

        # 权重初始化
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in) if activation in ("relu", "leaky_relu") \
                    else np.sqrt(1.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.b.append(np.zeros((1, layer_sizes[i+1])))

        # 动量缓冲
        self.vW: List[np.ndarray] = [np.zeros_like(w) for w in self.W]
        self.vb: List[np.ndarray] = [np.zeros_like(b) for b in self.b]

    # ------------------------------------------------------------------
    def _out_act(self, z):
        return softmax(z) if self.output_act_name == "softmax" else z

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._A = [X]
        self._Z = []          # 只存隐藏层 pre-activation
        a = X
        for i in range(self.num_layers):
            z = a @ self.W[i] + self.b[i]
            if i < self.num_layers - 1:
                self._Z.append(z)
                a = self._act_fn(z)
            else:
                a = self._out_act(z)
            self._A.append(a)
        return a

    # ------------------------------------------------------------------
    def _ce_loss(self, y_pred, y_true):
        yp = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(np.sum(y_true * np.log(yp), axis=1)))

    def _loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """仅返回主损失（不含正则项），便于监控训练进展"""
        if self.loss_name == "mse":
            return float(np.mean((y_pred - y_true) ** 2))
        return self._ce_loss(y_pred, y_true)

    # ------------------------------------------------------------------
    def backward(self, y_true: np.ndarray) -> None:
        m   = y_true.shape[0]
        out = self._A[-1]

        # 输出层 delta
        if self.loss_name == "cross_entropy":
            delta = (out - y_true) / m          # softmax + CE 联合导数
        else:
            delta = 2.0 * (out - y_true) / m    # MSE + linear

        dWs = []
        dbs = []
        for i in range(self.num_layers - 1, -1, -1):
            dW = self._A[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * self.W[i]
            dWs.insert(0, dW)
            dbs.insert(0, db)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._act_d(self._Z[i-1])

        # 全局梯度范数裁剪
        if self.max_grad_norm > 0:
            total_norm = np.sqrt(sum(np.sum(g*g) for g in dWs + dbs))
            if total_norm > self.max_grad_norm:
                scale = self.max_grad_norm / (total_norm + 1e-8)
                dWs = [g * scale for g in dWs]
                dbs = [g * scale for g in dbs]

        # SGD + Momentum 更新
        for i in range(self.num_layers):
            self.vW[i] = self.momentum * self.vW[i] + self.lr * dWs[i]
            self.vb[i] = self.momentum * self.vb[i] + self.lr * dbs[i]
            self.W[i] -= self.vW[i]
            self.b[i] -= self.vb[i]

    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 64,
        verbose: bool = True,
        print_every: int = 100,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val 和 y_val 必须同时提供，或同时为 None")

        m = len(X)
        history: dict = {"train_loss": [], "val_loss": []}

        for ep in range(1, epochs + 1):
            perm = np.random.permutation(m)
            Xs, ys = X[perm], y[perm]
            total_loss, n_bat = 0.0, 0

            for s in range(0, m, batch_size):
                Xb = Xs[s:s+batch_size]; yb = ys[s:s+batch_size]
                yp = self.forward(Xb)
                total_loss += self._loss(yp, yb)
                n_bat += 1
                self.backward(yb)

            tl = total_loss / n_bat
            history["train_loss"].append(tl)

            if X_val is not None and y_val is not None:
                vl = self._loss(self.forward(X_val), y_val)
                history["val_loss"].append(vl)

            if self.lr_decay_step > 0 and ep % self.lr_decay_step == 0:
                self.lr *= self.lr_decay_gamma

            if verbose and ep % print_every == 0:
                msg = f"Epoch {ep:5d}/{epochs}  loss={tl:.6f}"
                if X_val is not None and y_val is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y_labels: np.ndarray) -> float:
        return float((self.predict_class(X) == y_labels).mean())

    def save(self, path: str) -> None:
        data: dict = {"layer_sizes": np.array(self.layer_sizes)}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            data[f"W{i}"] = w
            data[f"b{i}"] = b
        np.savez(path, **data)
        print(f"Model saved: {path}.npz")

    def load(self, path: str) -> None:
        d = np.load(path, allow_pickle=True)
        n = len(self.W)
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
