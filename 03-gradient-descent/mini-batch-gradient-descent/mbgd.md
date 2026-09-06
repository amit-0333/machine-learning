# Mini-Batch Gradient Descent — Notes & Implementation

> Session 4 — Best of both worlds: BGD stability + SGD speed

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | Intro |
| 2 | What is Mini-Batch Gradient Descent |
| 3 | Coding Mini-Batch GD from Scratch |
| 4 | Visualization of the Code |

---

## 1. Intro

We've seen two extremes:
- **BGD** — uses all `n` samples per update → stable but slow
- **SGD** — uses 1 sample per update → fast but noisy

**Mini-Batch GD** sits right in the middle. It updates parameters using a small **batch** of samples (e.g. 32, 64, 128) — giving you the speed of SGD with much smoother convergence.

This is what almost every real-world ML and Deep Learning training loop uses.

---

## 2. What is Mini-Batch Gradient Descent?

Instead of all `n` samples (BGD) or 1 sample (SGD), Mini-Batch GD picks a **random subset** of size `batch_size` per update:

```
For each epoch:
    shuffle the data
    split into batches of size k
    For each batch (X_batch, y_batch):
        ŷ_batch = X_batch · w + b
        ∂J/∂w = (-2/k) * X_batchᵀ · (y_batch - ŷ_batch)
        ∂J/∂b = (-2/k) * Σ (y_batch - ŷ_batch)
        w = w - α * ∂J/∂w
        b = b - α * ∂J/∂b
```

### Comparison Table

| | BGD | SGD | Mini-Batch GD |
|-|-----|-----|---------------|
| Samples per update | All `n` | 1 | `k` (e.g. 32) |
| Updates per epoch | 1 | `n` | `n/k` |
| Loss curve | Smooth | Very noisy | Slightly noisy, stable |
| Speed | Slow | Fast | Fast |
| Memory | High | Low | Moderate |
| Used in practice | Rarely | Sometimes | **Almost always** |

### Typical batch sizes
- `8, 16, 32, 64, 128, 256`
- Powers of 2 — fits GPU memory efficiently
- **32 or 64** is the most common default

---

## 3. Coding Mini-Batch GD from Scratch

```python
import numpy as np

class MBGDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        # init coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # shuffle data at start of every epoch
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            # split into batches
            for j in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[j : j + self.batch_size]
                y_batch = y_train[j : j + self.batch_size]

                # forward pass on batch
                y_hat = np.dot(X_batch, self.coef_) + self.intercept_

                # gradients — same formula as BGD but over batch size k
                intercept_der = -2 * np.mean(y_batch - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_batch - y_hat), X_batch) / X_batch.shape[0]
                self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### How the code maps to the math

| Code | Math |
|------|------|
| `X_batch = X_train[j : j + batch_size]` | Select batch of size `k` |
| `y_hat = np.dot(X_batch, self.coef_) + self.intercept_` | `ŷ = X_batch · w + b` |
| `-2 * np.mean(y_batch - y_hat)` | `(-2/k) * Σ (yᵢ - ŷᵢ)` = `∂J/∂b` |
| `-2 * np.dot((y_batch - y_hat), X_batch) / X_batch.shape[0]` | `(-2/k) * X_batchᵀ(y - ŷ)` = `∂J/∂w` |
| `self.intercept_ - (self.lr * intercept_der)` | `b = b - α * ∂J/∂b` |
| `self.coef_ - (self.lr * coef_der)` | `w = w - α * ∂J/∂w` |

### Key design choices
- **Shuffle every epoch** — prevents the model from memorizing batch order
- `np.random.permutation` returns shuffled indices, not in-place shuffle
- Slicing `X_train[j : j + batch_size]` naturally handles the last batch (smaller than `batch_size`) without any special case

---

## 4. Visualization of the Code

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Track loss manually for visualization
class MBGDRegressorViz(MBGDRegressor):
    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        self.loss_history = []

        for i in range(self.epochs):
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            for j in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[j : j + self.batch_size]
                y_batch = y_train[j : j + self.batch_size]

                y_hat = np.dot(X_batch, self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_batch - y_hat)
                self.intercept_ -= self.lr * intercept_der

                coef_der = -2 * np.dot((y_batch - y_hat), X_batch) / X_batch.shape[0]
                self.coef_ -= self.lr * coef_der

            # record epoch loss on full training set
            epoch_loss = np.mean((y_train - self.predict(X_train))**2)
            self.loss_history.append(epoch_loss)

# Train with different batch sizes
results = {}
for bs in [1, 32, len(X_train)]:   # SGD, Mini-Batch, BGD
    model = MBGDRegressorViz(learning_rate=0.01, epochs=50, batch_size=bs)
    model.fit(X_train.copy(), y_train.copy())
    label = {1: "SGD (batch=1)", len(X_train): "BGD (full)", 32: "Mini-Batch (batch=32)"}[bs]
    results[label] = model.loss_history

# Plot
plt.figure(figsize=(10, 5))
for label, losses in results.items():
    plt.plot(losses, label=label)

plt.title("Loss Curve: SGD vs Mini-Batch vs BGD")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()
```

**What you'll observe:**
- **BGD** — smoothest curve, slowest per-epoch
- **SGD** — most noise, fastest to start dropping
- **Mini-Batch** — smooth like BGD, fast like SGD ✅

---

## Key Takeaways

- Mini-Batch GD = BGD formula applied to a **subset** of size `k`
- The `-2/k` gradient is identical to BGD's `-2/n` — just `n` is now `batch_size`
- **Shuffle every epoch** — critical to avoid order bias
- Batch size is a hyperparameter: start with `32`, tune if needed
- This is the default training loop in **PyTorch, TensorFlow, Keras**

---

## All Three — Quick Recap

```
BGD:       update once  using ALL n samples   → stable, slow
SGD:       update n times using 1 sample each → fast, noisy  
Mini-Batch: update n/k times using k samples  → fast, stable ✅
```

---

## Resources

- [Sklearn SGDRegressor Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [CS229 - Andrew Ng Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Deep Learning Book - Ch. 8 Optimization](https://www.deeplearningbook.org/contents/optimization.html)

---

*Learning in public. Mistakes are part of the process.*
