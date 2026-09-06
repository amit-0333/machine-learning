# Stochastic Gradient Descent — Notes & Implementation

> Session 3 — Problems with BGD, SGD from scratch, Learning Schedules & Sklearn

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | Intro |
| 2 | The Problem with Batch Gradient Descent |
| 3 | Stochastic Gradient Descent |
| 4 | Code Demo |
| 5 | Creating our own SGD Class from Scratch |
| 6 | Time Comparison / Question for you |
| 7 | Code Visualization |
| 8 | When should you use SGD? |
| 9 | Learning Schedules |
| 10 | SGD Regressor Class in Sklearn |

---

## 1. Intro

Batch Gradient Descent works well on small datasets but becomes painfully slow as data grows. **Stochastic Gradient Descent (SGD)** solves this by updating parameters using **one sample at a time** — making it much faster, at the cost of some noise.

---

## 2. The Problem with Batch Gradient Descent

In BGD, every single update requires passing through the **entire dataset**:

```
coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
```

| Problem | Why it hurts |
|---------|-------------|
| **Slow on large data** | 1M rows × 1000 epochs = 1B computations per parameter |
| **High memory usage** | Entire dataset must be loaded for every update |
| **Redundant computation** | Similar samples recalculated every epoch |
| **Stuck in local minima** | Smooth loss surface offers no escape |

> BGD is fine for small datasets. For anything large → SGD or Mini-Batch.

---

## 3. Stochastic Gradient Descent

Instead of the full dataset, SGD picks **one random sample** per update:

```
For each epoch:
    shuffle the data
    For each sample (xᵢ, yᵢ):
        ŷᵢ = w · xᵢ + b
        ∂J/∂w = -2 * (yᵢ - ŷᵢ) * xᵢ
        ∂J/∂b = -2 * (yᵢ - ŷᵢ)
        w = w - α * ∂J/∂w
        b = b - α * ∂J/∂b
```

**Key differences from BGD:**

| | BGD | SGD |
|-|-----|-----|
| Samples per update | All `n` | 1 |
| Updates per epoch | 1 | `n` |
| Loss curve | Smooth | Noisy |
| Speed | Slow | Fast |
| Convergence | Stable | Oscillates near minimum |

---

## 4. Code Demo

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

---

## 5. Creating our own SGD Class from Scratch

```python
import numpy as np

class SGDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        # init coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                # pick one random sample
                idx = np.random.randint(0, X_train.shape[0])
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

                # gradients on single sample (no /n — just one point)
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### BGD vs SGD — side by side

| | BGD | SGD |
|-|-----|-----|
| `y_hat` | `np.dot(X_train, coef_)` — full matrix | `np.dot(X_train[idx], coef_)` — single row |
| `intercept_der` | `-2 * np.mean(y - y_hat)` | `-2 * (y[idx] - y_hat)` |
| `coef_der` | `-2 * np.dot((y - y_hat), X) / n` | `-2 * (y[idx] - y_hat) * X[idx]` |

---

## 6. Time Comparison

```python
import time

# BGD timing
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=10000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

start = time.time()
bgd = GDRegressor(learning_rate=0.01, epochs=100)
bgd.fit(X_train, y_train)
print(f"BGD time: {time.time() - start:.4f}s")

# SGD timing
start = time.time()
sgd = SGDRegressor(learning_rate=0.01, epochs=100)
sgd.fit(X_train, y_train)
print(f"SGD time: {time.time() - start:.4f}s")
```

> **Question for you:** SGD does `n` updates per epoch vs BGD's `1`. So why is SGD faster overall?
> 
> **Answer:** Each SGD update is computed on 1 row instead of n rows — the per-update cost is `O(1)` vs `O(n)`. Total work per epoch is similar, but SGD converges in fewer epochs.

---

## 7. Code Visualization

```python
import matplotlib.pyplot as plt

# Track loss per epoch
loss_bgd, loss_sgd = [], []

for epoch in range(100):
    # BGD — one update per epoch
    y_hat = np.dot(X_train, bgd.coef_) + bgd.intercept_
    loss_bgd.append(np.mean((y_train - y_hat)**2))

    # SGD — noisy, one random sample at a time
    idx = np.random.randint(0, X_train.shape[0])
    y_hat_s = np.dot(X_train[idx], sgd.coef_) + sgd.intercept_
    loss_sgd.append((y_train[idx] - y_hat_s)**2)

plt.plot(loss_bgd, label='BGD Loss', linewidth=2)
plt.plot(loss_sgd, label='SGD Loss (noisy)', alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BGD vs SGD Loss Curve")
plt.legend()
plt.show()
```

**What you'll see:**
- BGD → smooth, steady decline
- SGD → noisy, zigzag but reaches low loss faster

---

## 8. When should you use SGD?

| Use SGD when... | Avoid SGD when... |
|----------------|------------------|
| Dataset is very large (millions of rows) | Dataset is small (BGD is fine) |
| Online learning (streaming data) | You need a smooth, stable loss curve |
| You want to escape local minima | Precise convergence is required |
| Memory is limited | Reproducibility is critical (SGD is random) |

**Rule of thumb:**
- Small data → **BGD**
- Large data → **Mini-Batch GD** (best of both)
- Streaming / online → **SGD**

---

## 9. Learning Schedules

A fixed learning rate can cause SGD to oscillate forever near the minimum. **Learning schedules** decay `α` over time so SGD settles down.

### Common Strategies

**1. Time-based decay**
```python
alpha = alpha_0 / (1 + decay * epoch)
```

**2. Step decay** — drop lr by a factor every N epochs
```python
if epoch % step_size == 0:
    alpha = alpha * drop_factor   # e.g. 0.5
```

**3. Exponential decay**
```python
alpha = alpha_0 * np.exp(-decay_rate * epoch)
```

**4. 1/t schedule (classic SGD)**
```python
alpha = t0 / (epoch + t1)   # t0, t1 are tunable constants
```

### Implementation in SGD class

```python
def fit(self, X_train, y_train, t0=5, t1=50):
    self.intercept_ = 0
    self.coef_ = np.ones(X_train.shape[1])
    t = 0

    for epoch in range(self.epochs):
        for j in range(X_train.shape[0]):
            # Learning schedule: lr decays over time
            lr_t = t0 / (t + t1)
            t += 1

            idx = np.random.randint(0, X_train.shape[0])
            y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

            intercept_der = -2 * (y_train[idx] - y_hat)
            self.intercept_ -= lr_t * intercept_der

            coef_der = -2 * (y_train[idx] - y_hat) * X_train[idx]
            self.coef_ -= lr_t * coef_der
```

---

## 10. SGD Regressor Class in Sklearn

Sklearn's `SGDRegressor` is a production-ready implementation:

```python
from sklearn.linear_model import SGDRegressor

sgd_sklearn = SGDRegressor(
    loss='squared_error',    # MSE loss (same as what we derived)
    learning_rate='invscaling',  # learning schedule
    eta0=0.01,               # initial learning rate
    max_iter=100,            # epochs
    tol=1e-3,                # stop if loss doesn't improve
    random_state=42
)

sgd_sklearn.fit(X_train, y_train)
print("Intercept:", sgd_sklearn.intercept_)
print("Coefs:", sgd_sklearn.coef_)

preds = sgd_sklearn.predict(X_test)
```

### Mapping our class → Sklearn

| Our SGDRegressor | Sklearn SGDRegressor |
|-----------------|---------------------|
| `epochs` | `max_iter` |
| `learning_rate` (fixed) | `eta0` + `learning_rate` schedule |
| Manual random index | Internal shuffle per epoch |
| `coef_` | `coef_` |
| `intercept_` | `intercept_` |

---

## Key Takeaways

- BGD is slow on large data — computes gradients over all `n` samples every update
- SGD updates on **1 sample at a time** — fast but noisy
- The `-2` factor remains; the `/n` disappears (only 1 sample, not a mean)
- **Learning schedules** are essential for SGD to converge properly
- Sklearn's `SGDRegressor` is the go-to for production use

---

## Resources

- [Sklearn SGDRegressor Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [CS229 - Andrew Ng Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest - SGD](https://www.youtube.com/watch?v=vMh0zPT0tLI)

---

*Learning in public. Mistakes are part of the process.*
