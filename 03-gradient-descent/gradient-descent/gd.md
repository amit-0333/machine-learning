# Gradient Descent — Full Study Notes

> Session 1 — Foundations, Intuition, Math & Visualization

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | Intro |
| 2 | Summary of Gradient Descent |
| 3 | What is Gradient Descent? |
| 4 | Plan of Attack |
| 5 | Intuition for GD |
| 6 | Mathematical Formulation of Gradient Descent |
| 7 | Code Demo |
| 8 | Creating our own Class and Methods |
| 9 | Visualizing our Class |
| 10 | Effect of Learning Rate |
| 11 | Universality of GD |
| 12 | Performing GD by adding 'm' |
| 13 | Visualisation |
| 14 | Code Demo and Visualization |
| 15 | Effect of Learning Rate |
| 16 | Effects of Loss Function |
| 17 | Effect of Data |

---

## 1. Intro

Gradient Descent is the backbone of almost every ML model training process. Before diving into complex models, understanding how GD works from scratch — mathematically and in code — is essential.

---

## 2. Summary of Gradient Descent

- An **optimization algorithm** that minimizes a cost function
- Works by computing the gradient (slope) and stepping in the opposite direction
- Core to training Linear Regression, Logistic Regression, Neural Networks, and more
- Three variants: **Batch**, **Stochastic**, **Mini-Batch**

---

## 3. What is Gradient Descent?

Gradient Descent iteratively updates model parameters to reduce prediction error.

```
θ_new = θ_old - α * ∂J/∂θ
```

- `θ` — parameters (weights, bias)
- `α` — learning rate
- `∂J/∂θ` — gradient of cost function

**Goal:** Find `θ` such that `J(θ)` is minimized.

---

## 4. Plan of Attack

1. Start with random parameters
2. Compute predictions `ŷ`
3. Compute loss `J`
4. Compute gradient `∂J/∂θ`
5. Update parameters
6. Repeat until convergence

---

## 5. Intuition for GD

Think of a ball rolling down a hill:
- The **hill** = cost function surface
- The **ball's position** = current parameters
- The **slope** = gradient
- The **step size** = learning rate

The ball always rolls downhill (negative gradient direction) until it settles at the lowest point (minimum loss).

---

## 6. Mathematical Formulation of Gradient Descent

### Model

```
ŷ = m * x + b
```

### Cost Function (MSE)

```
J(m, b) = (1/n) * Σ (yᵢ - ŷᵢ)²
```

### Gradients (via Chain Rule)

```
∂J/∂m = (-2/n) * Σ (yᵢ - ŷᵢ) * xᵢ
∂J/∂b = (-2/n) * Σ (yᵢ - ŷᵢ)
```

### Update Rules

```
m = m - α * ∂J/∂m
b = b - α * ∂J/∂b
```

---

## 7. Code Demo

```python
import numpy as np

# Dummy data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

m, b = 0.0, 0.0
alpha = 0.01
epochs = 1000
n = len(X)

for _ in range(epochs):
    y_hat = m * X + b
    dm = (-2/n) * np.sum((y - y_hat) * X)
    db = (-2/n) * np.sum(y - y_hat)
    m -= alpha * dm
    b -= alpha * db

print(f"m = {m:.4f}, b = {b:.4f}")
```

---

## 8. Creating our own Class and Methods

```python
import numpy as np

class GradientDescentRegressor:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.m = 0.0
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_hat = self.predict(X)
            loss = np.mean((y - y_hat) ** 2)
            self.loss_history.append(loss)

            dm = (-2/n) * np.sum((y - y_hat) * X)
            db = (-2/n) * np.sum(y - y_hat)

            self.m -= self.lr * dm
            self.b -= self.lr * db

    def predict(self, X):
        return self.m * X + self.b
```

---

## 9. Visualizing our Class

```python
import matplotlib.pyplot as plt

model = GradientDescentRegressor(lr=0.01, epochs=500)
model.fit(X, y)

# Plot 1: Loss curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model.loss_history)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

# Plot 2: Regression line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted')
plt.title("Regression Line")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 10. Effect of Learning Rate

| Learning Rate | Behaviour |
|--------------|-----------|
| Too small (e.g. 0.0001) | Converges very slowly |
| Just right (e.g. 0.01) | Smooth, steady convergence |
| Too large (e.g. 1.0) | Overshoots, diverges |

```python
for lr in [0.0001, 0.01, 0.5]:
    model = GradientDescentRegressor(lr=lr, epochs=300)
    model.fit(X, y)
    plt.plot(model.loss_history, label=f"lr={lr}")

plt.legend()
plt.title("Effect of Learning Rate on Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()
```

---

## 11. Universality of GD

Gradient Descent is **not limited to linear regression**. It works for any differentiable cost function:

- Logistic Regression → Binary Cross-Entropy loss
- Neural Networks → Backpropagation uses GD at its core
- SVMs, embeddings, and more

As long as you can compute `∂J/∂θ`, GD can optimize it.

---

## 12. Performing GD by Adding 'm'

Starting from `b` only, we add slope `m` to make the model more expressive:

```
# Only bias (horizontal line)
ŷ = b

# Adding slope m (regression line)
ŷ = m * x + b
```

Each parameter needs its own gradient and update. Adding `m` introduces the `xᵢ` term into the gradient.

---

## 13. Visualisation

```python
# Visualize how parameters evolve
m_vals, b_vals, loss_vals = [], [], []

m_temp, b_temp = 0.0, 0.0
for _ in range(200):
    y_hat = m_temp * X + b_temp
    loss_vals.append(np.mean((y - y_hat)**2))
    m_vals.append(m_temp)
    b_vals.append(b_temp)
    dm = (-2/n) * np.sum((y - y_hat) * X)
    db = (-2/n) * np.sum(y - y_hat)
    m_temp -= 0.01 * dm
    b_temp -= 0.01 * db

plt.plot(m_vals, label='m')
plt.plot(b_vals, label='b')
plt.title("Parameter Evolution")
plt.legend()
plt.show()
```

---

## 14. Code Demo and Visualization (Combined)

```python
model = GradientDescentRegressor(lr=0.01, epochs=500)
model.fit(X, y)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(model.loss_history)
axes[0].set_title("Loss Curve")

axes[1].scatter(X, y)
axes[1].plot(X, model.predict(X), 'r')
axes[1].set_title("Final Fit")

axes[2].scatter(range(len(X)), y - model.predict(X))
axes[2].axhline(0, color='r', linestyle='--')
axes[2].set_title("Residuals")

plt.tight_layout()
plt.show()
```

---

## 15. Effect of Learning Rate (Revisited)

A **high learning rate** causes the loss to oscillate or explode.
A **low learning rate** causes very slow learning.
The **sweet spot** depends on your data scale — always normalize features before training.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).flatten()
```

---

## 16. Effects of Loss Function

Different loss functions change the gradient shape:

| Loss Function | Formula | Use Case |
|--------------|---------|----------|
| MSE | `(1/n) Σ(y - ŷ)²` | Regression, penalizes large errors |
| MAE | `(1/n) Σ\|y - ŷ\|` | Robust to outliers |
| Huber | Mix of MSE + MAE | Best of both worlds |

The loss function you choose directly changes `∂J/∂θ` and therefore the update step.

---

## 17. Effect of Data

| Data Property | Impact on GD |
|--------------|-------------|
| Not normalized | Slow / unstable convergence |
| Outliers present | Gradients get skewed |
| Small dataset | Risk of overfitting |
| Large dataset | BGD becomes expensive → use Mini-Batch |

**Always normalize your data before running Gradient Descent.**

---

## Key Takeaways

- GD is universal — works for any differentiable loss
- The `-2/n` in gradients comes from chain rule on MSE
- Learning rate is the most sensitive hyperparameter
- Visualizing loss curves is essential for debugging training

---

## Resources

- [CS229 - Andrew Ng Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest - Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)

---

*Learning in public. Mistakes are part of the process.*
