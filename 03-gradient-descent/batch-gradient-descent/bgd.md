# Batch Gradient Descent — Math & GDRegressor

> Session 2 — Deep dive into BGD derivation and building GDRegressor from scratch

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | Intro |
| 2 | What is Gradient Descent |
| 3 | Types of Gradient Descent |
| 4 | Revision |
| 5 | Mathematical Formulation of BGD |
| 6 | Creating our own GDRegressor Class |

---

## 1. Intro

In this session we go deeper — deriving the gradient of MSE step by step and translating that directly into a working `GDRegressor` class. No black boxes.

---

## 2. What is Gradient Descent?

Gradient Descent minimizes a cost function by iteratively updating parameters in the direction of the negative gradient.

```
θ = θ - α * ∂J/∂θ
```

- `θ` — parameters (m, b)
- `α` — learning rate
- `∂J/∂θ` — gradient of cost w.r.t. parameter

---

## 3. Types of Gradient Descent

| Type | Dataset Used | Pros | Cons |
|------|-------------|------|------|
| **Batch GD (BGD)** | Full dataset | Stable convergence | Slow on large data |
| **Stochastic GD (SGD)** | 1 sample | Fast, noisy | High variance |
| **Mini-Batch GD** | Small batch | Balanced | Needs batch tuning |

---

## 4. Revision

Quick recap before the derivation:

- Model: `ŷ = m * x + b`
- Cost (MSE): `J = (1/n) * Σ (y - ŷ)²`
- Goal: find `m` and `b` that minimize `J`
- Chain Rule: `d/dx [f(g(x))] = f'(g(x)) * g'(x)`

---

## 5. Mathematical Formulation of BGD

### Step 1 — Model

```
ŷᵢ = m * xᵢ + b
```

### Step 2 — Cost Function (MSE)

```
J(m, b) = (1/n) * Σᵢ (yᵢ - ŷᵢ)²
```

---

### Step 3 — Gradient w.r.t. `m`

Differentiate using the chain rule:

```
∂J/∂m = ∂/∂m [ (1/n) * Σ (yᵢ - ŷᵢ)² ]
```

Pull constant out:

```
       = (1/n) * Σ ∂/∂m [ (yᵢ - ŷᵢ)² ]
```

Chain rule — let `u = (yᵢ - ŷᵢ)`:

```
       = (1/n) * Σ 2(yᵢ - ŷᵢ) * ∂(yᵢ - ŷᵢ)/∂m
```

Inner derivative → `∂(yᵢ - m*xᵢ - b)/∂m = -xᵢ`:

```
       = (1/n) * Σ 2(yᵢ - ŷᵢ) * (-xᵢ)
```

**Final:**

```
∂J/∂m = (-2/n) * Σ (yᵢ - ŷᵢ) * xᵢ
```

---

### Step 4 — Gradient w.r.t. `b`

Same process, inner derivative → `∂(yᵢ - m*xᵢ - b)/∂b = -1`:

```
∂J/∂b = (1/n) * Σ 2(yᵢ - ŷᵢ) * (-1)
```

**Final:**

```
∂J/∂b = (-2/n) * Σ (yᵢ - ŷᵢ)
```

---

### Step 5 — Update Rules

```
m = m - α * (-2/n) * Σ (yᵢ - ŷᵢ) * xᵢ
b = b - α * (-2/n) * Σ (yᵢ - ŷᵢ)
```

> The `(-2/n)` is the direct result of differentiating MSE with the chain rule. Some textbooks absorb the `2` into `α` — but here we keep it explicit.

---

## 6. Creating our own GDRegressor Class

Math → Code, 1:1 mapping:

```python
import numpy as np

class GDRegressor:
    
    def __init__(self, learning_rate=0.01, epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self, X_train, y_train):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            # update all the coef and the intercept
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            # ∂J/∂b = (-2/n) * Σ (y - ŷ)  →  using np.mean handles the /n
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            # ∂J/∂w = (-2/n) * Xᵀ (y - ŷ)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
        
        print(self.intercept_, self.coef_)
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### How the code maps to the math

| Code | Math |
|------|------|
| `y_hat = np.dot(X_train, self.coef_) + self.intercept_` | `ŷ = Xw + b` |
| `-2 * np.mean(y_train - y_hat)` | `(-2/n) * Σ (yᵢ - ŷᵢ)` = `∂J/∂b` |
| `-2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]` | `(-2/n) * Xᵀ(y - ŷ)` = `∂J/∂w` |
| `self.intercept_ - (self.lr * intercept_der)` | `b = b - α * ∂J/∂b` |
| `self.coef_ - (self.lr * coef_der)` | `w = w - α * ∂J/∂w` |

### Key design choices

- `coef_` initialized to `np.ones(X_train.shape[1])` — supports multiple features (not just 1)
- `intercept_` initialized to `0`
- `np.mean` on intercept gradient = dividing by `n` automatically
- `np.dot((y - ŷ), X)` computes the dot product across all samples in one shot — vectorized, no loops

### Usage

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = GDRegressor(learning_rate=0.01, epochs=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
```

---

## Key Takeaways

- **`(-2/n)`** comes from chain rule on MSE — always derive it, never memorize blindly
- Gradient w.r.t. `m` has an extra `xᵢ` factor; gradient w.r.t. `b` does not
- BGD uses the **entire dataset** every update — stable but slow on large data
- The `GDRegressor` above maps 1:1 with the math derivation

---

## Resources

- [CS229 - Andrew Ng Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest - Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Sklearn SGDRegressor Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)

---

*Learning in public. Mistakes are part of the process.*
