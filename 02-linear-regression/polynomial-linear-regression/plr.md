# Polynomial Regression — Complete Notes

> Definition, Why to use, Full Derivation, From Scratch + Sklearn

---

## Table of Contents

| # | Topic |
|---|-------|
| 1 | What is Polynomial Regression? |
| 2 | Why Use Polynomial Regression? |
| 3 | Mathematical Derivation |
| 4 | The Bias-Variance Tradeoff |
| 5 | Code From Scratch |
| 6 | Code Using Sklearn |
| 7 | Choosing the Right Degree |
| 8 | Key Takeaways |

---

## 1. What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression where the relationship between X and y is modelled as an **nth degree polynomial** instead of a straight line.

```
Linear Regression:
y = β₀ + β₁x

Polynomial Regression (degree 2):
y = β₀ + β₁x + β₂x²

Polynomial Regression (degree n):
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

**Key insight:** Despite being "non-linear" in X, polynomial regression is still **linear in the coefficients** (β₀, β₁, ..., βₙ). This means we can still use OLS / Gradient Descent to solve it — we just engineer new features first.

```
Original feature:   [x]
After degree-2 transform: [x, x²]   ← now it's just multivariate linear regression
```

---

## 2. Why Use Polynomial Regression?

### When Linear Regression Fails

Linear regression assumes a straight-line relationship between X and y. Real-world data often curves:

```
Linear fit on curved data:
                        * *
                      *     *
                   *           *     ← actual data (curved)
-------------------               ---   ← linear fit (wrong)

Polynomial fit:
                        * *
                  ----*-----*----
               --*             *--   ← polynomial fit (correct)
```

### Real-world examples where we need polynomial regression

| Dataset | Why linear fails | Degree needed |
|---------|-----------------|---------------|
| House price vs size | Diminishing returns (large houses don't scale linearly) | 2 |
| Drug dosage vs effect | Increases then plateaus/drops (bell curve) | 2–3 |
| Engine RPM vs efficiency | Non-linear mechanical relationship | 2–3 |
| Temperature vs ice cream sales | Seasonal curve | 2 |
| Population growth | Exponential-like curve | 3+ |

### The core reason

If the **linearity assumption of linear regression is violated** (residual plot shows a curve instead of random scatter), polynomial regression is the natural fix — adding `x²`, `x³` terms captures the curve without switching to a completely different algorithm.

---

## 3. Mathematical Derivation

### Step 1 — Feature Transformation

For a single feature `x` and degree `d`, we create new features:

```
Original:   x
New matrix: [1,  x,  x²,  x³, ..., xᵈ]
```

For `n` samples and degree `d`, the design matrix **X** becomes:

```
X = | 1   x₁   x₁²   x₁³  ...  x₁ᵈ |
    | 1   x₂   x₂²   x₂³  ...  x₂ᵈ |
    | 1   x₃   x₃²   x₃³  ...  x₃ᵈ |
    | .    .    .     .         .   |
    | 1   xₙ   xₙ²   xₙ³  ...  xₙᵈ |

Shape: (n × (d+1))
```

### Step 2 — Model Definition

```
ŷ = Xβ

Where β = [β₀, β₁, β₂, ..., βᵈ]ᵀ   (column vector of coefficients)
```

This is now identical in form to **multivariate linear regression** — the polynomial terms are just treated as separate features.

### Step 3 — Cost Function (MSE)

```
J(β) = (1/n) * Σ (yᵢ - ŷᵢ)²
     = (1/n) * ||y - Xβ||²
```

### Step 4 — OLS Closed-Form Solution

Minimize J(β) by setting the gradient to zero:

```
∂J/∂β = 0

∂/∂β [(y - Xβ)ᵀ(y - Xβ)] = 0

Expanding:
∂/∂β [yᵀy - 2βᵀXᵀy + βᵀXᵀXβ] = 0

-2Xᵀy + 2XᵀXβ = 0

XᵀXβ = Xᵀy

β = (XᵀX)⁻¹ Xᵀy   ← Normal Equation (OLS solution)
```

This is the **same Normal Equation as linear regression** — the math doesn't change, only the features do.

### Step 5 — Gradient Descent Alternative

If the dataset is large and inverting `XᵀX` is expensive, use Gradient Descent:

```
∂J/∂β = (-2/n) * Xᵀ(y - Xβ)

Update rule:
β = β - α * (-2/n) * Xᵀ(y - ŷ)
```

### Step 6 — Prediction

```
ŷ_new = β₀ + β₁x_new + β₂x_new² + ... + βᵈx_newᵈ
```

Or in matrix form:

```
ŷ_new = X_new · β
```

---

## 4. The Bias-Variance Tradeoff

Degree is the most important hyperparameter in polynomial regression:

```
degree = 1  →  Underfitting (high bias, low variance)
              Straight line, misses the curve
              Training error HIGH, Test error HIGH

degree = 2-3 →  Just right (sweet spot)
              Captures the actual curve
              Training error LOW, Test error LOW

degree = 15+ →  Overfitting (low bias, high variance)
              Wiggles through every training point
              Training error ≈ 0, Test error HIGH
```

```
Error
  |
  |  \                        /   ← Test error (U-shape)
  |   \                      /
  |    \----________________/
  |         \______________/      ← Train error (decreases)
  |
  +-----------------------------→ Degree
       1    2    3    4   ...
            ↑
       Sweet spot
```

> Always use cross-validation to find the right degree — never pick it by looking at training error alone.

---

## 5. Code From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

#  Feature Engineering 
def polynomial_features(X, degree):
    """
    Transform X into polynomial features up to given degree.
    Input:  X shape (n,) or (n,1)
    Output: X_poly shape (n, degree+1) — includes bias column of 1s
    """
    X = X.flatten()
    n = len(X)
    X_poly = np.ones((n, degree + 1))   # first column = 1 (bias)
    for d in range(1, degree + 1):
        X_poly[:, d] = X ** d
    return X_poly


# OLS Closed-Form Solution 
class PolynomialRegressionOLS:
    """Polynomial Regression using the Normal Equation: β = (XᵀX)⁻¹ Xᵀy"""

    def __init__(self, degree=2):
        self.degree = degree
        self.coef_  = None   # β₁, β₂, ..., βᵈ
        self.intercept_ = None  # β₀

    def fit(self, X, y):
        X_poly = polynomial_features(X, self.degree)

        # Normal equation: β = (XᵀX)⁻¹ Xᵀy
        XtX     = X_poly.T @ X_poly
        Xty     = X_poly.T @ y
        self.beta = np.linalg.inv(XtX) @ Xty   # shape: (degree+1,)

        self.intercept_ = self.beta[0]
        self.coef_      = self.beta[1:]

        print(f"Intercept (β₀): {self.intercept_:.4f}")
        print(f"Coefficients  : {self.coef_}")

    def predict(self, X):
        X_poly = polynomial_features(X, self.degree)
        return X_poly @ self.beta


# Gradient Descent Solution 
class PolynomialRegressionGD:
    """Polynomial Regression using Batch Gradient Descent"""

    def __init__(self, degree=2, learning_rate=0.01, epochs=1000):
        self.degree = degree
        self.lr     = learning_rate
        self.epochs = epochs
        self.beta   = None
        self.loss_history = []

    def fit(self, X, y):
        X_poly = polynomial_features(X, self.degree)
        n, d   = X_poly.shape
        self.beta = np.zeros(d)   # init all β to 0

        for epoch in range(self.epochs):
            y_hat = X_poly @ self.beta

            # Gradient: ∂J/∂β = (-2/n) * Xᵀ(y - ŷ)
            grad = (-2 / n) * X_poly.T @ (y - y_hat)

            self.beta -= self.lr * grad

            loss = np.mean((y - y_hat) ** 2)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

        self.intercept_ = self.beta[0]
        self.coef_      = self.beta[1:]

    def predict(self, X):
        X_poly = polynomial_features(X, self.degree)
        return X_poly @ self.beta


#  Generate non-linear data 
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 2*X**2 - 3*X + 1 + np.random.randn(100) * 2   # true: degree 2

X_train, y_train = X[:80], y[:80]
X_test,  y_test  = X[80:], y[80:]

# Normalize (important for GD)
X_mean, X_std = X_train.mean(), X_train.std()
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled  = (X_test  - X_mean) / X_std


#  OLS 
ols = PolynomialRegressionOLS(degree=2)
ols.fit(X_train, y_train)

y_pred_ols = ols.predict(X_test)
mse_ols = np.mean((y_test - y_pred_ols) ** 2)
print(f"\nOLS Test MSE: {mse_ols:.4f}")


#  Gradient Descent 
gd = PolynomialRegressionGD(degree=2, learning_rate=0.01, epochs=1000)
gd.fit(X_train_scaled, y_train)

y_pred_gd = gd.predict(X_test_scaled)
mse_gd = np.mean((y_test - y_pred_gd) ** 2)
print(f"\nGD  Test MSE: {mse_gd:.4f}")


#  Visualization 
X_plot = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: OLS fit
axes[0].scatter(X_train, y_train, alpha=0.5, label='Train')
axes[0].scatter(X_test, y_test, alpha=0.5, label='Test', color='orange')
axes[0].plot(X_plot, ols.predict(X_plot), color='red', label='Poly fit (OLS)')
axes[0].set_title("Polynomial Regression (OLS)")
axes[0].legend()

# Plot 2: GD loss curve
axes[1].plot(gd.loss_history)
axes[1].set_title("GD Loss Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")

# Plot 3: degree comparison
for deg, color in zip([1, 2, 5], ['blue', 'green', 'red']):
    m = PolynomialRegressionOLS(degree=deg)
    m.fit(X_train, y_train)
    axes[2].plot(X_plot, m.predict(X_plot), label=f'degree={deg}', color=color)
axes[2].scatter(X_train, y_train, alpha=0.3, color='gray')
axes[2].set_title("Effect of Degree")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## 6. Code Using Sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Data 
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = 2*X.flatten()**2 - 3*X.flatten() + 1 + np.random.randn(200) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#  PolynomialFeatures explained 
# sklearn's PolynomialFeatures does the feature engineering step for you:
pf = PolynomialFeatures(degree=2, include_bias=True)
X_poly = pf.fit_transform(X_train[:3])
print("Original X[:3]:\n", X_train[:3])
print("\nAfter PolynomialFeatures(degree=2):\n", X_poly)
# Output columns: [1, x, x²]


#  Simple Pipeline 
degree = 2

poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=True)),
    ('scaler',        StandardScaler()),
    ('linear_reg',    LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)

print(f"\nDegree {degree} Polynomial Regression")
print(f"Test MSE : {mean_squared_error(y_test, y_pred):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred):.4f}")


#  Finding best degree using Cross-Validation 
degrees = range(1, 12)
train_errors, test_errors, cv_scores = [], [], []

for d in degrees:
    model = Pipeline([
        ('poly',    PolynomialFeatures(degree=d, include_bias=True)),
        ('scaler',  StandardScaler()),
        ('lr',      LinearRegression())
    ])
    model.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse  = mean_squared_error(y_test,  model.predict(X_test))
    cv_mse    = -cross_val_score(model, X_train, y_train,
                                  cv=5, scoring='neg_mean_squared_error').mean()

    train_errors.append(train_mse)
    test_errors.append(test_mse)
    cv_scores.append(cv_mse)

best_degree = degrees[np.argmin(cv_scores)]
print(f"\nBest degree by CV: {best_degree}")


#  Visualization 
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1: Best model fit
best_model = Pipeline([
    ('poly',   PolynomialFeatures(degree=best_degree, include_bias=True)),
    ('scaler', StandardScaler()),
    ('lr',     LinearRegression())
])
best_model.fit(X_train, y_train)

axes[0].scatter(X_train, y_train, alpha=0.4, label='Train')
axes[0].scatter(X_test, y_test, alpha=0.4, color='orange', label='Test')
axes[0].plot(X_plot, best_model.predict(X_plot), color='red',
             label=f'Poly degree={best_degree}')
axes[0].set_title("Best Polynomial Fit")
axes[0].legend()

# Plot 2: Bias-variance tradeoff
axes[1].plot(degrees, train_errors, label='Train MSE', marker='o')
axes[1].plot(degrees, test_errors,  label='Test MSE',  marker='o')
axes[1].plot(degrees, cv_scores,    label='CV MSE',    marker='o', linestyle='--')
axes[1].axvline(best_degree, color='red', linestyle='--', label=f'Best degree={best_degree}')
axes[1].set_title("Bias-Variance Tradeoff")
axes[1].set_xlabel("Polynomial Degree")
axes[1].set_ylabel("MSE")
axes[1].legend()

# Plot 3: Underfitting vs Overfitting
for d, color, label in [(1, 'blue', 'Underfit (d=1)'),
                         (best_degree, 'green', f'Good fit (d={best_degree})'),
                         (10, 'red', 'Overfit (d=10)')]:
    m = Pipeline([
        ('poly',   PolynomialFeatures(degree=d)),
        ('scaler', StandardScaler()),
        ('lr',     LinearRegression())
    ])
    m.fit(X_train, y_train)
    axes[2].plot(X_plot, m.predict(X_plot), color=color, label=label)
axes[2].scatter(X_train, y_train, alpha=0.3, color='gray')
axes[2].set_title("Underfit vs Good Fit vs Overfit")
axes[2].set_ylim(-30, 50)
axes[2].legend()

plt.tight_layout()
plt.show()


#  Extract coefficients 
lr = best_model.named_steps['lr']
feature_names = best_model.named_steps['poly'].get_feature_names_out(['x'])
for name, coef in zip(feature_names, lr.coef_):
    print(f"  {name:10s}: {coef:.4f}")
print(f"  intercept  : {lr.intercept_:.4f}")
```

---

## 7. Choosing the Right Degree

```
Rule of thumb:
  Start with degree=2 or degree=3
  Use cross-validation to tune
  Stop when CV error stops improving (or starts rising)
```

| Degree | Train Error | Test Error | Verdict |
|--------|-------------|------------|---------|
| 1 | High | High | Underfitting |
| 2 | Low | Low | ✅ Sweet spot (for quadratic data) |
| 3 | Low | Low | ✅ Good if data needs cubic curve |
| 10 | ≈ 0 | Very High | Overfitting |

**Always:**
1. Normalize features before polynomial transform (large powers → huge values)
2. Use `Pipeline` in sklearn — prevents data leakage when scaling
3. Use `cross_val_score` — not test set — to pick degree

---

## 8. Key Takeaways

- Polynomial regression = **linear regression on engineered features** (`x`, `x²`, `x³`, ...)
- The math is **identical to multivariate linear regression** — same Normal Equation, same GD update
- The feature transformation step is: `[x] → [1, x, x², ..., xᵈ]`
- **OLS Normal Equation:** `β = (XᵀX)⁻¹ Xᵀy` — exact, but expensive for large n
- **Gradient Descent:** `β = β - α * (-2/n) * Xᵀ(y - ŷ)` — scalable alternative
- Degree is the critical hyperparameter — **cross-validation** is the right way to choose it
- High degree → overfitting; Low degree → underfitting → **Bias-Variance tradeoff**

---

## Resources

- [Sklearn PolynomialFeatures Docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Sklearn Pipeline Docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [StatQuest - Polynomial Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)

---

*Learning in public. Mistakes are part of the process.*
