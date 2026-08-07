# Simple Linear Regression

## 1. Introduction

Simple Linear Regression is a supervised machine learning algorithm used to model the relationship between **one input feature** and **one output/target** using a straight line.

**Example dataset — Placement Prediction**

| CGPA | Package (LPA) |
|------|----------------|
| 7.1  | 3.5            |
| 4.7  | 1.2            |
| 8.9  | 4.2            |
| 8.1  | 3.9            |
| ...  | ...            |

Given ~200 students' data (CGPA vs. Package), the goal is to build a **model** that, given a new CGPA (e.g. 7), predicts the expected package (e.g. 3.5 LPA).

The data looks *sort of linear*, so we try to fit a straight line through it.

### Why Linear Regression?
- Works well on real-world datasets that show a roughly linear trend.
- The goal is to find the **best fit line** through the data points.

---

## 2. The Model

The line is represented as:

```
y = m·x + b
```

- `x` → input feature (CGPA)
- `y` → predicted output (Package)
- `m` → **slope** of the line
- `b` → **y-intercept**

The entire learning problem reduces to finding the best values of **(m, b)** — this pair fully defines the model (this combination is often referred to as **LR** — Linear Regression).

From the scatter plot (Package vs CGPA), many lines could be drawn through the data (e.g. lines ①, ②, ③), but only one is the **best fit line** — the one that minimizes overall prediction error.

---

## 3. How to Find m and b?

To find the best line, we define an **error function E** that measures how far off our predictions are from the actual values, and then minimize it.

### Error Function (Sum of Squared Errors)

```
E = Σ (yᵢ − m·xᵢ − b)²      for i = 1 to n
```

This is a **convex function** — it has a single minimum point (like a bowl shape), both with respect to `m` and `b` individually, and jointly when plotted as a 3D surface `J(θ₀, θ₁)` (cost function) with axes `m` and `b`.

To find the minimum, we use calculus: set the partial derivatives to zero.

```
∂E/∂m = 0
∂E/∂b = 0
```

### Deriving b

```
∂E/∂b = ∂/∂b Σ (yᵢ − m·xᵢ − b)² = 0

⇒ Σ −2(yᵢ − m·xᵢ − b) = 0
⇒ Σ (yᵢ − m·xᵢ − b) = 0
⇒ Σyᵢ − mΣxᵢ − nb = 0
⇒ ȳ − m·x̄ − b = 0   (dividing through by n)

∴  b = ȳ − m·x̄
```

Where `x̄` and `ȳ` are the mean of x and y values respectively.

### Deriving m

Substituting `b = ȳ − m·x̄` back into E:

```
E = Σ (yᵢ − m·xᵢ − ȳ + m·x̄)²

∂E/∂m = Σ 2(yᵢ − m·xᵢ − ȳ + m·x̄)(−xᵢ + x̄) = 0
```

Solving this equation gives the closed-form (Ordinary Least Squares) solution for `m`:

```
        Σ (xᵢ − x̄)(yᵢ − ȳ)
m  =   ─────────────────────
        Σ (xᵢ − x̄)²
```

### Key takeaway
- Minimizing E gives a unique **(m, b)** pair — the equation of the best fit line.
- This approach of minimizing squared error is also connected to **gradient descent** — visualized as finding the minima on the cost surface J(θ₀, θ₁), where the bowl-shaped curve has one global minimum for m and one for b.

---

## 4. Code From Scratch

The notes reference implementing simple linear regression manually (without using `sklearn`), using the derived formulas for `m` and `b` directly from data — i.e., computing means, then applying the OLS formulas above to fit the line and make predictions.

---

## 5. Regression Metrics

Once a model is built, we need metrics to evaluate how good the predictions are.

### 5.1 MAE — Mean Absolute Error
Average of the absolute differences between actual and predicted values.

```
MAE = (1/n) Σ |yᵢ − ŷᵢ|
```
- Easy to interpret (same unit as target).
- Not differentiable at 0, less sensitive to outliers.

### 5.2 MSE — Mean Squared Error
Average of the squared differences between actual and predicted values.

```
MSE = (1/n) Σ (yᵢ − ŷᵢ)²
```
- Penalizes larger errors more heavily.
- Differentiable everywhere (useful for optimization).
- Unit is squared, harder to interpret directly.

### 5.3 RMSE — Root Mean Squared Error
Square root of MSE — brings error back to the original unit.

```
RMSE = √MSE = √[(1/n) Σ (yᵢ − ŷᵢ)²]
```

### 5.4 R² Score (Coefficient of Determination)
Measures how much of the variance in the target is explained by the model, relative to a baseline (mean) model.

```
R² = 1 − (SSres / SStot)

SSres = Σ (yᵢ − ŷᵢ)²        (residual sum of squares)
SStot = Σ (yᵢ − ȳ)²         (total sum of squares)
```
- R² close to 1 → model explains most of the variance.
- R² close to 0 → model performs no better than predicting the mean.
- R² can be negative if the model performs worse than the mean baseline.

### 5.5 Adjusted R² Score
R² always increases (or stays the same) as more features are added, even if they are irrelevant. Adjusted R² corrects this by penalizing the addition of non-useful features.

```
Adjusted R² = 1 − [ (1 − R²)(n − 1) / (n − k − 1) ]
```
Where:
- `n` = number of data points (samples)
- `k` = number of independent features

Adjusted R² is a more reliable metric than R² when comparing models with a different number of features.

---

## Summary

| Step | Concept |
|------|---------|
| 1 | Define model: `y = mx + b` |
| 2 | Define error function `E` (sum of squared residuals) |
| 3 | Minimize `E` via partial derivatives → closed-form `m`, `b` (OLS) |
| 4 | Implement from scratch |
| 5 | Evaluate using MAE, MSE, RMSE, R², Adjusted R² |
