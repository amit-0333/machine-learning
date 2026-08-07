# Multiple Linear Regression

## 1. Recap / Roadmap

- **Linear Regression** splits into two solution methods:
  - **OLS (Ordinary Least Squares)** — closed-form solution (this week)
  - **Gradient Descent** — iterative solution (next week)
- **MLR (Multiple Linear Regression)** topics planned across live sessions:
  1. Linear regression + gradient descent
  2. PCA

### Quick recap of Simple Linear Regression (SLR)
- 1 input (e.g. `exp` — years of experience) → 1 output (e.g. `salary`)

| exp (years) | salary (Rs) |
|---|---|
| 10 | 2,500,000 |
| 3  | 70,000 |
| 1  | 25,000 |

- Model: **y = mx + b**, where `m` = slope, `b` = intercept
- `x` → experience, `y` → salary

---

## 2. What is Multiple Linear Regression?

- **SLR**: 1 input → e.g. `exp | salary`
- **MLR**: more than 1 input (predictors) → e.g. `city | age | gender | edu`

> Simple Linear Regression and Multiple Linear Regression together fall under the umbrella of **Linear Regression**.

### Example dataset (1000 students)

| CGPA (x₁) | IQ (x₂) | Placement (LPA) |
|---|---|---|
| 8 | 80  | 8  |
| 9 | 90  | 9  |
| 5 | 120 | 15 |

- `x₁ → CGPA`, `x₂ → IQ` are the two input features feeding a **model** that outputs predicted **package (LPA)**.
- With 2 inputs, the data is 3-dimensional, and instead of fitting a **line**, MLR fits a **plane** (and for more than 2 inputs, a **hyperplane** in n-dimensional space).

### Equation progression

```
SLR:  y = mx + b

MLR (2 features):  y = β₀ + β₁x₁ + β₂x₂
MLR (3 features):  y = β₀ + β₁x₁ + β₂x₂ + β₃x₃
MLR (n features):  y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `β₀` corresponds to `b` (intercept)
- `β₁` corresponds to `m` (slope) — and generally, `β₁ ... βₙ` are the **coefficients**
- The full coefficient vector `β₀, β₁, ..., βₙ` plays the role that `(m, b)` played in SLR

**Dimensionality:** with 2 inputs and 1 output, this is 3D data (a plane); in general, n inputs + 1 output means fitting a hyperplane in n-dimensional coordinate space.

---

## 3. Mathematical Formulation (OLS Derivation)

### Setting up the predictions

For `m` columns (features) and `n` students (rows), each predicted value is:

```
ŷ₁ = β₀ + β₁x₁₁ + β₂x₁₂ + β₃x₁₃ + ... + βₘx₁ₘ
ŷ₂ = β₀ + β₁x₂₁ + β₂x₂₂ + ... + βₘx₂ₘ
  ⋮
ŷₙ = β₀ + β₁xₙ₁ + β₂xₙ₂ + ... + βₘxₙₘ
```

This system is written compactly in **matrix form**:

```
ŷ = Xβ
```

Where:
- `X` is the input matrix, shape **n × (m+1)** (a column of 1's prepended, for the β₀ intercept term)
- `β` is the coefficient vector, shape **(m+1) × 1**
- `ŷ` is the predicted output vector, shape **n × 1**

### Error / Loss function

Same idea as SLR — minimize the sum of squared residuals, now in matrix form:

```
E = Σᵢ (yᵢ − ŷᵢ)²         (scalar sum, i = 1 to n)
```

Written using the error vector `e = y − ŷ`:

```
eᵀe = (y₁−ŷ₁)² + (y₂−ŷ₂)² + ... + (yₙ−ŷₙ)² = Σ(yᵢ − ŷᵢ)²

E = eᵀe
```

### Expanding E

```
E = (y − ŷ)ᵀ(y − ŷ)
  = yᵀy − yᵀŷ − ŷᵀy + ŷᵀŷ
```

Using the identity **yᵀŷ = ŷᵀy** (both are 1×1 scalars, and a scalar equals its own transpose — proven separately using `(AᵀB)ᵀ = BᵀA` and the fact that `AᵀB` is symmetric when it's a scalar):

```
E = yᵀy − 2yᵀŷ + ŷᵀŷ                    ... (eqn 3)
```

Substituting `ŷ = Xβ`:

```
E = yᵀy − 2yᵀXβ + (Xβ)ᵀ(Xβ)
E = yᵀy − 2yᵀXβ + βᵀXᵀXβ                 ... (eqn 4)
```

### Minimizing E — take the derivative w.r.t. β

Goal: find the β vector that minimizes `E(β)` — same "bowl-shaped" loss curve idea as SLR, just in higher dimensions (slope = 0 at the minimum).

```
dE/dβ = 0
```

Using matrix calculus identities:
- `d(Aᵀx)/dx = A`  →  derivative of `−2yᵀXβ` w.r.t. β is `−2yᵀX`, rearranged to `−2XᵀY`
- `d(xᵀAx)/dx = 2xᵀA` (for symmetric A) → derivative of `βᵀXᵀXβ` w.r.t. β is `2βᵀXᵀX`

```
dE/dβ = 0 − 2yᵀX + 2βᵀXᵀX = 0
⇒ βᵀXᵀX = yᵀX
```

### Solving for β

```
βᵀXᵀX (XᵀX)⁻¹ = yᵀX (XᵀX)⁻¹
βᵀ · I = yᵀX(XᵀX)⁻¹
βᵀ = yᵀX(XᵀX)⁻¹
```

Taking the transpose of both sides (and using the fact that `(XᵀX)⁻¹` is symmetric, i.e. `[(XᵀX)⁻¹]ᵀ = (XᵀX)⁻¹`):

```
β = [(XᵀX)⁻¹]ᵀ Xᵀy
β = (XᵀX)⁻¹ Xᵀy
```

### ✅ Final Closed-Form OLS Solution

```
┌─────────────────────┐
│  β = (XᵀX)⁻¹ Xᵀ y    │
└─────────────────────┘
```

- `β` has shape **(m+1) × 1** — the values/coefficients
- `X` has shape **n × (m+1)**
- Dimension check: `(m+1)×n · n×(m+1) → (m+1)×(m+1)`, inverse stays `(m+1)×(m+1)`, times `(m+1)×n · n×1 → (m+1)×1` ✓ consistent with β's shape.

**Sanity-check example (from the notes):**
With features `exp | 12th marks`, target `salary`:
```
salary = exp·β₁ + (12th marks)·β₂ + β₀
```
if `β₀ = 0` for a particular fit, the plane passes through the origin along that axis.

---

## 4. Supporting Matrix-Algebra Identities Used

A few linear algebra facts the derivation leans on (also worked out in the notes):

- `(AB)ᵀ = BᵀAᵀ`
- `(Aᵀ)ᵀ = A`
- If `C = AᵀB` is a scalar (1×1), then `C = Cᵀ`, i.e. `AᵀB = BᵀA` for that case
- `(A⁻¹)ᵀ = (Aᵀ)⁻¹` — used to show `(XᵀX)⁻¹` is symmetric
- `d(xᵀAx)/dx = 2xᵀA` when `A` is symmetric — key step to differentiate `βᵀXᵀXβ`

---

## 5. Problem with the OLS Solution

Even though OLS gives a clean **closed-form** answer, it doesn't scale well:

```
β = (XᵀX)⁻¹ Xᵀy
```

- `X` has shape **n × (m+1)** (n rows/students, m+1 columns/features incl. intercept)
- Computing the **matrix inverse (XᵀX)⁻¹** has time complexity **O(n³)** (cubic in the number of features/dimensions of the square matrix being inverted)
- Example blow-up: with **100 input features**, `(XᵀX)` is a `100×100` matrix → inverse costs on the order of `(100)³ = 1,000,000` operations. With `10,000` features, that becomes `(10,000)³` — computationally very expensive.
- **This is why Gradient Descent exists**: it's an iterative alternative to OLS that avoids computing this expensive matrix inverse directly, and scales much better to large numbers of features.

| Method | Type | Scalability |
|---|---|---|
| OLS | Closed-form | Poor for large feature counts — O(n³) matrix inversion |
| Gradient Descent | Iterative | Scales much better, used in practice for large/high-dimensional data |

---
