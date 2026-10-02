# Ridge Regression: From Concept to the Math Behind the Formula

> **Goal:** understand *why* ridge regression exists, and see **exactly where the formula comes from**, one algebra step at a time. No step is skipped, and every number in this document was checked by running code.
>
> **Prerequisites:** basic linear regression, what a derivative is, and a little matrix notation (explained when it appears).

## Contents
1. [The 60-second version](#1-the-60-second-version)
2. [Why ridge exists](#2-why-ridge-exists)
3. [Notation](#3-notation)
4. [Recap: ordinary least squares (OLS) and its formula](#4-recap-ordinary-least-squares-ols-and-its-formula)
5. [Where OLS breaks](#5-where-ols-breaks)
6. [The ridge idea: punish big coefficients](#6-the-ridge-idea-punish-big-coefficients)
7. [Deriving the ridge formula step by step](#7-deriving-the-ridge-formula-step-by-step)
8. [Three more ways to arrive at the same formula](#8-three-more-ways-to-arrive-at-the-same-formula)
9. [The SVD view: what ridge really does to the data](#9-the-svd-view-what-ridge-really-does-to-the-data)
10. [Bias and variance of ridge (and why it can win)](#10-bias-and-variance-of-ridge-and-why-it-can-win)
11. [Choosing λ and practical rules](#11-choosing-λ-and-practical-rules)
12. [Ridge by gradient descent (weight decay)](#12-ridge-by-gradient-descent-weight-decay)
13. [Ridge vs Lasso vs OLS](#13-ridge-vs-lasso-vs-ols)
14. [Code (all tested)](#14-code-all-tested)
15. [Pitfalls and FAQ](#15-pitfalls-and-faq)
16. [Cheat sheet, exercises, glossary](#16-cheat-sheet-exercises-glossary)

> **Math display note:** formulas are written in plain text blocks so they read the same on any viewer. `Xᵀ` means "X transposed", `⁻¹` means "inverse", `‖v‖²` means "sum of squares of v's entries", `I` is the identity matrix.

---

## 1. The 60-second version

| | |
|---|---|
| **Problem** | Ordinary linear regression can produce huge, unstable coefficients when features are correlated or when you have many features and few rows. |
| **Idea** | Add a penalty on the **size of the coefficients** so the model prefers smaller ones. |
| **Loss** | `Σ(yᵢ − ŷᵢ)²  +  λ · Σβⱼ²` (squared error + λ × sum of squared coefficients) |
| **Formula** | `β̂ = (XᵀX + λI)⁻¹ Xᵀy` |
| **Effect of λ** | `λ = 0` → plain OLS. As `λ → ∞` → all coefficients → 0. |
| **Trade** | A little **bias** (coefficients shrunk) buys a lot less **variance** (more stable). |
| **Also called** | L2 regularisation, Tikhonov regularisation, weight decay. |

---

## 2. Why ridge exists

Three real situations where plain linear regression misbehaves:

1. **Overfitting.** With many features, OLS bends to fit noise. Train score is great, test score is poor. (Same overfitting idea as the bias–variance tradeoff.)
2. **Multicollinearity.** Two features carry almost the same information (e.g. height in cm and height in inches). OLS can't tell them apart, so it gives one a huge positive coefficient and the other a huge negative one that cancel out. Tiny changes in the data flip them wildly.
3. **More features than rows (p > n).** OLS has **no unique solution** at all, because the matrix `XᵀX` can't be inverted.

Ridge fixes all three with one change: **penalise large coefficients.**

---

## 3. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `n` | number of training rows | |
| `p` | number of features | |
| `X` | feature matrix (one row per example) | n × p |
| `y` | target vector | n × 1 |
| `β` | coefficient vector | p × 1 |
| `β₀` | intercept | scalar |
| `ε` | noise, mean 0, variance σ² | n × 1 |
| `λ` | ridge strength (λ ≥ 0). In scikit-learn it is called `alpha`. | scalar |
| `I` | identity matrix | p × p |

**Model:** `y = Xβ + ε`, so predictions are `ŷ = Xβ`.

**Simplification used throughout:** we first **centre** the data (subtract each column's mean from X, and the mean of y from y). Then the intercept disappears from the maths (§6 explains why this is legitimate), and we only need to find `β`.

---

## 4. Recap: ordinary least squares (OLS) and its formula

OLS chooses `β` to minimise the **residual sum of squares**:

```
RSS(β) = Σᵢ (yᵢ − xᵢᵀβ)²  =  ‖y − Xβ‖²  =  (y − Xβ)ᵀ(y − Xβ)
```

### Derivation of the OLS formula

**Step 1: expand.** Using `(a − b)ᵀ = aᵀ − bᵀ`, `(Xβ)ᵀ = βᵀXᵀ`:

```
RSS = yᵀy − yᵀXβ − βᵀXᵀy + βᵀXᵀXβ
```
`yᵀXβ` is a single number, so it equals its own transpose `βᵀXᵀy`. Combine them:

```
RSS = yᵀy − 2βᵀXᵀy + βᵀXᵀXβ
```

**Step 2: differentiate** with respect to β. Two standard rules of matrix calculus:

```
Rule 1:  ∂(cᵀβ)/∂β   = c
Rule 2:  ∂(βᵀAβ)/∂β  = 2Aβ        (when A is symmetric; XᵀX is symmetric)
```
So:
```
∂RSS/∂β = −2Xᵀy + 2XᵀXβ
```

**Step 3: set to zero** (minimum) and solve:

```
−2Xᵀy + 2XᵀXβ = 0
        XᵀXβ = Xᵀy                    ← the "normal equations"
           β̂ = (XᵀX)⁻¹ Xᵀy            ← the OLS formula
```

This needs `(XᵀX)⁻¹` to exist. That requirement is exactly where trouble starts.

---

## 5. Where OLS breaks

### 5.1 `XᵀX` may not be invertible
If one column is an exact combination of others (or `p > n`), then `XᵀX` is **singular**: its determinant is 0 and no inverse exists. There are infinitely many `β` that fit equally well.

### 5.2 Even when invertible, variance can explode
Since `y = Xβ + ε`, plug into the OLS formula:

```
β̂ = (XᵀX)⁻¹Xᵀ(Xβ + ε) = β + (XᵀX)⁻¹Xᵀε
```
So `β̂` equals the truth plus a noise term. Its covariance is:

```
Cov(β̂) = σ² (XᵀX)⁻¹
```

If features are nearly collinear, `XᵀX` has some **very small eigenvalues** `d²`. Its inverse then has **very large** entries (`1/d²`), so the variance `σ²/d²` becomes huge. That is why correlated features cause wild, unstable OLS coefficients.

**Need:** a way to stop `(XᵀX)⁻¹` from blowing up.

---

## 6. The ridge idea: punish big coefficients

Change the loss. Keep the squared error but **add a penalty on the squared size of the coefficients**:

```
J(β) = ‖y − Xβ‖²  +  λ‖β‖²
     = Σᵢ (yᵢ − xᵢᵀβ)²  +  λ Σⱼ βⱼ²
        └─── fit the data ──┘   └── keep β small ──┘
```

- `λ = 0`: no penalty, so we recover OLS.
- Large `λ`: keeping coefficients small matters more than fitting the data.
- The model now has to **justify** every unit of coefficient size by a matching improvement in fit.

### Two essential practical details

**(a) Do not penalise the intercept.** The intercept just sets the overall level of `y`. Shrinking it toward 0 would bias every prediction. The standard approach: centre `X` and `y`; solve for `β` on the centred data; then recover

```
β₀ = ȳ − x̄ᵀβ̂         (x̄ = vector of column means)
```
This is why the maths in §7 has no intercept.

**(b) Standardise the features first.** The penalty `Σβⱼ²` treats all coefficients alike, but a feature measured in thousands naturally needs a *tiny* coefficient and one measured in decimals needs a *big* one. Without standardising, ridge would unfairly shrink the "decimal" features. So scale every column to mean 0 and standard deviation 1 before fitting.

---

## 7. Deriving the ridge formula step by step

We minimise (on centred data):

```
J(β) = (y − Xβ)ᵀ(y − Xβ) + λ βᵀβ
```

**Step 1: expand.** From §4 the first term is `yᵀy − 2βᵀXᵀy + βᵀXᵀXβ`. Also `λβᵀβ = λ βᵀIβ`. So:

```
J(β) = yᵀy − 2βᵀXᵀy + βᵀXᵀXβ + λ βᵀIβ
     = yᵀy − 2βᵀXᵀy + βᵀ(XᵀX + λI)β
```
(The last line just groups the two quadratic terms.)

**Step 2: differentiate** using Rule 1 and Rule 2 from §4 (here `A = XᵀX + λI`, which is symmetric):

```
∂J/∂β = −2Xᵀy + 2(XᵀX + λI)β
```

**Step 3: set the gradient to zero:**

```
−2Xᵀy + 2(XᵀX + λI)β = 0
        (XᵀX + λI)β  = Xᵀy
```

**Step 4: solve for β** by multiplying both sides by the inverse:

```
┌───────────────────────────────┐
│  β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy   │
└───────────────────────────────┘
```

**That is the ridge formula.** Compare with OLS `(XᵀX)⁻¹Xᵀy`: the **only** difference is the `+ λI` added to `XᵀX`.

### 7.1 Why this inverse always exists (the "magic")
For any non-zero vector `v`:

```
vᵀ(XᵀX + λI)v = vᵀXᵀXv + λ vᵀv = ‖Xv‖² + λ‖v‖²
```
- `‖Xv‖² ≥ 0` (a squared length), and
- `λ‖v‖² > 0` whenever `λ > 0` and `v ≠ 0`.

So `vᵀ(XᵀX + λI)v > 0` for every `v ≠ 0`: the matrix is **positive definite**, therefore **always invertible**, even when `XᵀX` is singular or `p > n`. Ridge gives a unique answer in situations where OLS has none.

Intuition: adding `λ` to the diagonal lifts every eigenvalue of `XᵀX` from `d²` to `d² + λ`, so none can be (near) zero.

### 7.2 Why it's really the minimum (not a maximum)
The second derivative (Hessian) is `2(XᵀX + λI)`, which we just showed is positive definite. So `J` is a **strictly convex bowl** with exactly one minimum, and our stationary point is it.

### 7.3 A hand-checkable example (one feature)
With one feature and centred data, `XᵀX = Σx²` and `Xᵀy = Σxy`, so the formula becomes

```
β̂ = Σxy / (Σx² + λ)
```
(Check from scratch: `J = Σ(y − βx)² + λβ²` → `dJ/dβ = −2Σx(y − βx) + 2λβ = 0` → `Σxy = β(Σx² + λ)`. ✓)

Data: `x = [1, 2, 3]`, `y = [2, 4, 5]`, so `Σxy = 2 + 8 + 15 = 25` and `Σx² = 1 + 4 + 9 = 14`.

| λ | β̂ = 25 / (14 + λ) | Comment |
|---|---|---|
| 0 | **1.7857** | OLS |
| 2 | **1.5625** | slightly shrunk |
| 14 | **0.8929** | exactly half of OLS (λ = Σx²) |
| → ∞ | → 0 | everything shrinks to zero |

Shrinkage is visible: **larger λ → smaller coefficient**, smoothly, never hitting exactly zero.

---

## 8. Three more ways to arrive at the same formula

These show that the formula isn't arbitrary: it appears from several independent angles.

### 8.1 As ordinary regression on "fake extra data"
Stack `p` extra rows under your data: features `√λ · I`, targets `0`.

```
X̃ = [  X   ]        ỹ = [ y ]
    [ √λ·I ]            [ 0 ]
```
Then plain OLS on `(X̃, ỹ)` minimises `‖y − Xβ‖² + ‖√λ·β‖² = ‖y − Xβ‖² + λ‖β‖²`, which is exactly the ridge loss. Check with the normal equations:

```
X̃ᵀX̃ = XᵀX + λI        X̃ᵀỹ = Xᵀy      →   β̂ = (XᵀX + λI)⁻¹Xᵀy   ✓
```
Interpretation: ridge is like adding `p` imaginary data points that all say "each coefficient should be near 0". It also explains the invertibility: `X̃` always has full column rank.

### 8.2 As a constrained problem (Lagrange multipliers) + geometry
Ridge is equivalent to:

```
minimise  ‖y − Xβ‖²      subject to   ‖β‖² ≤ t        (a "budget" t for coefficient size)
```
Lagrangian: `L(β, λ) = ‖y − Xβ‖² + λ(‖β‖² − t)`. Setting `∂L/∂β = 0` gives `−2Xᵀy + 2XᵀXβ + 2λβ = 0`, the same equation as §7. Each budget `t` corresponds to one `λ`: **small budget ↔ large λ**.

Geometry in 2 coefficients:

```
 β₂
  │        ╭────╮      ellipses = contours of RSS
  │      ╭─┤ ╭──┤      (centre = OLS solution)
  │     ╱  │╱ ● │
  │    │  ╱╲ ↖  │      circle  = budget ‖β‖² ≤ t
  │     ╲╱  ╲ ──╯
  │   ○ ← ridge solution: where the smallest ellipse
  │        just touches the circle
  └──────────────────── β₁
```
The ridge solution is the point on the circle closest (in RSS terms) to the OLS solution. Because a circle has **no corners**, the touching point almost never lies exactly on an axis, so coefficients shrink but don't become exactly zero. (Lasso's diamond has corners, which is why Lasso can zero things out.)

### 8.3 As a Bayesian estimate (a "prior belief" that coefficients are small)
Assume:
```
Likelihood:  y | β ~ Normal(Xβ, σ²I)
Prior:       β     ~ Normal(0, τ²I)       (coefficients are probably small)
```
By Bayes' rule, `posterior ∝ likelihood × prior`. Taking the negative log:

```
−log posterior = ‖y − Xβ‖² / (2σ²)  +  ‖β‖² / (2τ²)  +  const
```
Multiply by `2σ²` (doesn't change where the minimum is):

```
‖y − Xβ‖²  +  (σ²/τ²) ‖β‖²
```
That is the ridge loss with

```
λ = σ² / τ²
```
So **ridge = the most probable β (MAP estimate) if you believe coefficients are bell-curve-distributed around 0.** A strong belief (small `τ²`) means large `λ`. Noisy data (large `σ²`) also means larger `λ`: you trust the data less, so lean on the prior more.

---

## 9. The SVD view: what ridge really does to the data

This section explains *which* directions ridge shrinks. It is the clearest picture of ridge's behaviour.

Write the **singular value decomposition** of the centred `X`:

```
X = U D Vᵀ        U: n×p orthonormal columns,  V: p×p orthogonal,  D = diag(d₁ ≥ d₂ ≥ … ≥ d_p ≥ 0)
```
(Each `dᵢ` says how much the data varies along direction `vᵢ`; small `dᵢ` means little variation, i.e. nearly collinear directions.)

Then `XᵀX = V D² Vᵀ`, so `XᵀX + λI = V (D² + λI) Vᵀ` and its inverse is `V (D² + λI)⁻¹ Vᵀ`. Also `Xᵀy = V D Uᵀy`. Multiply:

```
OLS:    β̂   = V · diag( 1/dᵢ )         · Uᵀy
Ridge:  β̂_λ = V · diag( dᵢ/(dᵢ²+λ) )   · Uᵀy
```

Dividing ridge by OLS, direction by direction, shows the **shrinkage factor**:

```
          dᵢ²
fᵢ  =  ─────────          (between 0 and 1)
         dᵢ² + λ
```

- Direction with **large** `dᵢ²` (lots of variation, well determined): `fᵢ ≈ 1`, barely touched.
- Direction with **small** `dᵢ²` (nearly collinear, poorly determined): `fᵢ ≈ 0`, shrunk hard.

**Ridge automatically shrinks the unreliable directions the most.** That is exactly the right behaviour, because those directions are where OLS variance `σ²/dᵢ²` explodes.

### Effective degrees of freedom
The "number of parameters actually used" is

```
df(λ) = Σᵢ dᵢ² / (dᵢ² + λ)
```
It equals `p` when `λ = 0` and decreases smoothly toward 0 as `λ → ∞`. So ridge acts like a **smoothly dialled-down model complexity** (the complexity axis of the bias–variance curve).

---

## 10. Bias and variance of ridge (and why it can win)

Let `A = (XᵀX + λI)⁻¹`, so `β̂_λ = A Xᵀy` and `y = Xβ + ε`.

### 10.1 Bias
```
E[β̂_λ] = A XᵀXβ
```
Since `XᵀX = (XᵀX + λI) − λI`, we get `A XᵀX = I − λA`, so

```
E[β̂_λ] = β − λAβ        →        Bias = −λ (XᵀX + λI)⁻¹ β
```
For `λ > 0` this is **non-zero**: ridge is a *biased* estimator, always pulling toward 0. (OLS has zero bias.)

### 10.2 Variance
```
Cov(β̂_λ) = σ² · A XᵀX A
```
In the SVD basis the variance of direction `i` is

```
OLS:    σ² / dᵢ²
Ridge:  σ² · dᵢ² / (dᵢ² + λ)²
```
Ridge's is smaller for every direction when `λ > 0`, and the reduction is **dramatic** when `dᵢ` is small (the collinear directions).

### 10.3 Total error and why a little ridge always helps
Total squared error of the coefficients (writing `αᵢ = vᵢᵀβ`, the true coefficient along direction i):

```
E‖β̂_λ − β‖²  =  σ² Σ dᵢ²/(dᵢ²+λ)²   +   λ² Σ αᵢ²/(dᵢ²+λ)²
                 └─── variance ───┘       └──── bias² ────┘
```

Differentiate at `λ = 0`:

- Variance term: derivative = `−2σ² Σ 1/dᵢ⁴`, which is **negative** (variance falls as λ grows).
- Bias² term: it contains `λ²`, so its derivative at 0 is **0** (bias starts to grow only gently).

So at `λ = 0` the total error is **decreasing**. Hence **there always exists some λ > 0 whose expected error is strictly lower than OLS** (the theorem of Hoerl & Kennard, 1970). Ridge isn't just a hack: a bit of bias is provably worth paying.

```
error
  │\                         total (variance + bias²)
  │ \        ___            ╱
  │  \  ____╱   ╲___    ___╱
  │   ╲╱             ╲╱╱   ← bias² (rises with λ)
  │    ╲_________________
  │       variance (falls with λ)
  └──────────────────────────── λ  (more regularisation →)
     OLS     best λ      too much (underfit)
```
This is the same U-shape as in the bias–variance tradeoff, with **λ as the complexity knob (larger λ = simpler model)**.

---

## 11. Choosing λ and practical rules

| λ | Behaviour | Bias | Variance | Risk |
|---|---|---|---|---|
| 0 | OLS | none | highest | overfit / unstable |
| small | mild shrinkage | low | lower | |
| moderate | **sweet spot** | moderate | much lower | |
| huge | all β ≈ 0, predicts ≈ mean of y | high | ~0 | underfit |

**How to choose:** use **cross-validation** on the training data across a log-spaced grid (e.g. `10⁻³ … 10³`). Scikit-learn's `RidgeCV` does this efficiently. Never choose λ using the test set.

**Checklist**
1. Centre the target and standardise the features (`StandardScaler`, fit on training data only).
2. Don't penalise the intercept (libraries handle this).
3. Pick λ by cross-validation.
4. Report coefficients on the standardised scale if you want to compare their importance.

---

## 12. Ridge by gradient descent (weight decay)

When `p` is huge, inverting a `p × p` matrix is costly (about `O(p³)`), so we can use gradient descent. The gradient is `∇J = −2Xᵀ(y − Xβ) + 2λβ`, so with learning rate `η`:

```
β ← β − η[ −2Xᵀ(y − Xβ) + 2λβ ]
  = (1 − 2ηλ) β  +  2η Xᵀ(y − Xβ)
    └─ decay ─┘      └─ usual OLS update ─┘
```
Every step first **multiplies β by a number slightly below 1** (it "decays"), then takes the usual fitting step. This is why deep-learning libraries call L2 regularisation **weight decay**. Gradient descent converges to exactly the closed-form answer (verified numerically: difference ≈ 1e-13).

---

## 13. Ridge vs Lasso vs OLS

| | **OLS** | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|---|
| Penalty | none | `λ Σβⱼ²` | `λ Σ|βⱼ|` |
| Closed form? | yes | **yes** | no (needs iterative solver) |
| Coefficients | unbiased, can be huge | shrunk, **rarely exactly 0** | shrunk, **many exactly 0** |
| Feature selection | no | no | **yes** |
| Correlated features | unstable | **shares weight among them** | picks one, drops the rest |
| p > n | fails | **works** | works (selects ≤ n features) |
| Use when | few clean features | many features that all matter a bit | you believe only a few features matter |

(**Elastic Net** combines both penalties.)

---

## 14. Code (all tested)

Run the blocks in order in one notebook. They need `numpy`, `matplotlib`, `scikit-learn`.

### 14.1 Main code: from scratch, collinearity, SVD, path and CV

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1. Ridge from scratch (closed form) ----------
def ridge_fit(X, y, lam):
    """Return (intercept, coefficients). The intercept is NOT penalised."""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean                     # centre -> intercept drops out
    d = X.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)   # (X'X + lam I) b = X'y
    return y_mean - x_mean @ beta, beta

X, y = load_diabetes(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
b0, b = ridge_fit(Xs, y, lam=10.0)
sk = Ridge(alpha=10.0).fit(Xs, y)
print("max |mine - sklearn| =", np.abs(b - sk.coef_).max(), "| intercepts:", round(b0, 4), round(sk.intercept_, 4))

# ---------- 2. Tiny worked example: one feature, no intercept ----------
x = np.array([1., 2., 3.]); yy = np.array([2., 4., 5.])
for lam in [0, 2, 14]:
    print(f"lambda={lam:>2}: beta = {np.sum(x*yy) / (np.sum(x*x) + lam):.4f}")

# ---------- 3. Multicollinearity: OLS explodes, ridge stays calm ----------
rng = np.random.default_rng(0)
n = 50
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(scale=0.01, size=n)            # almost a copy of x1
Xc = np.c_[x1, x2]
yc = 3 * x1 + rng.normal(scale=1.0, size=n)         # truth: 3*x1 + 0*x2
print("cond(X'X) =", f"{np.linalg.cond(Xc.T @ Xc):.1e}")
print("OLS  coefs:", LinearRegression().fit(Xc, yc).coef_.round(2))
print("Ridge coefs:", Ridge(alpha=1.0).fit(Xc, yc).coef_.round(2))

# ---------- 4. Variance reduction by simulation ----------
est_ols, est_ridge = [], []
for _ in range(1000):
    x1 = rng.normal(size=n); x2 = x1 + rng.normal(scale=0.05, size=n)
    Xm = np.c_[x1, x2]; ym = 3 * x1 + rng.normal(size=n)
    est_ols.append(LinearRegression().fit(Xm, ym).coef_[0])
    est_ridge.append(Ridge(alpha=5.0).fit(Xm, ym).coef_[0])
print(f"coef of x1 (true=3): OLS mean={np.mean(est_ols):.2f} std={np.std(est_ols):.2f} | "
      f"Ridge mean={np.mean(est_ridge):.2f} std={np.std(est_ridge):.2f}")

# ---------- 5. SVD view: shrinkage factors and effective degrees of freedom ----------
Xcen = Xs - Xs.mean(axis=0)
U, dvals, Vt = np.linalg.svd(Xcen, full_matrices=False)
lam = 10.0
beta_svd = Vt.T @ ((dvals / (dvals**2 + lam)) * (U.T @ (y - y.mean())))
print("SVD formula matches closed form:", np.allclose(beta_svd, b))
print("shrink factors d^2/(d^2+lam):", np.round(dvals**2 / (dvals**2 + lam), 3))
print("effective df at lam=10:", round(np.sum(dvals**2 / (dvals**2 + lam)), 2), "of", Xs.shape[1])

# ---------- 6. Ridge path + choosing lambda by cross-validation ----------
lams = np.logspace(-2, 4, 60)
path = np.array([ridge_fit(Xs, y, l)[1] for l in lams])
plt.semilogx(lams, path); plt.xlabel("lambda"); plt.ylabel("coefficient")
plt.title("Ridge path: coefficients shrink smoothly toward 0"); plt.show()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
sc = StandardScaler().fit(X_tr); A, B = sc.transform(X_tr), sc.transform(X_te)
ols = LinearRegression().fit(A, y_tr)
cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(A, y_tr)
for name, m in [("OLS", ols), (f"Ridge (alpha={cv.alpha_:.2f})", cv)]:
    print(f"{name:24s} test RMSE={mean_squared_error(y_te, m.predict(B))**.5:.2f}  R2={r2_score(y_te, m.predict(B)):.3f}")
```

### 14.2 Where ridge really shines: many features, few samples

```python
# ---------- 7. Where ridge really shines: many features, few samples ----------
rng = np.random.default_rng(3)
n_tr, n_te, d = 40, 1000, 30
beta_true = rng.normal(size=d) * (rng.random(d) < 0.3)       # sparse-ish truth
def make(n):
    Z = rng.normal(size=(n, d)); return Z, Z @ beta_true + rng.normal(scale=2.0, size=n)
Z_tr, z_tr = make(n_tr); Z_te, z_te = make(n_te)
ols = LinearRegression().fit(Z_tr, z_tr)
rcv = RidgeCV(alphas=np.logspace(-2, 3, 40), cv=5).fit(Z_tr, z_tr)
for name, m in [("OLS", ols), (f"Ridge (alpha={rcv.alpha_:.1f})", rcv)]:
    print(f"{name:22s} train R2={m.score(Z_tr, z_tr):.2f}  test R2={m.score(Z_te, z_te):.2f}")
```

### 14.3 What the outputs show

**Block 1: my formula equals scikit-learn's**
```
max |mine - sklearn| = 1.67e-14 | intercepts: 152.1335 152.1335
```
Our 3-line implementation of `(XᵀX + λI)⁻¹Xᵀy` matches the library to machine precision.

**Block 2: the hand example (§7.3)**
```
lambda= 0: beta = 1.7857      lambda= 2: beta = 1.5625      lambda=14: beta = 0.8929
```
Matches the table above.

**Block 3: multicollinearity** (x2 is x1 plus tiny noise; truth is `3·x1 + 0·x2`)
```
cond(X'X) = 3.4e+04
OLS  coefs: [-9.26 12.29]
Ridge coefs: [1.46 1.51]
```
OLS gives wild opposite-sign coefficients, yet notice `−9.26 + 12.29 ≈ 3.03`: the **sum** is well determined, but the *split* between the twin features is not. Ridge splits the effect evenly (`1.46 + 1.51 ≈ 3`), a stable and sensible answer.

**Block 4: variance reduction by simulation (1000 repeated datasets)**
```
coef of x1 (true=3): OLS mean=2.98 std=2.83 | Ridge mean=1.44 std=0.08
```
- OLS is **unbiased** (mean ≈ 3) but its estimate swings by ±2.83 between datasets.
- Ridge is **biased** (mean 1.44, because it shares the effect with x2) but extremely **stable** (std 0.08).
- This is the bias–variance tradeoff, measured directly.

**Block 5: SVD view**
```
SVD formula matches closed form: True
shrink factors d^2/(d^2+lam): [0.994 0.985 0.982 0.977 0.967 0.964 0.96 0.95 0.776 0.275]
effective df at lam=10: 8.83 of 10
```
Directions with large singular values keep ~99% of their coefficient. The weakest direction is cut to **27.5%**. The 10 parameters behave like **8.83**.

**Block 6: ridge path and test comparison (diabetes dataset)**
```
OLS                      test RMSE=53.88  R2=0.444
Ridge (alpha=33.93)      test RMSE=54.18  R2=0.438
```
Here ridge **does not beat** OLS, and that's expected: there are 442 rows for only 10 mostly independent features, so OLS is already stable. Ridge is not magic. It pays off when there is instability to fix.

**Block 7: instability present** (40 rows, 30 features)
```
OLS                    train R2=0.94  test R2=0.04
Ridge (alpha=21.5)     train R2=0.72  test R2=0.44
```
OLS memorises the training set (0.94) and collapses on new data (0.04). Ridge gives up some training fit and generalises about **11× better**. This is the situation ridge is made for.

---

## 15. Pitfalls and FAQ

**Why does ridge need standardised features?**
The penalty is not scale-invariant. Rescaling a feature changes how much its coefficient is penalised, so results would depend on your units.

**Does ridge select features?**
No. Coefficients get small but stay non-zero. Use Lasso or Elastic Net to zero them out.

**Why isn't the intercept penalised?**
It only sets the baseline level of `y`. Penalising it would make predictions depend on where your target happens to be centred.

**λ vs `alpha` vs `C`?**
Scikit-learn's `Ridge(alpha=…)` is the same λ used here. In `LogisticRegression` and `SVC`, the parameter `C` is the **inverse** (`C = 1/λ`, roughly).

**Is the `(XᵀX + λI)⁻¹` formula how libraries compute it?**
Mostly they avoid forming the inverse explicitly (it's numerically fragile) and use Cholesky, SVD or iterative solvers. Our code uses `np.linalg.solve`, which solves the system directly.

**Can ridge make things worse?**
Yes: too large a λ underfits (predictions collapse toward the mean of y), and on clean low-dimensional data the benefit may be ~0 (Block 6).

**Does it work for classification?**
The same L2 penalty is used in logistic regression, SVMs and neural networks (as weight decay), though the loss differs and there is no closed form for logistic regression.

---

## 16. Cheat sheet, exercises, glossary

### Cheat sheet
```
Loss                 J(β) = ‖y − Xβ‖² + λ‖β‖²
Closed form          β̂ = (XᵀX + λI)⁻¹ Xᵀy             ; intercept β₀ = ȳ − x̄ᵀβ̂
Why invertible       XᵀX is PSD, +λI makes it positive definite
SVD shrinkage        factor dᵢ²/(dᵢ²+λ) on direction i
Effective df         Σ dᵢ²/(dᵢ²+λ)
Bias                 −λ(XᵀX+λI)⁻¹β          Cov = σ²·A·XᵀX·A, A=(XᵀX+λI)⁻¹
Bayesian view        Normal prior on β  ⇒  λ = σ²/τ²
Gradient step        β ← (1 − 2ηλ)β + 2ηXᵀ(y − Xβ)    (weight decay)
```

### Exercises
1. By hand: for `x = [1, 2, 3]`, `y = [2, 4, 5]`, find the λ that makes `β̂ = 1.0`. *(Answer: 25/(14+λ) = 1 → λ = 11.)*
2. Prove `E[β̂_λ] = β − λ(XᵀX + λI)⁻¹β` yourself using `A XᵀX = I − λA`.
3. Show ridge fits the `(X̃, ỹ)` augmented-data trick on a 2-feature example using `np.linalg.lstsq`.
4. Plot test error against `log₁₀(λ)` for the 40 × 30 dataset in §14.2. Where is the minimum?
5. Verify the shrinkage factor `dᵢ²/(dᵢ²+λ)` by computing OLS and ridge coefficients in the SVD basis.
6. Repeat Block 4 with `x2 = x1 + noise(scale=1.0)`. How does the OLS std change as collinearity drops?

### Glossary
| Term | Meaning |
|---|---|
| **Regularisation** | Adding a penalty to discourage overly complex models. |
| **L2 norm** | `‖β‖ = √Σβⱼ²`; ridge penalises its square. |
| **Multicollinearity** | Features strongly correlated with each other. |
| **Singular matrix** | A matrix with no inverse (determinant 0). |
| **Positive definite** | `vᵀMv > 0` for all `v ≠ 0`; guarantees invertibility. |
| **SVD** | Decomposition `X = UDVᵀ` revealing the main directions of variation. |
| **MAP estimate** | Most probable parameter value given data **and** a prior. |
| **Effective degrees of freedom** | Smoothly-measured number of parameters a regularised model uses. |
| **Weight decay** | Gradient-descent name for L2 regularisation. |
| **Cross-validation** | Estimating generalisation by training on folds and testing on the held-out parts. |
