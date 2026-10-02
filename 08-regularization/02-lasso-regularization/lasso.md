# Lasso Regression: From Concept to the Math Behind the Solution

> **Goal:** understand why Lasso exists, why it can set coefficients to **exactly zero**, and where its solution comes from, including the key fact that **Lasso has no closed-form formula** like ridge does and what we do instead. Every number here was checked by running code.
>
> **Prerequisites:** read the **Ridge guide** first (Lasso reuses its notation, OLS recap and ideas). Basic derivatives and a little matrix notation are enough.

## Contents
1. [The 60-second version](#1-the-60-second-version)
2. [Why Lasso exists](#2-why-lasso-exists)
3. [Notation and the Lasso loss](#3-notation-and-the-lasso-loss)
4. [Why there is no closed-form formula](#4-why-there-is-no-closed-form-formula)
5. [Deriving the solution: one feature](#5-deriving-the-solution-one-feature)
6. [Optimality conditions for many features (subgradients)](#6-optimality-conditions-for-many-features-subgradients)
7. [Coordinate descent: how Lasso is actually solved](#7-coordinate-descent-how-lasso-is-actually-solved)
8. [Why zeros? Two pictures](#8-why-zeros-two-pictures)
9. [The Bayesian view (Laplace prior)](#9-the-bayesian-view-laplace-prior)
10. [The Lasso path and λ_max](#10-the-lasso-path-and-λ_max)
11. [Strengths and weaknesses](#11-strengths-and-weaknesses)
12. [Choosing λ and practical rules](#12-choosing-λ-and-practical-rules)
13. [Ridge vs Lasso side by side](#13-ridge-vs-lasso-side-by-side)
14. [Code (all tested)](#14-code-all-tested)
15. [Pitfalls and FAQ](#15-pitfalls-and-faq)
16. [Cheat sheet, exercises, glossary](#16-cheat-sheet-exercises-glossary)

> **Math display note:** formulas are in plain text blocks. `‖β‖₁` = sum of absolute values of the coefficients, `xⱼ` = the j-th column of X, `(·)₊` = "keep it if positive, otherwise 0".

---

## 1. The 60-second version

| | |
|---|---|
| **Name** | **L**east **A**bsolute **S**hrinkage and **S**election **O**perator |
| **Loss** | `Σ(yᵢ − ŷᵢ)²  +  λ · Σ|βⱼ|` (squared error + λ × sum of **absolute** coefficients) |
| **Key behaviour** | Shrinks coefficients **and sets some exactly to 0**, so it performs **automatic feature selection**. |
| **Formula?** | **No closed form.** Solved iteratively (coordinate descent). |
| **Core building block** | The **soft-thresholding** operator `S(z, t) = sign(z)·max(|z| − t, 0)` |
| **Use when** | Many features, you believe only a few really matter, and you want an interpretable sparse model. |
| **Weak spot** | Among correlated features it keeps one more or less arbitrarily. (Elastic Net fixes this.) |

---

## 2. Why Lasso exists

Ridge shrinks all coefficients but **never removes a feature**. With hundreds of features that leaves a model that is stable but hard to interpret: every feature still has a say.

Often we want the opposite: *"find the few features that actually matter and ignore the rest."* A model with 8 non-zero coefficients out of 500 is easier to understand, cheaper to run and often predicts better when most features are noise.

Lasso gets this by changing **one thing** in ridge: penalise `|β|` instead of `β²`. That small change makes some coefficients land exactly on zero.

---

## 3. Notation and the Lasso loss

Same symbols as the Ridge guide: `X` is n × p, `y` is n × 1, `β` is p × 1, `λ ≥ 0` is the strength. We again **centre** `X` and `y` so the intercept drops out (and is not penalised), and we **standardise** the features.

```
J(β) = ‖y − Xβ‖²  +  λ‖β‖₁
     = Σᵢ (yᵢ − xᵢᵀβ)²  +  λ Σⱼ |βⱼ|
```

| | Ridge | Lasso |
|---|---|---|
| Penalty | `λ Σ βⱼ²` | `λ Σ |βⱼ|` |
| Penalty shape (1 coefficient) | smooth parabola `U` | sharp `V` with a **corner at 0** |

> **Library warning:** scikit-learn minimises `(1/2n)‖y − Xβ‖² + α‖β‖₁`. Our λ and its `alpha` are linked by **`λ = 2n·α`**. (Verified in the code below.)

---

## 4. Why there is no closed-form formula

For ridge we differentiated the loss, set the gradient to zero, and solved. For Lasso this breaks at the first step: **`|β|` has no derivative at β = 0** (its graph has a sharp corner).

```
 |β|
  │\     /
  │ \   /       slope is −1 on the left, +1 on the right,
  │  \ /        and undefined exactly at 0
──┴───●───── β
```

Even where it is differentiable, the derivative is `sign(β)`, which makes the stationarity equation `−2Xᵀ(y − Xβ) + λ·sign(β) = 0` a non-linear system with no neat matrix solution. So Lasso is solved **numerically**. We'll get exact solutions for one feature (§5), exact *conditions* for many (§6), and an algorithm that uses the one-feature result repeatedly (§7).

---

## 5. Deriving the solution: one feature

With one centred feature `x`, define the two numbers

```
z = Σ xᵢyᵢ  (= xᵀy)         s = Σ xᵢ²  (= xᵀx)
```

The loss as a function of the single coefficient β is

```
J(β) = Σ(y − βx)² + λ|β|  =  Σy² − 2βz + β²s + λ|β|
```

Split into cases by the sign of β:

**Case β > 0** (so |β| = β):
```
dJ/dβ = −2z + 2sβ + λ = 0   →   β = (z − λ/2) / s        valid only if this is > 0, i.e. z > λ/2
```

**Case β < 0** (so |β| = −β):
```
dJ/dβ = −2z + 2sβ − λ = 0   →   β = (z + λ/2) / s        valid only if this is < 0, i.e. z < −λ/2
```

**Otherwise** (`−λ/2 ≤ z ≤ λ/2`) neither case has a valid solution, so the minimum is **at the corner, β = 0**.

Combine all three cases into one line using the **soft-thresholding operator**

```
S(z, t) = sign(z) · max(|z| − t, 0)
```
```
┌──────────────────────────────┐
│   β̂_lasso = S(z, λ/2) / s    │
└──────────────────────────────┘
```

In words: take the ordinary least-squares quantity `z`, **subtract λ/2 from its size** (a constant "tax"), and if that pushes it past zero, **clamp it to exactly 0**.

### Compare with ridge (same one-feature setting)
```
Ridge:  β̂ = z / (s + λ)               ← shrinks by a PROPORTION (divides), never reaches 0
Lasso:  β̂ = S(z, λ/2) / s             ← shrinks by a CONSTANT (subtracts), hits 0 once the tax exceeds |z|
```

```
 β̂                              Lasso: parallel to OLS line, shifted down, flat at 0
  │        OLS (y = x)          Ridge: a flatter line through the origin
  │       ╱  ╱ Ridge
  │      ╱ ╱╱
  │     ╱╱╱  Lasso
──┼───●───────────────── z
  │ dead zone: −λ/2 … λ/2  →  β̂ = 0
```

### Hand-checkable example
Data `x = [1, 2, 3]`, `y = [2, 4, 5]`: `z = 25`, `s = 14` (same as the ridge guide). OLS gives `25/14 = 1.7857`.

| λ | λ/2 | β̂ = S(25, λ/2) / 14 | Comment |
|---|---|---|---|
| 0 | 0 | **1.7857** | OLS |
| 10 | 5 | **1.4286** | (25 − 5)/14, shrunk |
| 50 | 25 | **0** | tax equals |z|, so the coefficient is exactly zero |
| 60 | 30 | **0** | stays at zero |

Ridge never got to zero for any finite λ (§7.3 of the ridge guide). Lasso reaches it at **λ = 2|z| = 50**.

---

## 6. Optimality conditions for many features (subgradients)

For many features we cannot write one formula, but we can write exact **conditions** that the answer must satisfy. Because `|β|` has a corner, we use its **subgradient**: at β = 0 the "slope" can be *any value between −1 and +1*.

```
∂|β|/∂β  =  +1          if β > 0
            −1          if β < 0
            any v ∈ [−1, 1]   if β = 0
```

Differentiating `J(β)` with respect to `βⱼ` and setting it to zero ("0 must be in the subgradient"):

```
−2 xⱼᵀ(y − Xβ)  +  λ·vⱼ  =  0       where  vⱼ = sign(βⱼ) if βⱼ ≠ 0, otherwise vⱼ ∈ [−1, 1]
```

Let `cⱼ = 2 xⱼᵀ(y − Xβ)` be "how strongly feature j still correlates with the current residual." Then:

| If… | Then… |
|---|---|
| `βⱼ ≠ 0` | `cⱼ = λ·sign(βⱼ)` → the correlation sits **exactly at the threshold λ** |
| `βⱼ = 0` | requires `|cⱼ| ≤ λ` → the feature isn't correlated enough with what's left to be worth its penalty |

> **Plain-English rule:** a feature is allowed into the model only if it explains the leftover error by more than the penalty λ. Otherwise its coefficient is exactly 0.

---

## 7. Coordinate descent: how Lasso is actually solved

Idea: **optimise one coefficient at a time**, holding the others fixed, and cycle until nothing changes. Each one-coefficient problem is exactly the one-feature problem from §5.

For coefficient `j`, remove its own contribution to get the **partial residual** `r⁽ʲ⁾ = y − Σ_{k≠j} xₖβₖ`. The loss as a function of `βⱼ` alone is a one-feature problem with `z = xⱼᵀr⁽ʲ⁾` and `s = xⱼᵀxⱼ`. By §5:

```
βⱼ ← S( xⱼᵀ r⁽ʲ⁾ ,  λ/2 ) / ( xⱼᵀ xⱼ )
```

**Algorithm**
```
1. Start with β = 0.
2. Repeat until β stops changing:
     for j = 1 … p:
         r  = y − Σ_{k≠j} xₖβₖ            (partial residual)
         βⱼ = S(xⱼᵀ r, λ/2) / (xⱼᵀxⱼ)    (soft-threshold update)
```
Because the loss is convex, this is guaranteed to converge to the global minimum. It's what scikit-learn's `Lasso` uses, and it's cheap: each update costs `O(n)`. The code in §14 implements it in about 12 lines and matches scikit-learn to ~1e-10.

---

## 8. Why zeros? Two pictures

### 8.1 The constraint picture (diamond vs circle)
Lasso is equivalent to: minimise `‖y − Xβ‖²` subject to `‖β‖₁ ≤ t`. For two coefficients the allowed region is a **diamond**; for ridge it's a **circle**.

```
        β₂                            β₂
         │   ╱╲                        │   ╭──╮
  ellipse│ ╱    ╲                      │ ╭╯    ╰╮
  touches│╱  ●   ╲ ← corner on         │ │      │   smooth circle: touching point
   ──────┼─────────── β₁               ┼─┼──────┼─  almost never exactly on an axis
         │╲      ╱   an axis ⇒ β₁ = 0  │ ╰╮    ╭╯
         │  ╲  ╱                       │   ╰──╯
        Lasso (diamond)               Ridge (circle)
```
The RSS contours (ellipses) grow outward from the OLS solution until they first touch the allowed region. A diamond's **corners lie on the axes**, and in high dimensions the ellipse almost always touches a corner or an edge. When it touches a corner, some coefficients are exactly 0. A circle has no corners, so ridge doesn't produce zeros.

### 8.2 The slope picture (why a constant push beats a shrinking push)
Ridge's penalty `λβ²` has slope `2λβ`, which **fades to 0 as β → 0**, so near zero ridge barely pushes. Lasso's penalty `λ|β|` has a **constant slope of ±λ no matter how small β is**, so it keeps pushing all the way to 0, and a coefficient that isn't pulled hard enough by the data gets stuck there. That's exactly the "dead zone" in the picture from §5.

---

## 9. The Bayesian view (Laplace prior)

Same recipe as in the Ridge guide, but with a different prior.

```
Likelihood:  y | β ~ Normal(Xβ, σ²I)
Prior:       each βⱼ ~ Laplace(0, b)   i.e.  p(βⱼ) ∝ exp(−|βⱼ| / b)
```
(The Laplace distribution is a pointy "double-exponential" spike at 0 with heavy tails: it expects *many coefficients to be tiny and a few to be large*.)

Negative log posterior, then multiply by `2σ²`:

```
−log posterior = ‖y − Xβ‖²/(2σ²)  +  ‖β‖₁/b  +  const
×2σ²   →   ‖y − Xβ‖²  +  (2σ²/b)‖β‖₁
```
So Lasso is the **MAP estimate under a Laplace prior**, with

```
λ = 2σ² / b
```
| Prior | Shape at 0 | Estimator |
|---|---|---|
| Gaussian | smooth bell | Ridge |
| Laplace | **sharp spike** | Lasso |

The sharp spike is the Bayesian reason for exact zeros.

---

## 10. The Lasso path and λ_max

As λ increases from 0, coefficients shrink and drop out **one by one**, and the path in between is piecewise linear. Plotting coefficients against λ gives the **Lasso path**: at the left everything is non-zero (like OLS), at the right everything is 0.

**The smallest λ that makes everything zero.** From §6, all coefficients are zero iff `|2xⱼᵀy| ≤ λ` for every j (with β = 0 the residual is just y). So

```
λ_max = 2 · max_j |xⱼᵀ y|          (on centred data)
```
Above `λ_max` the model is just "predict the mean". CV grids usually run from `λ_max` down to about `λ_max/1000` on a log scale.

---

## 11. Strengths and weaknesses

**Strengths**
- **Automatic feature selection** → sparse, interpretable models.
- Works when `p > n` (selects at most `n` features).
- Often better than ridge when only a few features are truly relevant.
- Fast solvers for very large problems.

**Weaknesses**
- **Correlated features:** Lasso tends to keep **one** from a correlated group and drop the others, and *which one* can flip from one dataset to the next (shown in §14, block 5).
- **Extra shrinkage bias:** big, truly important coefficients are also pulled down by the same constant tax (compare the recovered `2.72` vs true `3` in the code).
- **At most n non-zero coefficients** when `p > n`.
- **CV-chosen λ often keeps a few extra noise features.** Selection is not perfect.
- When many features matter *a little each*, ridge usually predicts better.

---

## 12. Choosing λ and practical rules

1. **Standardise features** (mean 0, std 1; fit the scaler on training data only). The penalty `Σ|β|` is not scale-invariant.
2. **Don't penalise the intercept** (libraries handle this).
3. Pick λ by **cross-validation** over a log-spaced grid: `LassoCV`.
4. **One-standard-error rule:** choose the *largest* λ whose CV error is within 1 standard error of the best, giving a sparser model with similar accuracy.
5. Remember sklearn's `alpha` ≠ our λ: `λ = 2n·alpha`.

---

## 13. Ridge vs Lasso side by side

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | `λΣβ²` | `λΣ|β|` |
| Solution | closed form `(XᵀX + λI)⁻¹Xᵀy` | iterative (coordinate descent) |
| One-feature solution | `z / (s + λ)` (divide) | `S(z, λ/2) / s` (subtract & clamp) |
| Exact zeros | **no** | **yes** |
| Feature selection | no | **yes** |
| Correlated features | spreads weight evenly | picks one, unstable |
| p > n | works, keeps all p | works, keeps ≤ n |
| Prior | Gaussian | Laplace |
| Geometry | circle | diamond |
| Best when | many small effects | few strong effects |

---

## 14. Code (all tested)

Run the blocks in order in one notebook. They need `numpy`, `matplotlib`, `scikit-learn`.

### 14.1 Soft-thresholding and coordinate descent from scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV, lasso_path
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# ---------- 1. Soft-thresholding + coordinate descent, from scratch ----------
def soft_threshold(z, t):
    """S(z, t) = sign(z) * max(|z| - t, 0)"""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso_cd(X, y, lam, n_iter=2000, tol=1e-10):
    """Minimise ||y - X b||^2 + lam * ||b||_1  (intercept unpenalised)."""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    p = Xc.shape[1]
    beta = np.zeros(p)
    col_sq = (Xc ** 2).sum(axis=0)
    r = yc.copy()                                   # current residual
    for _ in range(n_iter):
        beta_old = beta.copy()
        for j in range(p):
            r += Xc[:, j] * beta[j]                 # remove feature j's contribution
            beta[j] = soft_threshold(Xc[:, j] @ r, lam / 2) / col_sq[j]
            r -= Xc[:, j] * beta[j]                 # put the updated contribution back
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return y_mean - x_mean @ beta, beta

X, y = load_diabetes(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
n = len(y); lam = 2000.0
b0, b = lasso_cd(Xs, y, lam)
sk = Lasso(alpha=lam / (2 * n), tol=1e-12, max_iter=100000).fit(Xs, y)   # sklearn's alpha = lam / (2n)
print("max |mine - sklearn| =", f"{np.abs(b - sk.coef_).max():.2e}")
print("coefficients:", b.round(2))
print("exact zeros:", int(np.sum(b == 0)), "of", len(b))

# ---------- 2. One-feature example, no intercept ----------
x = np.array([1., 2., 3.]); yy = np.array([2., 4., 5.])
z, s = np.sum(x * yy), np.sum(x * x)
for lam1 in [0, 10, 50, 60]:
    mine = soft_threshold(z, lam1 / 2) / s
    skl = Lasso(alpha=lam1 / (2 * len(x)) if lam1 else 1e-12, fit_intercept=False, tol=1e-12).fit(x.reshape(-1, 1), yy).coef_[0]
    print(f"lambda={lam1:>2}: beta = {mine:.4f}  (sklearn {skl:.4f})")
```

### 14.2 Sparse recovery, λ_max, correlated twins, path, real data

```python
# ---------- 3. Sparse recovery: 5 real features hidden among 50 ----------
rng = np.random.default_rng(0)
n_tr, n_te, p = 100, 2000, 50
beta_true = np.zeros(p); beta_true[:5] = [3, -2, 1.5, 4, -3]
def make(n):
    Z = rng.normal(size=(n, p)); return Z, Z @ beta_true + rng.normal(size=n)
Z_tr, z_tr = make(n_tr); Z_te, z_te = make(n_te)

ols   = LinearRegression().fit(Z_tr, z_tr)
ridge = RidgeCV(alphas=np.logspace(-2, 3, 40), cv=5).fit(Z_tr, z_tr)
lasso = LassoCV(cv=5, random_state=0).fit(Z_tr, z_tr)
print(f"{'model':8s} {'test R2':>8s} {'non-zero coefs':>15s}")
for name, m in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    print(f"{name:8s} {m.score(Z_te, z_te):8.3f} {int(np.sum(np.abs(m.coef_) > 1e-8)):15d}")
print("Lasso picked features:", np.flatnonzero(lasso.coef_))
print("Lasso coefs on true 5:", lasso.coef_[:5].round(2), "(truth", beta_true[:5], ")")

# ---------- 4. The smallest lambda that zeroes everything ----------
Zc, zc = Z_tr - Z_tr.mean(0), z_tr - z_tr.mean()
lam_max = 2 * np.max(np.abs(Zc.T @ zc))
print("lambda_max =", round(lam_max, 1))
print("non-zeros at lam_max      :", int(np.sum(lasso_cd(Z_tr, z_tr, lam_max * 1.0001)[1] != 0)))
print("non-zeros at 0.9*lam_max  :", int(np.sum(lasso_cd(Z_tr, z_tr, lam_max * 0.9)[1] != 0)))

# ---------- 5. Correlated twins: lasso is unstable about WHICH twin it keeps ----------
L, R = [], []
for _ in range(500):
    x1 = rng.normal(size=100); x2 = x1 + rng.normal(scale=0.05, size=100); x3 = rng.normal(size=100)
    Xt = np.c_[x1, x2, x3]; yt = 3 * x1 + 3 * x2 + 0.5 * x3 + rng.normal(size=100)   # both twins matter equally
    L.append(Lasso(alpha=1.5).fit(Xt, yt).coef_[:2]); R.append(Ridge(alpha=5).fit(Xt, yt).coef_[:2])
L, R = np.array(L), np.array(R)
print("Lasso: x1 coef mean/std =", L[:, 0].mean().round(2), "/", L[:, 0].std().round(2),
      "| one twin exactly 0 in", f"{np.mean((L[:, 0] == 0) | (L[:, 1] == 0)):.0%}", "of runs")
print("Ridge: x1 coef mean/std =", R[:, 0].mean().round(2), "/", R[:, 0].std().round(2))

# ---------- 6. Lasso path ----------
alphas, coefs, _ = lasso_path(Z_tr - Z_tr.mean(0), z_tr - z_tr.mean(), n_alphas=60)
for j in range(p):
    plt.semilogx(alphas, coefs[j], color="red" if j < 5 else "gray", lw=1.5 if j < 5 else .6)
plt.xlabel("alpha (lambda)"); plt.ylabel("coefficient")
plt.title("Lasso path: red = true features, gray = noise"); plt.show()

# ---------- 7. Real data: which diabetes features survive? ----------
cvm = LassoCV(cv=5, random_state=0).fit(Xs, y)
names = load_diabetes().feature_names
print(f"alpha chosen by CV = {cvm.alpha_:.3f}")
for nme, c in zip(names, cvm.coef_):
    print(f"  {nme:4s} {c:8.2f}" + ("   <- dropped" if c == 0 else ""))
```

### 14.3 What the outputs show

**Block 1: my solver equals scikit-learn's**
```
max |mine - sklearn| = 7.95e-11
exact zeros: 3 of 10
```
Our 12-line coordinate descent matches the library, and with `λ = 2000` on the diabetes data three coefficients are exactly zero.

**Block 2: the hand example (§5)**
```
lambda= 0: 1.7857   lambda=10: 1.4286   lambda=50: 0.0000   lambda=60: 0.0000
```
Matches the table, and scikit-learn agrees.

**Block 3: finding 5 real features among 50** (100 training rows)
```
model     test R2  non-zero coefs
OLS         0.950              50
Ridge       0.948              50
Lasso       0.968              15
Lasso coefs on true 5: [ 2.72 -1.82  1.39  3.78 -2.78]  (truth [3, -2, 1.5, 4, -3])
```
Lasso uses 15 features instead of 50 and predicts best. It found **all 5 true features**, but also kept 10 noise features (CV tends to over-select), and its estimates of the true coefficients are shrunk (2.72 vs 3).

**Block 4: λ_max (§10)**
```
lambda_max = 747.4
non-zeros at lam_max      : 0
non-zeros at 0.9*lam_max  : 2
```
At `λ_max` everything is 0; a bit below it, the two strongest features enter first.

**Block 5: correlated twins** (500 repeated datasets, both twins truly matter: 3 and 3)
```
Lasso: x1 coef mean/std = 1.91 / 1.82 | one twin exactly 0 in 54% of runs
Ridge: x1 coef mean/std = 2.92 / 0.07
```
Lasso gives one twin everything and the other nothing, in a nearly random way (std 1.82), whereas ridge splits it evenly and stably (std 0.07). That is Lasso's main weakness and why **Elastic Net** exists.

**Block 7: real data (diabetes)**: with CV-chosen `alpha = 0.079` only one feature (`s3`) is dropped, because most of the 10 features carry some signal. Lasso's selection helps most when there are many irrelevant features.

---

## 15. Pitfalls and FAQ

**Why are exact zeros possible with Lasso but not ridge?**
The corner of `|β|` at 0 (constant slope pushing toward 0) vs the smooth `β²` (slope fading to 0). See §8.

**Can I use Lasso's non-zero features and refit with plain OLS?**
Yes, this "relaxed Lasso" / post-selection refit removes the shrinkage bias, but do it on separate data or with CV, otherwise you overstate accuracy.

**What if two important features are correlated?**
Lasso may drop one arbitrarily. Use **Elastic Net** if you want correlated features kept together.

**Is the chosen set of features "the truth"?**
No. It's the set that predicts well under the penalty. With noisy or correlated data the selected set can change between samples, so don't over-interpret it.

**Does Lasso need feature scaling?**
Yes, always. Otherwise features with small units get penalised less than features with large units.

**What about convergence warnings?**
Increase `max_iter`, standardise the features, or loosen `tol`.

---

## 16. Cheat sheet, exercises, glossary

### Cheat sheet
```
Loss                 J(β) = ‖y − Xβ‖² + λ‖β‖₁
No closed form       |β| has a corner at 0
One feature          β̂ = S(xᵀy, λ/2) / (xᵀx),   S(z,t) = sign(z)·max(|z| − t, 0)
Optimality (KKT)     βⱼ ≠ 0 ⇒ 2xⱼᵀr = λ·sign(βⱼ);   βⱼ = 0 ⇒ |2xⱼᵀr| ≤ λ
Coordinate descent   βⱼ ← S(xⱼᵀ r⁽ʲ⁾, λ/2) / (xⱼᵀxⱼ)
All-zero threshold   λ_max = 2·max|xⱼᵀy|
Bayesian view        Laplace prior ⇒ λ = 2σ²/b
sklearn mapping      λ = 2n·alpha
```

### Exercises
1. For `x = [1, 2, 3]`, `y = [2, 4, 5]`, what λ gives `β̂ = 1.0`? *(Answer: (25 − λ/2)/14 = 1 → λ = 22.)*
2. Prove `S(z, t)` minimises `½(β − z)² + t|β|` by checking the three cases.
3. Verify numerically that `λ_max = 2·max|xⱼᵀy|` makes all coefficients zero on your own data.
4. Run Block 3 with `p = 200` features. How many noise features does `LassoCV` keep?
5. Implement the one-standard-error rule using `LassoCV`'s `mse_path_`.
6. Refit plain OLS on only the features Lasso selected (using a fresh test set). Does the test error improve?

### Glossary
| Term | Meaning |
|---|---|
| **L1 norm** | `‖β‖₁ = Σ|βⱼ|` |
| **Sparse model** | Most coefficients are exactly zero. |
| **Subgradient** | Generalised slope at a corner; at 0 it can be any value in [−1, 1]. |
| **Soft-thresholding** | `S(z, t)`: shrink by t and clamp at 0. |
| **Coordinate descent** | Optimise one parameter at a time, cycling until convergence. |
| **Partial residual** | The residual with one feature's contribution added back. |
| **KKT conditions** | The exact optimality conditions of a constrained/penalised problem. |
| **Lasso path** | Coefficients plotted as a function of λ. |
| **λ_max** | Smallest λ that sets all coefficients to zero. |
