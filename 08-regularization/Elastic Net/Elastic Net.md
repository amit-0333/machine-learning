# Elastic Net Regression: The Essentials

> **Goal:** know what Elastic Net is, why it exists, how its loss and parameters work, and when to use it. This guide is intentionally lighter than the Ridge and Lasso guides: it gives all the knowledge you need without the full derivations.
>
> **Prerequisites:** the **Ridge** and **Lasso** guides (Elastic Net is just the two combined).

## Contents
1. [The 60-second version](#1-the-60-second-version)
2. [Why Elastic Net exists](#2-why-elastic-net-exists)
3. [The loss function](#3-the-loss-function)
4. [How the solution works (short version)](#4-how-the-solution-works-short-version)
5. [The grouping effect](#5-the-grouping-effect)
6. [The two hyperparameters (and the sklearn trap)](#6-the-two-hyperparameters-and-the-sklearn-trap)
7. [Which model should I use?](#7-which-model-should-i-use)
8. [Code (all tested)](#8-code-all-tested)
9. [Pitfalls and FAQ](#9-pitfalls-and-faq)
10. [Cheat sheet, exercises, glossary](#10-cheat-sheet-exercises-glossary)

---

## 1. The 60-second version

| | |
|---|---|
| **Idea** | Use **both** penalties: L1 (Lasso) for sparsity **and** L2 (Ridge) for stability. |
| **Loss** | `‖y − Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖β‖²` |
| **Gets you** | Feature selection (zeros) **and** sensible handling of correlated features. |
| **Closed form?** | No (L1 part), solved by coordinate descent like Lasso. |
| **Two knobs** | Overall strength + the **mix** between L1 and L2. |
| **Special cases** | Mix = all L1 → Lasso. Mix = all L2 → Ridge. |

---

## 2. Why Elastic Net exists

Each earlier method has a weakness:

| Method | Weakness |
|---|---|
| **Ridge** | Never removes features, so no selection and a dense, harder-to-interpret model. |
| **Lasso** | (1) With **correlated features** it keeps one and drops the rest, randomly. (2) When `p > n` it can keep at most `n` features. (3) Unstable selection. |

Elastic Net (Zou & Hastie, 2005) keeps Lasso's sparsity but adds a ridge term that **stabilises the solution and keeps correlated features together**.

---

## 3. The loss function

```
J(β) = ‖y − Xβ‖²  +  λ₁ Σ|βⱼ|  +  λ₂ Σβⱼ²
        └── fit ──┘   └── L1 ──┘   └── L2 ──┘
```

**Geometry:** the allowed region is a **diamond with bulging sides** (between Lasso's diamond and Ridge's circle). It keeps Lasso's **corners** (so zeros are still possible) but its sides are curved, which makes it behave better with correlated features.

```
   Lasso ◇         Elastic Net ⬭ (diamond with rounded sides)         Ridge ○
```

---

## 4. How the solution works (short version)

- **Standardise** features and **centre** y first (as always). The intercept is not penalised.
- The L1 part means there is **no closed form**, so we use the same **coordinate descent** as Lasso. The only change in the one-coefficient update is an extra `+ λ₂` in the denominator:

```
Lasso:        βⱼ ← S( xⱼᵀ r, λ/2 )  /  xⱼᵀxⱼ
Elastic Net:  βⱼ ← S( xⱼᵀ r, λ₁/2 ) / ( xⱼᵀxⱼ + λ₂ )
```
Read it as **"Lasso's soft-threshold (selects), then Ridge's extra shrinkage in the denominator (stabilises)."**

- With the ridge part, the problem is always **strictly convex** (a unique solution), even when features are perfectly correlated or `p > n`. Lasso alone doesn't guarantee that.
- **Double shrinkage:** because both penalties shrink, coefficients can end up smaller than ideal. The original paper rescales them by `(1 + λ₂)`; scikit-learn does **not** do this, which is fine in practice since you tune the penalties by CV anyway.

---

## 5. The grouping effect

> If several features are highly correlated, Elastic Net tends to give them **similar coefficients** (all in or all out), while Lasso keeps one and drops the others.

Why it matters: imagine 5 gene measurements that all track the same biological signal. Lasso might keep gene 3 today and gene 5 on tomorrow's sample. Elastic Net keeps the **whole group** with shared weight, which is more stable and more faithful to the truth.

Sanity-check numbers from the code below (two groups of 5 near-duplicate true features, 40 training rows, 50 features, 30 repeats):

| Model | True features kept (of 10) | Test R² |
|---|---|---|
| Lasso | **5.4** | 0.939 |
| Elastic Net | **9.6** | 0.940 |

Same accuracy, but Elastic Net recovers nearly the whole true group while Lasso drops about half of it.

---

## 6. The two hyperparameters (and the sklearn trap)

Elastic Net has **two** things to tune.

| Concept | Meaning | Range |
|---|---|---|
| **Overall strength** | How hard to regularise | `alpha` > 0 (larger = simpler model) |
| **Mix** | L1 vs L2 share | `l1_ratio` in [0, 1] |

**Scikit-learn's exact loss** (this is the parametrisation you will actually use):

```
(1 / 2n)‖y − Xβ‖²  +  alpha · l1_ratio · ‖β‖₁  +  0.5 · alpha · (1 − l1_ratio) · ‖β‖²
```

| `l1_ratio` | Behaves like |
|---|---|
| 1.0 | **Lasso** |
| 0.5 | even mix |
| close to 0 | **Ridge-like** (but not exactly 0: use `Ridge` instead, because `l1_ratio=0` is numerically unreliable) |

**Conversion from our loss:** `λ₁ = 2n · alpha · l1_ratio` and `λ₂ = n · alpha · (1 − l1_ratio)`.

**How to tune:** use `ElasticNetCV`, with a grid like `l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]`. It searches an `alpha` grid for each ratio and returns the best pair. A ratio near 1 is a good first guess when you want a sparse model; use lower ratios when features are strongly correlated.

---

## 7. Which model should I use?

| Situation | Best choice |
|---|---|
| Few clean features, `n ≫ p` | Plain OLS (or tiny ridge) |
| Many features, **all matter a little**, correlated | **Ridge** |
| Many features, **only a few matter**, mostly independent | **Lasso** |
| Many features, **a few matter**, and they are **correlated in groups** | **Elastic Net** |
| `p > n` and you want selection of more than `n` features | **Elastic Net** |
| Unsure | Try all three with CV and compare |

```
         Need to drop features?
            ┌─────┴─────┐
           no           yes
            │            │
         Ridge     Are features correlated in groups?
                     ┌─────┴─────┐
                    no           yes
                     │            │
                   Lasso     Elastic Net
```

**Summary table**

| | OLS | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| Penalty | none | L2 | L1 | L1 + L2 |
| Exact zeros | no | no | **yes** | **yes** |
| Correlated groups | unstable | **shared** | one only | **shared** |
| Closed form | yes | yes | no | no |
| `p > n` | fails | works | ≤ n kept | works |
| Tuning knobs | 0 | 1 | 1 | **2** |

---

## 8. Code (all tested)

Needs `numpy`, `matplotlib`, `scikit-learn`.

### 8.1 Compare Ridge, Lasso and Elastic Net on correlated groups

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, RidgeCV, Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# ---------- 1. Data: two GROUPS of correlated true features + lots of noise, p close to n ----------
rng = np.random.default_rng(1)
n_tr, n_te, p = 60, 3000, 50
def make(n):
    g1, g2 = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    X = rng.normal(size=(n, p))
    X[:, :5]   = g1 + 0.3 * rng.normal(size=(n, 5))      # group 1: features 0-4 move together
    X[:, 5:10] = g2 + 0.3 * rng.normal(size=(n, 5))      # group 2: features 5-9 move together
    y = 2 * X[:, :10].sum(axis=1) + rng.normal(scale=3.0, size=n)   # all 10 matter equally
    return X, y
X_tr, y_tr = make(n_tr); X_te, y_te = make(n_te)

# ---------- 2. Fit the three regularised models, each tuned by 5-fold CV ----------
ridge = RidgeCV(alphas=np.logspace(-2, 3, 40), cv=5).fit(X_tr, y_tr)
lasso = LassoCV(cv=5, random_state=0).fit(X_tr, y_tr)
enet  = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, .99], cv=5, random_state=0).fit(X_tr, y_tr)

print(f"{'model':11s} {'test R2':>8s} {'nonzero':>8s} {'true feats kept (of 10)':>25s}")
for name, m in [("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", enet)]:
    nz = np.abs(m.coef_) > 1e-8
    print(f"{name:11s} {m.score(X_te, y_te):8.3f} {nz.sum():8d} {nz[:10].sum():25d}")
print("chosen: alpha =", round(enet.alpha_, 3), " l1_ratio =", enet.l1_ratio_)
print("Lasso coefs, group 1 :", lasso.coef_[:5].round(2))
print("ENet  coefs, group 1 :", enet.coef_[:5].round(2))
print("Ridge coefs, group 1 :", ridge.coef_[:5].round(2))

# ---------- 3. Sliding l1_ratio from ridge-like to lasso-like ----------
Xs = StandardScaler().fit_transform(X_tr)
for r in [0.1, 0.5, 0.9, 1.0]:
    m = ElasticNet(alpha=0.3, l1_ratio=r, max_iter=50000).fit(Xs, y_tr)
    print(f"l1_ratio={r:.1f}: non-zero coefficients = {int(np.sum(m.coef_ != 0)):2d}")
```

### 8.2 Repeat many times: does Elastic Net really keep the group?

```python
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNetCV, LassoCV
def make(n, rng, p=50, noise=0.1):
    g1, g2 = rng.normal(size=(n,1)), rng.normal(size=(n,1))
    X = rng.normal(size=(n,p))
    X[:,:5]=g1+noise*rng.normal(size=(n,5)); X[:,5:10]=g2+noise*rng.normal(size=(n,5))
    return X, 2*X[:,:10].sum(1)+rng.normal(scale=3,size=n)
rng=np.random.default_rng(5); res={"lasso":[], "enet":[]}; r2={"lasso":[], "enet":[]}
for _ in range(30):
    X,y=make(40,rng); Xt,yt=make(2000,rng)
    for name,m in [("lasso",LassoCV(cv=5)),("enet",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95],cv=5))]:
        m.fit(X,y); res[name].append(np.sum(np.abs(m.coef_[:10])>1e-8)); r2[name].append(m.score(Xt,yt))
for k in res: print(k,"true feats kept (of 10): mean",np.mean(res[k]).round(2),"| test R2 mean",np.mean(r2[k]).round(3))
```

### 8.3 What the outputs show

**Block 1:** (one dataset: 60 rows, 50 features, two groups of 5 correlated true features)
```
model        test R2  nonzero  true feats kept (of 10)
Ridge          0.923       50                       10
Lasso          0.940       22                       10
ElasticNet     0.941       20                       10
chosen: alpha = 0.248  l1_ratio = 0.9
```
- Ridge uses all 50 features and is least accurate.
- Lasso and Elastic Net are better and sparser. On this single run both happened to find all 10.
- The group-1 coefficients show the effect: Ridge `[1.96 1.48 1.88 1.96 2.33]` is evenly shared; Lasso `[1.47 0.38 1.89 2.87 3.01]` is uneven; Elastic Net `[1.67 0.81 1.91 2.49 2.77]` sits in between.
- Sliding `l1_ratio` from 0.1 to 1.0 shrinks the number of non-zero coefficients (44, 26, 19, 19), which shows the mix knob moving from Ridge-like to Lasso-like.

**Block 2:** the more robust test (30 repeated datasets)
```
lasso true feats kept (of 10): mean 5.43 | test R2 mean 0.939
enet  true feats kept (of 10): mean 9.57 | test R2 mean 0.94
```
Single runs can be lucky. Averaged over many datasets the grouping effect is clear: **Lasso loses about half the correlated true features; Elastic Net keeps almost all of them.**

---

## 9. Pitfalls and FAQ

**Is Elastic Net always better than Lasso or Ridge?**
No. If your features are independent, Lasso is as good; if you don't need sparsity, Ridge is simpler. Elastic Net costs an extra hyperparameter to tune.

**Do I need to scale features?**
Yes, always (both penalties are scale-sensitive).

**Why does `l1_ratio=0` give warnings?**
It becomes pure Ridge, and the coordinate-descent solver isn't designed for that. Use `Ridge` instead.

**Is `alpha` the same as λ?**
No. The sklearn loss divides the fit term by `2n` and splits `alpha` by `l1_ratio`. Use the conversion in §6 if you need to compare with the formulas.

**Tuning is slow, any tips?**
Use `ElasticNetCV` (it reuses solutions along the path), standardise features, and start with a coarse `l1_ratio` grid.

**Do I need the rescaling `(1 + λ₂)` from the paper?**
Usually no, because CV chooses penalties that already account for the extra shrinkage.

---

## 10. Cheat sheet, exercises, glossary

### Cheat sheet
```
Loss          ‖y − Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖²
sklearn       (1/2n)‖y − Xβ‖² + α·ρ‖β‖₁ + ½·α·(1−ρ)‖β‖²      (α = alpha, ρ = l1_ratio)
Update        βⱼ ← S(xⱼᵀr, λ₁/2) / (xⱼᵀxⱼ + λ₂)
Special cases ρ = 1 → Lasso ; ρ → 0 → Ridge
Strength      sparsity (L1) + stability & grouping (L2)
Tune          ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.99], cv=5) on standardised features
```

### Exercises
1. Set `l1_ratio=1.0` in `ElasticNet` and compare with `Lasso` using `alpha` equal. Are the coefficients identical?
2. Increase the correlation inside the groups (change `0.1` noise to `0.02` in §8.2). How do the "true features kept" numbers change?
3. Plot the number of non-zero coefficients against `l1_ratio` for a fixed `alpha`.
4. Which `l1_ratio` does `ElasticNetCV` choose when the true features are independent? Why?

### Glossary
| Term | Meaning |
|---|---|
| **Grouping effect** | Correlated features receive similar coefficients. |
| **l1_ratio** | Fraction of the penalty that is L1 (1 = Lasso). |
| **Double shrinkage** | Both L1 and L2 shrink, so coefficients can be too small. |
| **Strict convexity** | The loss has exactly one minimum; guaranteed by the L2 part. |
| **Coordinate descent** | Optimise one coefficient at a time until convergence. |
