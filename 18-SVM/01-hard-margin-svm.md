# Support Vector Machines (SVM): Session 1 (Maximal Margin Classifier / Hard Margin SVM)

## Table of Contents

1. [Introduction](#1-introduction)
2. [Plan of Attack](#2-plan-of-attack)
3. [Maximal Margin Classifier](#3-maximal-margin-classifier)
4. [Support Vectors](#4-support-vectors)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [How to Solve This?](#6-how-to-solve-this)
7. [Prediction](#7-prediction)
8. [Code](#8-code)
9. [Problem with Hard Margin SVM](#9-problem-with-hard-margin-svm)
10. [Summary & Cheat Sheet](#10-summary--cheat-sheet)

> **Note:** most of the source PDF for this session was handwritten (only the headings and the introduction were extractable). Everything below follows the standard treatment of each heading, so check it against what was drawn in class.

---

## 1. Introduction

**Support Vector Machine (SVM)** is a supervised learning algorithm used mainly for **classification** (and also regression, as SVR).

From the notes:

- **Born in the 90's.** (Vapnik and colleagues; the modern soft-margin version by Cortes & Vapnik, 1995.)
- **"Best out-of-the-box classifier":** with little tuning it gives strong results, which is why it dominated before deep learning became popular.
- **Applicable to a wide variety of problems, including non-linear data** (through the *kernel trick*, covered in later sessions).
- **Difficult maths:** it needs constrained optimization, Lagrange multipliers and duality. Take it step by step.

### 1.1 The core idea in one line

> Among all the lines (hyperplanes) that separate the two classes, pick the one that is **as far as possible from the nearest points of both classes**, the one with the **maximum margin**.

### 1.2 Why this is different from Logistic Regression / Perceptron

When data is linearly separable, there are **infinitely many** separating lines. Logistic regression and the perceptron accept whichever one training gives them. SVM asks: *which of these lines is the safest?*

| | Logistic Regression | SVM |
|---|---|---|
| Chooses the boundary by | Maximizing likelihood (uses **all** points) | Maximizing the **margin** (only the closest points matter) |
| Output | Probability | Class (and a signed distance score) |
| Influenced by far-away points? | Yes (a little) | **No** |

---

## 2. Plan of Attack

A typical roadmap for learning SVM (and the order in which the topics are covered):

```
1. Maximal Margin Classifier  ──► works only when classes are perfectly separable by a line   (this session)
        │
        ▼ problem: no separating line / outliers
2. Support Vector Classifier (Soft Margin) ──► allows some mistakes, controlled by C
        │
        ▼ problem: data is not linearly separable at all
3. Support Vector Machine (Kernels) ──► linear boundary in a higher-dimensional space = curved boundary
        │
        ▼
4. Tuning (C, kernel, gamma), multiclass, SVR, pros & cons
```

**This session:** geometry → margin → support vectors → optimization problem → solving it (Lagrangian/dual) → prediction → code → why the *hard* margin fails.

---

## 3. Maximal Margin Classifier

### 3.1 Hyperplane recap

In `d` dimensions a hyperplane is

```
wᵀx + b = 0          (w = weight/normal vector, b = bias)
```

- In 2D it's a line `w1·x1 + w2·x2 + b = 0`.
- `w` is perpendicular to the hyperplane and points toward the **positive side**.
- As in logistic regression: `wᵀx + b > 0` → positive side, `< 0` → negative side.

**Labels:** for SVM we use **`y ∈ {+1, −1}`** (not 0/1). This makes the maths compact.

A point is **correctly classified** when `y·(wᵀx + b) > 0` (same sign as its label).

### 3.2 Distance of a point from the hyperplane

```
distance(x, hyperplane) = |wᵀx + b| / ‖w‖
```

For a correctly classified point, `y·(wᵀx + b) / ‖w‖` is its (positive) **geometric distance** from the boundary.

### 3.3 Margin

> The **margin** is the width of the empty "street" around the hyperplane. It is twice the distance from the hyperplane to the closest training point.

The margin is bounded by two parallel hyperplanes passing through the closest points of each class:

```
   x2
    │   ○ ○        wᵀx + b = +1  ◄── "positive margin" plane
    │  ○   ○ ────────────────────────
    │            ← margin = 2/‖w‖ →
    │ ────────── wᵀx + b = 0   ◄── decision boundary (middle of the street)
    │            
    │ ────────────────────────
    │  ✕  ✕        wᵀx + b = −1  ◄── "negative margin" plane
    │    ✕
    └──────────────────────── x1
```

### 3.4 The Maximal Margin Classifier

Among all separating hyperplanes, choose the one with the **largest margin**.

**Why is a large margin good?**

- **Robustness:** new points from the same distribution lie near the training ones; a wide street leaves more room for noise before a point crosses the line.
- **Better generalization:** a large margin acts like a built-in regularizer. Statistical learning theory links wide margins to lower test error.
- **Uniqueness:** exactly one best (maximum margin) hyperplane exists, instead of infinitely many.

Also known as the **Hard Margin SVM** (no point is allowed inside the margin or on the wrong side).

---

## 4. Support Vectors

> **Support vectors** are the training points that lie **exactly on the margin boundaries** (`y·(wᵀx + b) = 1`), the points closest to the decision boundary.

They "support" (hold up) the hyperplane:

- The hyperplane is determined **only** by the support vectors.
- If you **delete any non-support-vector point**, or move it without crossing the margin, the solution **does not change**.
- If you move a support vector, the hyperplane moves.

```
        ○        ○
          ○   [○]  ◄── support vector (on +1 plane)
   ───────────────────
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   decision boundary
   ───────────────────
          [✕]  ✕      ◄── support vector (on −1 plane)
      ✕        ✕
```

**Why this matters**

- **Sparse model:** prediction needs only the support vectors, usually a small fraction of the data, so it is memory-efficient.
- **Robust to far-away points:** unlike logistic regression, points deep inside their class region don't influence the boundary.
- Usually **at least one support vector per class** (≥ 2 in total).

---

## 5. Mathematical Formulation

### 5.1 Fixing the scale

The hyperplane `wᵀx + b = 0` is unchanged if we multiply `(w, b)` by any constant. So we **choose the scale** such that the closest points satisfy

```
y_i (wᵀx_i + b) = 1      for the support vectors
y_i (wᵀx_i + b) ≥ 1      for all other points
```

### 5.2 Margin width

The distance from the middle hyperplane to a support vector is `1/‖w‖`, so

```
margin = 2 / ‖w‖
```

### 5.3 The optimization problem

**Maximize** the margin `2/‖w‖` ⇔ **minimize** `‖w‖` ⇔ **minimize** `½‖w‖²` (squared and halved for easier differentiation; same optimum).

```
     minimize     ½ ‖w‖²
     w, b

     subject to   y_i (wᵀx_i + b) ≥ 1      for every i = 1 … n
```

- **Objective:** make the margin as wide as possible.
- **Constraints:** every training point must be on the correct side *and* outside the margin.

This is a **convex quadratic programming (QP)** problem: a convex quadratic objective with linear constraints. Therefore:

- It has a **single global optimum** (no local minima).
- A solution exists **only if the data is linearly separable** (otherwise the constraints can't all be satisfied).

### 5.4 Quick sanity check with a tiny example

Two points: `x₁ = (1, 1)` with `y = −1`, and `x₂ = (3, 3)` with `y = +1`.

Both are support vectors, so

```
(3,3):   +1 · (3w₁ + 3w₂ + b) = 1
(1,1):   −1 · ( w₁ +  w₂ + b) = 1
```

By symmetry `w₁ = w₂ = a`: `6a + b = 1` and `2a + b = −1` ⇒ `a = 0.5`, `b = −2`.

```
w = (0.5, 0.5),   b = −2
‖w‖ = √0.5 ≈ 0.7071
margin = 2/‖w‖ = 2.828 = distance between the two points (√8) ✓
decision boundary: 0.5x₁ + 0.5x₂ − 2 = 0  ⇒  x₁ + x₂ = 4
```

---

## 6. How to Solve This?

The constrained problem is solved with **Lagrange multipliers**, which also leads to the **dual form**, the key to the kernel trick.

### 6.1 Step 1: the Lagrangian

Introduce a multiplier `α_i ≥ 0` for each constraint:

```
L(w, b, α) = ½‖w‖² − Σᵢ α_i [ y_i (wᵀx_i + b) − 1 ]
```

We minimize `L` over `(w, b)` and maximize over `α ≥ 0`.

### 6.2 Step 2: set the derivatives to zero

```
∂L/∂w = w − Σ α_i y_i x_i = 0    ⇒   w = Σᵢ α_i y_i x_i
∂L/∂b = −Σ α_i y_i       = 0    ⇒   Σᵢ α_i y_i = 0
```

> **Important:** `w` is a **weighted combination of the training points**, weighted by `α_i y_i`.

### 6.3 Step 3: substitute back → the Dual Problem

Plugging `w` into `L` eliminates `w` and `b`:

```
maximize    Σᵢ α_i  −  ½ Σᵢ Σⱼ α_i α_j y_i y_j (x_iᵀ x_j)
   α

subject to  α_i ≥ 0     for all i
            Σᵢ α_i y_i = 0
```

Notes:

- Another convex QP (in `α`). Solved by specialized algorithms, e.g. **SMO (Sequential Minimal Optimization)**, which is what `libsvm` (inside scikit-learn's `SVC`) uses.
- The data enters **only through dot products `x_iᵀx_j`**. Replace the dot product with a **kernel** `K(x_i, x_j)` and you get a non-linear SVM. That's why the dual form matters.

### 6.4 Step 4: KKT conditions (why only support vectors matter)

At the optimum the **complementary slackness** condition holds:

```
α_i · [ y_i (wᵀx_i + b) − 1 ] = 0
```

For every point, one of two things is true:

| Case | Meaning | α_i |
|---|---|---|
| `y_i(wᵀx_i + b) > 1` | Point is strictly **outside** the margin (constraint is not tight) | **α_i = 0** (no influence) |
| `y_i(wᵀx_i + b) = 1` | Point is **on** the margin = **support vector** | **α_i > 0** |

So `w = Σ α_i y_i x_i` only involves the **support vectors**.

### 6.5 Step 5: recover b

Take any support vector `x_s` (where `y_s(wᵀx_s + b) = 1`, so `b = y_s − wᵀx_s`):

```
b = y_s − Σᵢ α_i y_i (x_iᵀ x_s)
```

In practice, average this over all support vectors for numerical stability.

### 6.6 Checking the tiny example with the dual

For `x₁ = (1,1), y₁ = −1` and `x₂ = (3,3), y₂ = +1`:

```
Σ α_i y_i = 0   ⇒   −α₁ + α₂ = 0   ⇒   α₁ = α₂ = α
w = α₂·(+1)·(3,3) + α₁·(−1)·(1,1) = α·(2,2) = (0.5, 0.5)   ⇒   α = 0.25
```

So `α₁ = α₂ = 0.25`, which matches what scikit-learn returns (`dual_coef_ = [-0.25, 0.25]`, since it stores `α_i y_i`).

### 6.7 Alternatives

Because the problem is a QP, you can also solve the **primal** directly (gradient-based methods on the hinge-loss form, which is what `LinearSVC`/`SGDClassifier(loss="hinge")` do), but that works for the linear case only.

---

## 7. Prediction

### 7.1 Decision rule

For a new point `x`:

```
ŷ = sign( wᵀx + b )
  = sign( Σ_{i ∈ SV} α_i y_i (x_iᵀ x) + b )
```

- Result `+1` → positive class, `−1` → negative class.
- The number `f(x) = wᵀx + b` is the **decision function**. Its magnitude shows how far the point is from the boundary (`|f(x)|/‖w‖` is the geometric distance). Points with `|f(x)| < 1` lie **inside the margin**.
- Only the **support vectors** are needed to predict, which is fast and compact.
- SVM gives **class labels / scores, not probabilities**. (scikit-learn can add probabilities with `probability=True`, using an internal Platt-scaling step; it is slower.)

### 7.2 Using the tiny example (`w = (0.5, 0.5)`, `b = −2`)

| New point | f(x) = 0.5x₁ + 0.5x₂ − 2 | Prediction |
|---|---|---|
| (4, 2) | 1.0 | **+1** (right on the +margin) |
| (0, 1) | −1.5 | **−1** |
| (2, 2) | 0 | on the boundary (tie) |
| (3, 1.5) | 0.25 | **+1**, but inside the margin |

---

## 8. Code

### 8.1 Hard-margin linear SVM with scikit-learn

scikit-learn has no explicit "hard margin" switch. Use a **linear kernel with a very large `C`** (so no violations are tolerated). `C` is explained in the next session.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Linearly separable data
X, y = make_blobs(n_samples=60, centers=[(-2, -2), (2, 2)], cluster_std=0.6, random_state=7)

clf = SVC(kernel="linear", C=1e10)        # huge C ≈ hard margin
clf.fit(X, y)

w = clf.coef_[0]                          # weights  (only for kernel="linear")
b = clf.intercept_[0]                     # bias

print("w =", w, " b =", b)
print("Support vectors per class:", clf.n_support_)          # e.g. [2 1]
print("Support vectors:\n", clf.support_vectors_)
print("Margin width = 2/||w|| =", 2 / np.linalg.norm(w))     # ≈ 3.66 here
print("Training accuracy:", clf.score(X, y))                 # 1.0 (separable)
```

**Useful attributes**

| Attribute | Meaning |
|---|---|
| `support_vectors_` | The support vectors themselves |
| `support_` | Their indices in the training set |
| `n_support_` | Number of support vectors per class |
| `dual_coef_` | `α_i · y_i` for each support vector |
| `coef_` | `w` (linear kernel only) |
| `intercept_` | `b` |
| `decision_function(X)` | `wᵀx + b` for each sample |

### 8.2 Verify the maths: the dual form gives the same prediction

```python
x_new = np.array([[0.5, 1.0]])

# f(x) = Σ α_i y_i (x_i · x) + b   (dual_coef_ already stores α_i y_i)
f_dual = (clf.dual_coef_ @ (clf.support_vectors_ @ x_new.T)).item() + b
print(f_dual, clf.decision_function(x_new))       # the two numbers match
```

All support vectors satisfy `y(wᵀx + b) ≈ 1` and every other point has `y(wᵀx + b) > 1`:

```python
ys = np.where(y == 1, 1, -1)
print((ys * (X @ w + b)).min())                    # ≈ 1.0
```

### 8.3 Plot the boundary, margins and support vectors

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")  # margins & boundary
plt.scatter(*clf.support_vectors_.T, s=200, facecolors="none", edgecolors="g", linewidths=2)
plt.title("Maximal margin classifier")
plt.show()
```

### 8.4 Preprocessing note

SVM is based on distances, so **always scale features** (`StandardScaler`). Otherwise features with large ranges dominate the margin.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1e10))
```

---

## 9. Problem with Hard Margin SVM

The maximal margin classifier is elegant, but it has serious weaknesses.

### 9.1 Problem 1: it needs perfectly linearly separable data

The constraints `y_i(wᵀx_i + b) ≥ 1` can only all be satisfied if some line separates the classes perfectly. **Real data almost never is:** classes overlap, there is label noise, and so on. When no separating hyperplane exists, the optimization problem is **infeasible** (no solution). With a huge `C` in scikit-learn the solver struggles or fails to converge.

```
   ○ ○ ✕ ○         ← a ✕ sitting inside the ○ cluster:
  ○ ○ ○ ✕ ✕           no straight line can separate them
   ✕ ✕ ○ ✕
```

### 9.2 Problem 2: extreme sensitivity to outliers

Even when the data is still separable, **a single point near the boundary** can drastically change the hyperplane and shrink the margin, because the hard margin treats every point as a strict constraint.

**Demonstration** (same blobs as in section 8; one extra class-0 point placed closer to the other class):

| Data | Margin width (2/‖w‖) |
|---|---|
| Original separable data | ≈ **3.66** |
| + one class-0 point at (0.0, 0.2) | ≈ 1.74 |
| + one class-0 point at (0.8, 0.8) | ≈ **0.74** (a single point shrinks the margin by ~80%) |

The resulting boundary is squeezed through a narrow gap, tuned to one unusual point, so it likely **overfits** (high variance).

### 9.3 Problem 3: linear only

A straight hyperplane can't separate data like concentric circles or half-moons, no matter how the margin is chosen. (Fixed later with kernels.)

### 9.4 The fix (next session)

**Soft Margin SVM / Support Vector Classifier:** allow some points to violate the margin, but pay a penalty. Introduce **slack variables** `ξ_i ≥ 0`:

```
minimize   ½‖w‖² + C · Σᵢ ξ_i
subject to y_i (wᵀx_i + b) ≥ 1 − ξ_i ,   ξ_i ≥ 0
```

- `ξ_i = 0` → point is correctly classified and outside the margin.
- `0 < ξ_i < 1` → inside the margin but still on the correct side.
- `ξ_i > 1` → misclassified.
- **`C`** is a hyperparameter trading margin width against violations:
  - **Large C:** few violations allowed → narrow margin, closer to hard margin, risk of overfitting.
  - **Small C:** more violations tolerated → wide margin, simpler model, risk of underfitting.

Then **kernels** extend this to non-linear boundaries.

---

## 10. Summary & Cheat Sheet

1. **SVM** finds the separating hyperplane with the **maximum margin**, not just any separating line.
2. **Hyperplane:** `wᵀx + b = 0`; labels are `±1`; correct classification means `y(wᵀx + b) > 0`.
3. **Margin** = `2/‖w‖`, bounded by planes `wᵀx + b = ±1`.
4. **Support vectors** are the points lying on the margin planes. They alone define the solution; all other points can be removed without changing it.
5. **Optimization (primal):** `min ½‖w‖²  s.t.  y_i(wᵀx_i + b) ≥ 1`, a convex QP.
6. **Solving:** Lagrangian → `w = Σ α_i y_i x_i`, `Σ α_i y_i = 0` → **dual problem** in terms of dot products (kernel-ready). By **KKT**, `α_i > 0` only for support vectors.
7. **Prediction:** `ŷ = sign(Σ_SV α_i y_i x_iᵀx + b)`.
8. **Code:** `SVC(kernel="linear", C=very large)`; scale features first.
9. **Hard margin problems:** requires linear separability, hypersensitive to outliers, linear only → fixed by **soft margin (C)** and **kernels** in the next sessions.

```
Data ─► scale ─► find w, b maximizing margin (2/‖w‖)
                    │  only support vectors matter
                    ▼
        ŷ = sign(wᵀx + b)
        fails if: classes overlap · outliers · non-linear shape  ──► Soft margin + Kernels
```

| Question | Answer |
|---|---|
| What does SVM maximize? | The margin `2/‖w‖` |
| Which points define the boundary? | Support vectors only |
| Why use the dual form? | Depends only on dot products ⇒ enables kernels; `α` is sparse |
| What breaks the hard margin? | Overlapping classes, outliers |
| What is the remedy? | Soft margin (slack `ξ`, penalty `C`) + kernel trick |
