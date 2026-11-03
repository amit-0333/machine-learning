# Support Vector Machines (SVM): Session 2 (Soft Margin SVM)

## Table of Contents

1. [Recap](#1-recap)
2. [Problems with Hard Margin SVM](#2-problems-with-hard-margin-svm)
3. [Slack Variable](#3-slack-variable)
4. [Soft Margin SVM](#4-soft-margin-svm)
5. [Introduction of C](#5-introduction-of-c)
6. [Bias-Variance Tradeoff](#6-bias-variance-tradeoff)
7. [Code Example](#7-code-example)
8. [Relationship with Logistic Regression](#8-relationship-with-logistic-regression)
9. [Summary & Cheat Sheet](#9-summary--cheat-sheet)

> **Note:** parts of the source PDF (Soft Margin SVM, Bias-Variance, Code, Relationship with Logistic Regression) were handwritten or image-only, so they are written from the standard treatment of each heading. The slack-variable definition follows your typed notes.

---

## 1. Recap

From Session 1 (Hard Margin SVM / Maximal Margin Classifier):

| Concept | Summary |
|---|---|
| Hyperplane | `wᵀx + b = 0`, labels `y ∈ {+1, −1}` |
| Margin | `2/‖w‖`, the width of the empty street around the boundary |
| Support vectors | Points exactly on the margin planes `y(wᵀx + b) = 1`. They alone define the solution |
| Hard margin problem | `min ½‖w‖²  s.t.  y_i(wᵀx_i + b) ≥ 1` for **all** i |
| Solved by | Lagrangian → dual (depends only on dot products `x_iᵀx_j`) |
| Prediction | `ŷ = sign(wᵀx + b)` |

---

## 2. Problems with Hard Margin SVM

| Problem | Why it happens |
|---|---|
| **1. Needs perfectly linearly separable data** | Every constraint `y_i(wᵀx_i + b) ≥ 1` must hold. If classes overlap, no hyperplane satisfies them all, so the problem has **no solution** (infeasible). |
| **2. Very sensitive to outliers** | One point near (or on the wrong side of) the other class forces the boundary to squeeze through a tiny gap, shrinking the margin drastically. In Session 1's demo, one extra point cut the margin from 3.66 to 0.74. |
| **3. Overfitting** | A boundary tuned to a few extreme points generalizes poorly (high variance). |
| **4. Real data is noisy** | Label noise and overlap are normal, so a model that tolerates a few errors is more realistic. |

```
Hard margin:  "No mistakes allowed."   →  fails or overfits on real data
Soft margin:  "A few mistakes are OK, but each one has a price."
```

---

## 3. Slack Variable

The concept of **slack variables** was introduced by **Vladimir Vapnik in 1995** and is used in the formulation of the **"soft-margin" SVM** to handle cases where the data is not linearly separable, or when one allows for some degree of error in classification.

### 3.1 Definition

For each data point `i`, a slack variable **ξᵢ ≥ 0** is introduced. It **measures the degree of misclassification** (margin violation) of the point `xᵢ`.

| Value of ξᵢ | Where the point is |
|---|---|
| **ξᵢ = 0** | `xᵢ` is on the **correct side of the margin** (no violation) |
| **0 < ξᵢ < 1** | `xᵢ` is on the **correct side of the hyperplane but on the wrong side of the margin** (inside the street) |
| **ξᵢ ≥ 1** | `xᵢ` is on the **wrong side of the hyperplane**, i.e. **misclassified** |

### 3.2 Relaxed constraint

The hard constraint `y_i(wᵀx_i + b) ≥ 1` becomes

```
y_i (wᵀx_i + b) ≥ 1 − ξ_i ,        ξ_i ≥ 0
```

ξᵢ is exactly the amount by which point `i` falls short of the margin requirement.

### 3.3 Closed form for ξ

Given `f(x_i) = wᵀx_i + b`, the smallest slack that satisfies the constraint is

```
ξ_i = max( 0 , 1 − y_i · f(x_i) )
```

This is the **hinge loss** (see section 8).

### 3.4 Picture

```
         +1 margin ─────────────────────────
  ○  ○        ○ (ξ=0)      ○ (0<ξ<1, inside margin)
         0 boundary ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  ✕ (ξ≥1, wrong side)   ✕  ✕ (ξ=0)
         −1 margin ─────────────────────────
```

### 3.5 Worked example

Hyperplane `f(x) = x₁ + x₂ − 4`. Compute ξ for some positive-class points (`y = +1`):

| Point | f(x) | y·f(x) | ξ = max(0, 1 − y·f) | Status |
|---|---|---|---|---|
| (4, 3) | 3 | 3 | 0 | outside margin ✓ |
| (3, 2) | 1 | 1 | 0 | exactly on margin |
| (2.5, 2) | 0.5 | 0.5 | 0.5 | inside margin, correct side |
| (2, 2) | 0 | 0 | 1 | on the boundary |
| (1, 1) | −2 | −2 | 3 | misclassified |

---

## 4. Soft Margin SVM

Also called the **Support Vector Classifier (SVC)**.

### 4.1 Formulation (primal)

```
minimize        ½ ‖w‖²  +  C · Σᵢ ξ_i
 w, b, ξ

subject to      y_i (wᵀx_i + b) ≥ 1 − ξ_i       for all i
                ξ_i ≥ 0
```

Two competing goals:

| Term | Goal |
|---|---|
| `½‖w‖²` | **Wide margin** (small `‖w‖` ⇔ large `2/‖w‖`) |
| `C · Σ ξᵢ` | **Few and small violations** (total slack penalized) |

`C` decides how much each unit of violation costs (next section).

### 4.2 Dual form

Using Lagrange multipliers (as in Session 1), the dual is

```
maximize    Σᵢ α_i − ½ Σᵢ Σⱼ α_i α_j y_i y_j (x_iᵀ x_j)
subject to  0 ≤ α_i ≤ C
            Σᵢ α_i y_i = 0
```

The **only change** from the hard-margin dual is the **upper bound `α_i ≤ C`** (the "box constraint"). With `C = ∞` we recover the hard margin.

### 4.3 What each α tells you (KKT conditions)

| Condition | Position of point | ξᵢ | Role |
|---|---|---|---|
| **αᵢ = 0** | Outside the margin (correct) | 0 | Not a support vector, no influence |
| **0 < αᵢ < C** | Exactly **on** the margin | 0 | Support vector ("free") |
| **αᵢ = C** | **Inside** the margin or **misclassified** | > 0 | Support vector ("bounded"), it violates the margin |

So in a soft-margin SVM, **support vectors = points on the margin + points violating it**. Allowing violations typically makes **more** points support vectors.

Weights and prediction are the same as before: `w = Σ α_i y_i x_i`, `ŷ = sign(Σ_SV α_i y_i x_iᵀx + b)`.

### 4.4 Alternative view: regularized hinge loss

Eliminating `ξ` (using `ξᵢ = max(0, 1 − yᵢ f(xᵢ))`) gives an **unconstrained** problem:

```
minimize   ½‖w‖²  +  C · Σᵢ max(0, 1 − y_i (wᵀx_i + b))
              ▲              ▲
        L2 regularizer    hinge loss
```

---

## 5. Introduction of C

### 5.1 What is C?

- **C is a hyperparameter**, the **"tuning knob"** of the soft-margin SVM. It's chosen by you (not learned from data), `C > 0`.
- It sets the **penalty for each margin violation**, i.e. the trade-off between a wide margin and training accuracy.

In your notes the objective is written with the average slack: `‖w‖/2 + C·(1/n)·Σξᵢ`. Whether the sum is averaged by `n` or not only rescales `C`. (scikit-learn uses the plain sum: `C · Σ ξᵢ`.)

### 5.2 Effect of C

| | **Large C** (e.g. 100, 1000) | **Small C** (e.g. 0.01, 0.001) |
|---|---|---|
| Penalty per violation | **High** | **Low** |
| Tolerance for violations | Low, tries to classify every point correctly | High, accepts many violations |
| Margin | **Narrow** | **Wide** |
| Behaviour | Close to **hard margin** | Very "soft" |
| Number of support vectors | Fewer | **More** |
| Risk | **Overfitting** (follows noise/outliers) | **Underfitting** (too simple) |

Extremes: `C → ∞` ⇒ hard margin SVM. `C → 0` ⇒ violations are free, so the margin becomes huge and the model ignores the data.

### 5.3 The "SVM Parameter C" figure (from the notes)

Same data, two settings:

- **C = 1:** wide margin; several points (circled) sit inside the margin or on the wrong side, and are tolerated.
- **C = 100:** narrow margin; the boundary bends to respect almost every point.

### 5.4 Numbers from an experiment

Overlapping two-class data (200 points, standardized), linear kernel:

| C | Support vectors | Margin width 2/‖w‖ | Train acc. | Test acc. |
|---|---|---|---|---|
| 0.001 | 140 | 15.5 | 0.864 | 0.883 |
| 0.01 | 108 | 2.91 | 0.864 | 0.883 |
| 0.1 | 60 | 1.75 | 0.864 | 0.883 |
| 1 | 47 | 1.23 | 0.871 | 0.883 |
| 10 to 1000 | 45 | 1.18 | 0.879 | 0.883 |

As `C` grows: **margin shrinks**, **fewer support vectors**, training accuracy creeps up. (On this simple dataset test accuracy doesn't change, but on harder data the high-C side overfits.)

### 5.5 Relation to regularization strength

`C` plays the role of **1/λ** in regularized models (the same `C` as in scikit-learn's `LogisticRegression`): **small C = strong regularization**, **large C = weak regularization**.

---

## 6. Bias-Variance Tradeoff

| | Small C | Large C |
|---|---|---|
| Margin | Wide | Narrow |
| Flexibility of the boundary | Low (stiff) | High (bends toward points) |
| **Bias** | **High** | Low |
| **Variance** | Low | **High** |
| Fitting | **Underfitting** | **Overfitting** |

```
Error
  │\                                   /  ← test error
  │ \                                 /
  │  \_______ sweet spot ____________/
  │      (best C)
  └──────────────────────────────────────► C (small → large)
   underfit (high bias)       overfit (high variance)
   train error ↓ keeps falling as C grows
```

**How to choose C:** treat it as a hyperparameter and tune it with **cross-validation** on a **log scale** (e.g., 0.001, 0.01, 0.1, 1, 10, 100, 1000).

**Think of it this way:** a hard margin makes one decision-maker who must please everyone (fragile). A soft margin lets a few unhappy voters exist if that gives a more stable overall decision.

---

## 7. Code Example

### 7.1 Effect of C

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Overlapping classes → not perfectly separable
X, y = make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=1.2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C)).fit(X_train, y_train)
    svc = model[-1]
    print(f"C={C:<6} #SV={svc.n_support_.sum():<4} "
          f"margin={2/np.linalg.norm(svc.coef_[0]):.2f}  "
          f"train={model.score(X_train, y_train):.3f}  test={model.score(X_test, y_test):.3f}")
```

### 7.2 Tuning C with cross-validation

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
grid = GridSearchCV(pipe, {"svc__C": np.logspace(-3, 3, 7)}, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best C   :", grid.best_params_)
print("CV score :", round(grid.best_score_, 3))
print("Test acc :", round(grid.score(X_test, y_test), 3))
```

### 7.3 Inspecting slack and α

```python
svc = SVC(kernel="linear", C=1).fit(X_train, y_train)
f = svc.decision_function(X_train)
y_pm = np.where(y_train == 1, 1, -1)

xi = np.maximum(0, 1 - y_pm * f)                    # slack variables
print("Points with ξ = 0     :", (xi == 0).sum())
print("Inside margin (0<ξ<1) :", ((xi > 0) & (xi < 1)).sum())
print("Misclassified (ξ≥1)   :", (xi >= 1).sum())

# dual_coef_ stores α_i·y_i for support vectors, so |dual_coef_| ≤ C
print("max α =", np.abs(svc.dual_coef_).max(), "(≤ C = 1)")
```

### 7.4 Notes

- **Scale features** before SVM (distance-based).
- `LinearSVC` is much faster for big linear problems (but uses squared hinge loss and also regularizes the intercept by default).
- `SVC` has no probabilities by default (`probability=True` adds Platt scaling and is slower).
- For **class imbalance** use `class_weight="balanced"` (it effectively scales `C` per class).

---

## 8. Relationship with Logistic Regression

Both are **linear classifiers** with a **regularized loss**. The difference is the loss function.

### 8.1 Same structure, different loss

```
Logistic regression:   minimize   Σᵢ log(1 + e^(−y_i f(x_i)))    +  λ‖w‖²     (log loss)
Soft-margin SVM:       minimize   Σᵢ max(0, 1 − y_i f(x_i))      +  λ‖w‖²     (hinge loss)

                       where f(x) = wᵀx + b and y ∈ {−1, +1};   C ≈ 1/λ
```

### 8.2 Loss values as a function of the margin m = y·f(x)

| m = y·f(x) | Meaning | 0-1 loss | **Hinge** (SVM) | **Log loss** (LR) |
|---|---|---|---|---|
| −2 | badly wrong | 1 | 3 | 2.13 |
| −1 | wrong | 1 | 2 | 1.31 |
| 0 | on boundary | 1 | 1 | 0.69 |
| 0.5 | correct, inside margin | 0 | 0.5 | 0.47 |
| **1** | on margin | 0 | **0** | 0.31 |
| 2 | confidently correct | 0 | **0** | 0.13 |

Both are **convex upper bounds of the 0-1 loss**, which is hard to optimize directly.

### 8.3 Key differences

| Aspect | Logistic regression | Soft margin SVM |
|---|---|---|
| Loss | Log loss (smooth, never exactly 0) | **Hinge loss** (exactly 0 once `y·f ≥ 1`) |
| Points that affect the solution | **All** points (a little) | Only **support vectors** (margin violators and points on the margin) |
| Sparsity | Dense | **Sparse** in terms of support vectors |
| Output | **Probabilities** (calibrated) | Scores / classes (not probabilities by default) |
| Interpretation | Statistical (maximum likelihood, odds ratio) | Geometric (maximum margin) |
| Regularization knob | `C` (= 1/λ) | `C` (= 1/λ), same meaning |
| Non-linear extension | Polynomial features (or kernels, rarely) | **Kernel trick** (natural, next session) |
| Training | Gradient descent / LBFGS | QP/SMO (dual), or SGD on hinge loss |
| Far-away correct points | Still contribute slightly | Contribute **nothing** |
| Outliers | Affected (log loss grows linearly for large errors) | Also affected (hinge loss grows linearly), but any single point's influence is capped by `α ≤ C` |

### 8.4 In practice

- On many linearly separable-ish problems they give **very similar decision boundaries** and accuracy (in a quick test on the same data, both reached 0.865 CV accuracy).
- Choose **logistic regression** when you need **probabilities** or interpretability. Choose **SVM** when you want a strong margin-based classifier, or non-linear kernels.
- The two can be compared in scikit-learn using the same SGD solver: `SGDClassifier(loss="hinge")` (linear SVM) vs `SGDClassifier(loss="log_loss")` (logistic regression).

```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score

for name, clf in [("Logistic", LogisticRegression()),
                  ("Linear SVM", SVC(kernel="linear")),
                  ("SGD hinge", SGDClassifier(loss="hinge", random_state=0)),
                  ("SGD log", SGDClassifier(loss="log_loss", random_state=0))]:
    print(name, cross_val_score(make_pipeline(StandardScaler(), clf), X, y, cv=5).mean().round(3))
```

---

## 9. Summary & Cheat Sheet

1. **Hard margin** fails when data overlaps, and it is hypersensitive to outliers.
2. **Slack variable ξᵢ ≥ 0** measures how much a point violates the margin: `ξ = 0` correct side of the margin; `0 < ξ < 1` inside the margin but correct side; `ξ ≥ 1` misclassified. (Introduced by Vapnik, 1995.)
3. **Soft-margin SVM:** `min ½‖w‖² + C Σξᵢ  s.t.  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0`. Dual: same as before but with `0 ≤ αᵢ ≤ C`.
4. **C is the tuning knob:** large C → narrow margin, few violations, **overfit (high variance)**; small C → wide margin, many violations, **underfit (high bias)**. Tune with cross-validation on a log scale.
5. **Support vectors** now include points on the margin *and* those inside it or misclassified.
6. **Soft margin SVM = hinge loss + L2 regularization**; logistic regression = log loss + L2 regularization. Same family, different loss.
7. Still a **linear** classifier. Non-linear data needs **kernels** (next session).

```
        ┌── large C ──► narrow margin, low bias, high variance (overfit)
C ──────┤
        └── small C ──► wide margin, high bias, low variance (underfit)

  Loss per point:   ξ = max(0, 1 − y·f(x))      ← hinge loss
```
