# Support Vector Machines (SVM): Session 3 (Kernels in SVM)

## Table of Contents

1. [Problem with SVC](#1-problem-with-svc)
2. [Kernels Intuition](#2-kernels-intuition)
3. [Mathematics of SVM](#3-mathematics-of-svm)
   - [3.1 Unconstrained vs constrained optimization](#31-unconstrained-vs-constrained-optimization)
   - [3.2 Gradients and contour lines](#32-gradients-and-contour-lines)
   - [3.3 Worked example: maximize x²y on a circle](#33-worked-example-maximize-xy-on-a-circle)
   - [3.4 Lagrange multipliers](#34-lagrange-multipliers)
   - [3.5 Back to SVM: the dual and where the kernel appears](#35-back-to-svm-the-dual-and-where-the-kernel-appears)
4. [Why is it Called a Trick?](#4-why-is-it-called-a-trick)
5. [Types of Kernels](#5-types-of-kernels)
6. [Code](#6-code)
7. [Tuning C and gamma](#7-tuning-c-and-gamma)
8. [Summary & Cheat Sheet](#8-summary--cheat-sheet)

> **Note:** much of the source PDF (kernel intuition, code, types of kernels, the "trick", and the hand-drawn contour/gradient plots) was handwritten or image-only, so those parts are written from the standard treatment of each heading. The gradient/contour definitions follow your typed notes. The page order in the PDF puts the maths after the kernel headings; here the maths comes before "Why is it called a trick?" because the trick depends on it.

---

## 1. Problem with SVC

A (hard or soft margin) **Support Vector Classifier** draws a **straight line / flat hyperplane**. It works when the classes are (mostly) linearly separable, but fails on data where the true boundary is **curved**:

- one class **surrounded** by the other (concentric circles),
- interleaving half-moons,
- XOR-type patterns.

No choice of `C` fixes this, because the *shape* of the boundary is the limitation, not the margin.

**Experiment:** two concentric rings (`make_circles`), linear SVM:

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Linear SVC | 0.65 | **0.51** (about a coin flip) |
| Kernel SVM (RBF) | 1.00 | **1.00** |

Previously, with logistic regression, we fixed this by **creating polynomial features by hand**. Kernels give SVM a much more elegant and efficient way to do the same.

---

## 2. Kernels Intuition

### 2.1 The big idea

> If the data is not linearly separable in the original space, **map it into a higher-dimensional space** where it **is** linearly separable, and draw a flat hyperplane there. Seen back in the original space, that flat hyperplane is a **curved boundary**.

```
Original space (2D)                  Feature space (3D)
 not separable by a line      φ      separable by a plane
   ○ ○ ○ ○                  ───────►   ○ ○ ○ ○  (high z)
  ○ ✕ ✕ ○                              ─ ─ ─ ─ ─  ← flat plane
   ○ ✕ ✕ ○                              ✕ ✕ ✕ ✕   (low z)
   ○ ○ ○ ○
```

### 2.2 1-D example

Points on a line: class ✕ at `x = −1, 0, 1`, class ○ at `x = −3, −2, 2, 3`. No single threshold separates them (✕ is in the middle).

Map each point to 2D: `φ(x) = (x, x²)`.

| x | φ(x) = (x, x²) | class |
|---|---|---|
| −3 | (−3, 9) | ○ |
| −2 | (−2, 4) | ○ |
| −1 | (−1, 1) | ✕ |
| 0 | (0, 0) | ✕ |
| 1 | (1, 1) | ✕ |
| 2 | (2, 4) | ○ |
| 3 | (3, 9) | ○ |

Now the horizontal line `x² = 2.5` separates the classes perfectly: ○ has `x² ≥ 4`, ✕ has `x² ≤ 1`. A **linear** separator in `(x, x²)` space = two thresholds (`|x| = 1.58`) in the original space.

### 2.3 2-D circles example

Concentric rings with features `(x₁, x₂)`. Add the feature `z = x₁² + x₂²` (squared distance from origin):

- Inner ring: small `z`. Outer ring: large `z`.
- The plane `z = c` separates them; in the original space `x₁² + x₂² = c` is a **circle**.

### 2.4 The catch (and the rescue)

Explicitly creating the new features `φ(x)` can be **extremely expensive** (even infinite-dimensional). The rescue: SVM's dual form only needs **dot products** between points. Kernels compute the dot product in the high-dimensional space **without ever building** `φ(x)`. That's the **kernel trick** (sections 3.5 and 4).

---

## 3. Mathematics of SVM

SVM training is a **constrained optimization problem**. This section builds the tools needed to understand why the kernel appears.

### 3.1 Unconstrained vs constrained optimization

| | Unconstrained | Constrained |
|---|---|---|
| Form | `min L(β)` | `min f(x)  s.t.  g(x) = 0` or `h(x) ≥ 0` |
| Condition for the optimum | `∂L/∂β = 0` (gradient is zero) | Gradient of the objective is balanced by gradient of the constraint (Lagrange) |
| Examples | **Linear regression** (OLS closed form, or gradient descent), **logistic regression** (log loss, gradient descent) | **SVM**: `min ½‖w‖² + C·Σξᵢ  s.t.  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0` |

In the notes:

- Linear/logistic regression: `L = argmin_β (1/n) Σ (yᵢ − ŷᵢ)²` (or log loss) → set `∂L/∂β = 0` (find the bottom of the curve).
- SVM: `argmin over (A, B, C)  of  ‖w‖/2 + C·(1/n)·Σξᵢ  such that  yᵢ(Ax₁ᵢ + Bx₂ᵢ + C) ≥ 1 − ξᵢ` is a **constrained optimization problem**: the **main function** is the thing to minimize, and the **restriction** (constraint) limits which `(A, B, C)` are acceptable.

### 3.2 Gradients and contour lines

**Gradient.** The gradient of a function at a point is a vector that points in the direction of the **steepest ascent** of the function at that point. Its magnitude (length) equals the **rate of increase** in that direction.

```
∇f = [ ∂f/∂x ,  ∂f/∂y ]
```

**Contour lines (level sets).** Curves connecting points where the function has the **same value**. For a 2-D function, they're like lines of constant altitude on a topographic map.

**Key relationships**

1. The gradient at a point is **perpendicular (orthogonal)** to the contour line through that point. The contour is the direction of *no change*; the gradient is the direction of *maximum change*.
2. The gradient points in the direction where the function **increases most rapidly**. Walking along a contour doesn't change `f`; to start climbing as steeply as possible, go in the gradient direction.
3. The **magnitude** of the gradient shows steepness: **contours close together** → function changes rapidly → **large** gradient; **contours far apart** → changes slowly → **small** gradient.

### 3.3 Worked example: maximize x²y on a circle

**Problem (from the notes):**

```
maximize     f(x, y) = x²·y
subject to   x² + y² = 1        (constraint: (x, y) must lie on the unit circle)
```

**Without the constraint** there's no maximum: `(x, y) = (5, 7)` gives `5²·7 = 175`, and larger values give even more. But `(5, 7)` is **not acceptable** because `25 + 49 ≠ 1`.

**Only points on the circle are allowed.** For example `(1/√2, 1/√2)` is acceptable (`½ + ½ = 1`) and gives `f = ½ · 1/√2 = 1/(2√2) ≈ 0.354`. Can we do better on the circle?

**Geometric intuition (the contour picture in the notes).**

- Plot the contours of `f(x, y) = x²y` (curves where `z = x²y` is constant, e.g., `z = 0.2, 0.5, 4, 5, 10, −4, −5, −10, ...`). The 3-D surface looks like a saddle.
- Draw the constraint circle on top.
- Walk along the circle and watch which contour you're on. The highest value of `z` the circle can reach is at the point where the circle **just touches (is tangent to)** a contour, without crossing into higher ones.
- Crossing a contour means `z` can still be increased by moving along the circle, so that's not the optimum. At the optimum the **circle is tangent to the contour**.

**Key fact:** at a tangent point, the gradient of `f` (perpendicular to its contour) and the gradient of the constraint `g = x² + y²` (perpendicular to the circle) point in the **same (or opposite) direction**, so they are parallel:

```
∇f = λ ∇g
```

**Solve it:**

```
∇f = (2xy, x²)          ∇g = (2x, 2y)

2xy = λ·2x     ⇒ y = λ        (if x ≠ 0)
x²  = λ·2y     ⇒ x² = 2λ²
x² + y² = 1    ⇒ 2λ² + λ² = 1 ⇒ λ = ±1/√3
```

- For `λ = +1/√3`: `y = 1/√3`, `x² = 2/3` ⇒ `x = ±√(2/3)`
  `f = (2/3)(1/√3) = 2/(3√3) ≈ 0.385`  ← **maximum**, at `(±√(2/3), 1/√3)`.
- For `λ = −1/√3`: `y = −1/√3`: `f ≈ −0.385` ← minimum.

So the best acceptable `(x, y)` is `(±√(2/3), 1/√3)` with `z ≈ 0.385`, better than the `0.354` we found by guessing. (Checked numerically with an optimizer.)

### 3.4 Lagrange multipliers

The recipe that turns the tangency condition into algebra:

1. Write the **Lagrangian**: `L(x, y, λ) = f(x, y) − λ·( g(x, y) − 1 )`.
2. Set **all partial derivatives to 0**: `∂L/∂x = 0`, `∂L/∂y = 0`, `∂L/∂λ = 0` (the last one recovers the constraint).
3. Solve the resulting equations.

**For inequality constraints** (`h(x) ≥ 0`, which is what SVM has) we use multipliers `αᵢ ≥ 0` and the **KKT conditions**, including **complementary slackness**: `αᵢ · (constraint_i) = 0`, i.e. a constraint either is *tight* (active, `α > 0`) or has multiplier 0.

### 3.5 Back to SVM: the dual and where the kernel appears

From Sessions 1 and 2, the soft-margin SVM Lagrangian leads to

```
w = Σᵢ αᵢ yᵢ xᵢ                   Σᵢ αᵢ yᵢ = 0                0 ≤ αᵢ ≤ C
```

and the **dual problem**:

```
maximize   Σᵢ αᵢ  −  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ ⟨xᵢ, xⱼ⟩
subject to 0 ≤ αᵢ ≤ C ,   Σᵢ αᵢ yᵢ = 0
```

and the **prediction**:

```
f(x) = Σ_{i ∈ SV} αᵢ yᵢ ⟨xᵢ, x⟩ + b
```

> **Observation:** the training data appears **only through dot products `⟨xᵢ, xⱼ⟩`**, in both training and prediction. Never `x` alone.

**The kernel step:** replace every dot product by a kernel function `K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩`:

```
maximize   Σᵢ αᵢ  −  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
f(x) = Σ_{i ∈ SV} αᵢ yᵢ K(xᵢ, x) + b
```

Nothing else in the algorithm changes. We now have a **linear SVM in the feature space `φ(x)`**, i.e. a **non-linear** SVM in the original space, without ever computing `φ`.

(This is also why we need the dual: the primal contains `w` explicitly, and in feature space `w` could be infinite-dimensional.)

---

## 4. Why is it Called a Trick?

Because we get the result of a very expensive computation **without doing it**.

### 4.1 Example: degree-2 polynomial kernel

For `x = (x₁, x₂)` and `z = (z₁, z₂)`, define the explicit map

```
φ(x) = ( x₁² ,  √2·x₁x₂ ,  x₂² )
```

Then

```
⟨φ(x), φ(z)⟩ = x₁²z₁² + 2x₁x₂z₁z₂ + x₂²z₂²
             = (x₁z₁ + x₂z₂)²
             = ⟨x, z⟩²
```

So the kernel `K(x, z) = ⟨x, z⟩²` gives the 3-D dot product using only a 2-D dot product and a square.

**Numeric check:** `x = (1, 2)`, `z = (3, −1)`:

- Via `φ`: `φ(x) = (1, 2.83, 4)`, `φ(z) = (9, −4.24, 1)` → `9 − 12 + 4 = 1`
- Via kernel: `⟨x, z⟩² = (3 − 2)² = 1` ✓

### 4.2 The savings

| | Explicit mapping `φ` | Kernel trick |
|---|---|---|
| Degree-3 polynomial, 100 features | `C(103, 3) = 176,851` new features per sample | One 100-dim dot product, then cube it |
| Gaussian (RBF) | **Infinite** dimensions, impossible to build | One distance computation + `exp` |

### 4.3 Why RBF has infinite dimensions

```
K(x, z) = exp(−γ‖x − z‖²) = exp(−γ‖x‖²) · exp(−γ‖z‖²) · exp(2γ⟨x, z⟩)

exp(2γ⟨x,z⟩) = Σₖ (2γ)ᵏ ⟨x,z⟩ᵏ / k!      ← an infinite sum of polynomial kernels of every degree
```

So RBF behaves like polynomial features of **all degrees** at once. We can't write `φ` down, yet we can compute `K` instantly. That's the "trick".

### 4.4 What makes a function a valid kernel? (Mercer's condition)

`K` is valid if it's **symmetric** and for any set of points the **Gram matrix** `Kᵢⱼ = K(xᵢ, xⱼ)` is **positive semi-definite**. Then a feature map `φ` is guaranteed to exist (even if we never see it), and the optimization stays **convex**.

### 4.5 Not only for SVM

The trick works for any algorithm that depends only on dot products: kernel PCA, kernel ridge regression, Gaussian processes, kernel k-means, etc.

---

## 5. Types of Kernels

| Kernel | Formula | Key hyperparameters | Boundary / use |
|---|---|---|---|
| **Linear** | `K(x, z) = xᵀz` | none (just `C`) | Straight hyperplane. Best for high-dimensional / sparse data (text) and big datasets |
| **Polynomial** | `K(x, z) = (γ·xᵀz + r)ᵈ` | `degree d`, `gamma γ`, `coef0 r` | Curves up to degree `d`; captures feature interactions |
| **RBF / Gaussian** | `K(x, z) = exp(−γ‖x − z‖²)` | `gamma γ` | Very flexible, smooth, local. **Best default for non-linear data** |
| **Sigmoid** | `K(x, z) = tanh(γ·xᵀz + r)` | `gamma γ`, `coef0 r` | Mimics a neural-net neuron. Not always a valid (PSD) kernel; rarely the best choice |
| **Custom / precomputed** | any valid `K` | n/a | String kernels, graph kernels, chi-squared for histograms, ... |

### 5.1 RBF intuition

`K(x, z)` is a **similarity score**: `K = 1` when `x = z`, and it decays toward 0 as the points move apart. The decision function

```
f(x) = Σ αᵢ yᵢ exp(−γ‖xᵢ − x‖²) + b
```

is a weighted sum of **bumps** centred on the support vectors, each positive-class bump pushing the score up and each negative-class bump pushing it down. `γ` sets the bump width (`γ = 1/(2σ²)`).

| γ | Bump width | Boundary | Risk |
|---|---|---|---|
| **Small** | Wide, far-reaching influence | Smooth, nearly linear | **Underfitting** (high bias) |
| **Large** | Narrow, very local influence | Wiggly, islands around single points | **Overfitting** (high variance) |

### 5.2 Polynomial notes

- `degree` controls complexity (like the polynomial features in the logistic regression notes).
- `coef0 = r` shifts between lower and higher degree terms. With `r = 0` and an **odd** degree, the kernel is an odd function, which can't model symmetric shapes like rings.
- Large degrees lead to numerical trouble and overfitting.

### 5.3 Comparison on concentric circles (test result)

| Kernel | Train acc. | Test acc. |
|---|---|---|
| Linear | 0.65 | 0.51 |
| Polynomial, degree 2 | 1.00 | 1.00 |
| Polynomial, degree 3 (`coef0 = 0`) | 0.67 | 0.68 (odd degree, can't capture the ring) |
| **RBF** | **1.00** | **1.00** |
| Sigmoid | 0.60 | 0.63 |

### 5.4 Which kernel should I use?

1. Start with **linear** if you have many features (text, genomics) or a huge dataset. It's fast.
2. Try **RBF** as the general-purpose non-linear choice.
3. Try **polynomial** if you know interactions of a given degree matter.
4. Rarely: sigmoid, or a custom kernel for structured data.
5. Always compare by **cross-validation**.

---

## 6. Code

### 6.1 Linear SVC vs kernel SVM on non-linear data

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

for kernel, params in [("linear", {}), ("poly", {"degree": 2}), ("rbf", {}), ("sigmoid", {})]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, **params))
    model.fit(X_train, y_train)
    print(f"{kernel:8} train={model.score(X_train, y_train):.3f}  test={model.score(X_test, y_test):.3f}")
```

### 6.2 Plot the decision boundary

```python
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
    clf = SVC(kernel=kernel, degree=2, C=1).fit(X_train, y_train)
    DecisionBoundaryDisplay.from_estimator(clf, X_train, ax=ax, response_method="predict", alpha=0.3)
    ax.scatter(*X_train.T, c=y_train, edgecolors="k", s=15)
    ax.scatter(*clf.support_vectors_.T, s=80, facecolors="none", edgecolors="g")
    ax.set_title(kernel)
plt.show()
```

### 6.3 Seeing the trick in code

Explicit feature map + linear SVM **equals** a polynomial-kernel SVM:

```python
from sklearn.svm import LinearSVC

# explicit map: (x1, x2) → (x1, x2, x1² + x2²)
phi = lambda X: np.c_[X, (X ** 2).sum(axis=1)]

explicit = LinearSVC(C=10, max_iter=100000).fit(phi(X_train), y_train)
print("Linear SVM on explicit φ(x):", explicit.score(phi(X_test), y_test))     # 1.0

# kernel identity: ⟨φ(x), φ(z)⟩ = ⟨x, z⟩²
x, z = np.array([1., 2.]), np.array([3., -1.])
phi2 = lambda v: np.array([v[0]**2, np.sqrt(2) * v[0] * v[1], v[1]**2])
print(phi2(x) @ phi2(z), (x @ z) ** 2)                                         # 1.0 1.0
```

### 6.4 Custom kernel

```python
def my_poly_kernel(A, B):
    return (A @ B.T + 1) ** 2            # K(a, b) = (aᵀb + 1)²   (must return the Gram matrix)

clf = SVC(kernel=my_poly_kernel).fit(X_train, y_train)
print(clf.score(X_test, y_test))

# or precompute the Gram matrix yourself
from sklearn.metrics.pairwise import rbf_kernel
G_train = rbf_kernel(X_train, X_train, gamma=1.0)
clf = SVC(kernel="precomputed").fit(G_train, y_train)
print(clf.score(rbf_kernel(X_test, X_train, gamma=1.0), y_test))
```

### 6.5 Useful `SVC` arguments

| Argument | Meaning |
|---|---|
| `kernel` | `"linear"`, `"poly"`, `"rbf"` (default), `"sigmoid"`, `"precomputed"`, or a callable |
| `C` | Soft-margin penalty (default 1.0) |
| `gamma` | Kernel width for rbf/poly/sigmoid. `"scale"` (default) = `1/(n_features · X.var())`, `"auto"` = `1/n_features`, or a float |
| `degree` | Degree for the `poly` kernel (default 3) |
| `coef0` | Independent term `r` in `poly` and `sigmoid` |
| `class_weight` | e.g. `"balanced"` for imbalanced data |
| `probability` | `True` adds (slow) probability estimates |

---

## 7. Tuning C and gamma

For the RBF kernel there are **two** hyperparameters, and they interact:

| | Small γ | Large γ |
|---|---|---|
| **Small C** | Very smooth, underfit | Underfit/noisy |
| **Large C** | Smooth but follows the data | **Overfit** (wiggly, one island per point) |

- **C:** margin softness (penalty for violations).
- **γ:** flexibility of the boundary.

### 7.1 Effect of γ (half-moons, C = 1, test result)

| γ | Train acc. | Test acc. | # support vectors | Reading |
|---|---|---|---|---|
| 0.01 | 0.825 | 0.842 | 182 | underfit |
| 0.1 | 0.839 | 0.842 | 117 | underfit |
| **1** | 0.936 | **0.908** | 73 | good |
| 10 | 0.950 | 0.900 | 102 | still fine |
| 100 | 0.982 | 0.883 | 241 | overfitting begins |
| 1000 | 1.000 | **0.700** | 279 | **severe overfit** (every point is its own island) |

### 7.2 Grid search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.25, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = GridSearchCV(
    pipe,
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]},
    cv=5, n_jobs=-1,
).fit(X_train, y_train)

print(grid.best_params_, round(grid.best_score_, 3), round(grid.score(X_test, y_test), 3))
# e.g. {'svc__C': 1, 'svc__gamma': 10}  CV ≈ 0.95  test ≈ 0.91
```

Search `C` and `γ` on **log scales** (see the Hyperparameter Tuning notes; `RandomizedSearchCV` with `loguniform` works well).

### 7.3 Practical notes

- **Always scale features.** RBF and polynomial kernels depend on distances / dot products.
- **Training cost:** kernel SVM needs the `n × n` Gram matrix; fit time grows roughly between `O(n²)` and `O(n³)`. It's comfortable up to ~10⁴ to 10⁵ samples. For bigger data use `LinearSVC`, `SGDClassifier(loss="hinge")`, or kernel approximations (`Nystroem`, `RBFSampler`) followed by a linear model.
- **Prediction cost** is proportional to the number of support vectors.
- Multiclass: `SVC` uses one-vs-one internally; `decision_function_shape="ovr"` only changes the output shape.

---

## 8. Summary & Cheat Sheet

1. **Problem:** a linear SVC can't separate curved/non-linear data (rings, moons, XOR).
2. **Idea:** map data to a higher-dimensional space `φ(x)` where it's linearly separable; a flat hyperplane there is a curve in the original space.
3. **Maths prerequisite:** SVM is a **constrained optimization** problem solved with **Lagrange multipliers / KKT**. At a constrained optimum the **gradient of the objective is parallel to the gradient of the constraint** (`∇f = λ∇g`); **gradients are perpendicular to contour lines** and point toward steepest ascent. (Example: max `x²y` on `x² + y² = 1` gives `(±√(2/3), 1/√3)`, `f ≈ 0.385`.)
4. **Dual form:** training and prediction use the data **only through dot products** `⟨xᵢ, xⱼ⟩`.
5. **Kernel trick:** replace `⟨xᵢ, xⱼ⟩` with `K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩`, giving the feature-space result **without computing φ** (even for infinite-dimensional `φ`).
6. **Kernels:** linear, polynomial `(γxᵀz + r)ᵈ`, **RBF** `exp(−γ‖x−z‖²)`, sigmoid, custom. RBF is the usual default.
7. **Hyperparameters:** `C` (margin softness) and `γ` (boundary flexibility), plus `degree`/`coef0` for polynomial. Large `C` or large `γ` → overfit; small → underfit. Tune with cross-validation on log scales.
8. **Practice:** scale features, start simple (linear → RBF), watch training time on large data.

```
Non-linear data ──φ(x)──► higher-dim space (linearly separable) ──► max-margin hyperplane
        │                                │
        └──── never compute φ ───────────┘
              only K(x, z) = ⟨φ(x), φ(z)⟩      f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b
```

| Question | Answer |
|---|---|
| Why do kernels help? | They give non-linear boundaries while keeping a convex, linear max-margin problem |
| Why is it a "trick"? | We use `K(x, z)` and never construct the high-dimensional `φ(x)` |
| Which kernel first? | Linear for huge/sparse data; otherwise RBF |
| What does γ do? | Larger γ → narrower influence → more complex boundary |
| What does C do? | Larger C → fewer margin violations → narrower margin, higher variance |
