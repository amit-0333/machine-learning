# PCA: Mathematics and Implementation from Scratch

> PCA finds new axes (directions) along which the data varies the most, and re-expresses the data using only the top few of them.

**Prerequisite:** [Curse of Dimensionality](./Curse_of_Dimensionality_README.md)

---

## Table of Contents

1. [Practical example on MNIST](#1-practical-example-on-mnist)
2. [Problem formulation](#2-problem-formulation)
3. [Covariance and covariance matrix](#3-covariance-and-covariance-matrix)
4. [Eigenvectors and eigenvalues](#4-eigenvectors-and-eigenvalues)
5. [Visualizing linear transformations](#5-visualizing-linear-transformations)
6. [Eigendecomposition of a covariance matrix](#6-eigendecomposition-of-a-covariance-matrix)
7. [How to solve PCA](#7-how-to-solve-pca)
8. [How to transform points](#8-how-to-transform-points)
9. [Code demo with visualization](#9-code-demo-with-visualization)
10. [Outro](#10-outro)

---

## 1. Practical example on MNIST

MNIST is a dataset of handwritten digits (0-9). Each image is **28 x 28 = 784 pixels**, so each image is a point in **784-dimensional space**.

**Problems:**
- We can't visualize 784 dimensions.
- Many pixels are useless (the borders are almost always black).
- Neighbouring pixels are highly correlated (redundant information).
- High dimensionality causes the curse of dimensionality (slow training, overfitting).

**Question:** can we describe each digit with, say, 50 numbers instead of 784 and lose very little information?

**PCA says yes.** Find a small set of new axes that captures most of the variation across all images.

---

## 2. Problem formulation

**Given:** a dataset `X` of shape `(n, d)`, with `n` samples and `d` features.

**Goal:** find `k < d` orthonormal directions (unit vectors, perpendicular to each other) such that when the data is **projected** onto them, the **variance is maximized**.

### Why maximize variance?

- Directions with high variance carry the most information (they separate points the most).
- Directions with tiny variance are nearly constant, so dropping them loses little.
- Maximizing the variance of the projection is equivalent to **minimizing the reconstruction error** (the squared distance between each point and its projection).

### One-direction version

Let `u` be a unit vector (`||u|| = 1`) and `X` be mean-centered. The variance of the data projected on `u` is:

```
Var(u) = (1/(n-1)) * sum_i (u^T x_i)^2  =  u^T Σ u
```

where `Σ` is the covariance matrix. So we solve:

```
maximize   u^T Σ u
subject to u^T u = 1
```

Using a Lagrange multiplier:

```
L(u, λ) = u^T Σ u - λ (u^T u - 1)
dL/du = 2Σu - 2λu = 0   =>   Σ u = λ u
```

**Key result:** the best direction `u` is an **eigenvector of the covariance matrix**, and the variance along it is its **eigenvalue** `λ`. To maximize variance, choose the eigenvector with the **largest eigenvalue**.

---

## 3. Covariance and covariance matrix

### Variance (spread of one feature)

```
Var(x) = (1/(n-1)) * sum (x_i - mean_x)^2
```

### Covariance (how two features move together)

```
Cov(x, y) = (1/(n-1)) * sum (x_i - mean_x)(y_i - mean_y)
```

| Sign | Meaning |
|---|---|
| Cov > 0 | When x goes up, y tends to go up |
| Cov < 0 | When x goes up, y tends to go down |
| Cov ~ 0 | No linear relationship |

### Covariance matrix

For `d` features, `Σ` is a `d x d` matrix:

```
        | Var(x1)     Cov(x1,x2)  ...  Cov(x1,xd) |
Σ =     | Cov(x2,x1)  Var(x2)     ...  Cov(x2,xd) |
        |   ...         ...       ...     ...     |
        | Cov(xd,x1)  Cov(xd,x2)  ...  Var(xd)    |
```

In matrix form, for mean-centered `X`:

```
Σ = (1 / (n-1)) * X^T X
```

**Properties:**
- Diagonal = variance of each feature.
- Off-diagonal = covariance between feature pairs.
- **Symmetric** (`Σ = Σ^T`).
- **Positive semi-definite**: all eigenvalues are >= 0.
- Because it's symmetric, its eigenvectors are **orthogonal**, which is exactly why PCA axes are perpendicular.

### Worked example (used throughout this README)

| Point | x | y |
|---|---|---|
| 1 | 2 | 1 |
| 2 | 3 | 3 |
| 3 | 4 | 3 |
| 4 | 5 | 5 |

Means: `mean_x = 3.5`, `mean_y = 3`. Centered points: `(-1.5,-2), (-0.5,0), (0.5,0), (1.5,2)`.

```
Var(x)    = (2.25 + 0.25 + 0.25 + 2.25) / 3 = 1.667
Var(y)    = (4 + 0 + 0 + 4) / 3             = 2.667
Cov(x,y)  = (3 + 0 + 0 + 3) / 3             = 2

Σ = | 1.667  2.000 |
    | 2.000  2.667 |
```

---

## 4. Eigenvectors and eigenvalues

For a square matrix `A`, a non-zero vector `v` is an **eigenvector** if:

```
A v = λ v
```

- `v` = eigenvector: a direction that **doesn't change direction** when `A` is applied.
- `λ` = eigenvalue: the factor by which that direction is **stretched** (or flipped if negative).

### Finding them

```
det(A - λI) = 0          ->  gives eigenvalues
(A - λI) v = 0           ->  gives eigenvectors
```

### Continuing the worked example

```
trace(Σ) = 1.667 + 2.667 = 4.333
det(Σ)   = 1.667*2.667 - 2*2 = 0.444

λ^2 - 4.333 λ + 0.444 = 0
λ1 = 4.228,   λ2 = 0.105
```

Eigenvector for `λ1` (solve `(Σ - λ1 I)v = 0`, then normalize):

```
v1 = (0.615, 0.788)
v2 = (-0.788, 0.615)     (perpendicular to v1)
```

**Interpretation:**
- `v1` is the direction of maximum variance (the "line of best fit" through the data).
- `λ1 / (λ1 + λ2) = 4.228 / 4.333 = 97.6%`, so **one axis captures 97.6% of the total variance**. We can go from 2D to 1D with almost no loss.

### Key facts for PCA

| Term | In PCA |
|---|---|
| Eigenvector of Σ | A **principal component** (a direction) |
| Eigenvalue of Σ | **Variance** along that principal component |
| Sum of eigenvalues | Total variance (= trace of Σ) |

---

## 5. Visualizing linear transformations

A matrix `A` acts on vectors as a **linear transformation**: it can rotate, stretch, shear, or flip space.

Apply `A` to a **unit circle** and you get an **ellipse**. For a symmetric matrix like Σ:
- The **ellipse axes point along the eigenvectors**.
- The **axis lengths are proportional to the eigenvalues**.

That is the geometric intuition: the covariance matrix describes an ellipse-shaped cloud, and PCA finds the ellipse's axes.

```python
import numpy as np
import matplotlib.pyplot as plt

Sigma = np.array([[1.667, 2.0],
                  [2.0,   2.667]])

# unit circle
theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])

# transform it
ellipse = Sigma @ circle

# eigen stuff
vals, vecs = np.linalg.eigh(Sigma)

plt.figure(figsize=(6, 6))
plt.plot(*circle, label="unit circle")
plt.plot(*ellipse, label="Σ · circle")
for i in range(2):
    v = vecs[:, i] * vals[i]
    plt.arrow(0, 0, v[0], v[1], head_width=0.1, color="red")
plt.axis("equal"); plt.grid(True); plt.legend()
plt.title("Eigenvectors are the axes of the transformed ellipse")
plt.show()
```

**Observation:** the red arrows (eigenvectors scaled by eigenvalues) line up with the ellipse's major and minor axes.

---

## 6. Eigendecomposition of a covariance matrix

Because `Σ` is symmetric, it can be factored as:

```
Σ = V Λ V^T
```

| Symbol | Meaning |
|---|---|
| `V` | Matrix whose **columns are eigenvectors** (orthonormal, so `V^T V = I`) |
| `Λ` | **Diagonal** matrix of eigenvalues `λ1 >= λ2 >= ... >= λd` |

**Interpretation of `Σ = V Λ V^T`** (read right to left):
1. `V^T`: rotate data into the eigenvector axes.
2. `Λ`: stretch each axis by its variance.
3. `V`: rotate back.

In the eigenvector coordinate system the covariance matrix is **diagonal**, which means the new features are **uncorrelated**. PCA is precisely a change of basis to this coordinate system.

---

## 7. How to solve PCA

### Algorithm

```
Input:  X (n x d), number of components k

1. Center the data:            Xc = X - mean(X)
   (Optional) Standardize:     divide each feature by its std
2. Compute covariance matrix:  Σ = Xc^T Xc / (n-1)          (d x d)
3. Eigendecomposition of Σ:    get eigenvalues and eigenvectors
4. Sort eigenvalues in descending order (and reorder eigenvectors)
5. Pick top-k eigenvectors:    W = [v1, v2, ..., vk]         (d x k)
6. Project the data:           Z = Xc W                      (n x k)
```

### Notes

- **Always center** the data. PCA assumes the cloud is centered at the origin.
- **Standardize** when features have different units or scales (e.g., age vs salary). Otherwise large-scale features dominate. (For MNIST pixels, which share one scale, dividing by 255 is enough.)
- Use `np.linalg.eigh` (not `eig`) for symmetric matrices; it's faster and returns real, sorted values.
- **SVD alternative:** most libraries (including scikit-learn) compute PCA via SVD of `Xc` instead of forming `Σ`, which is more numerically stable and faster for large `d`. The right singular vectors are the principal components, and `λ_i = s_i^2 / (n-1)`.

---

## 8. How to transform points

### Projection (dimensionality reduction)

```
Z = (X - μ) W          shape: (n x d)(d x k) = (n x k)
```

Each row of `Z` holds the **principal component scores** (the coordinates of that point in the new axes).

### Worked example: 2D -> 1D

With `v1 = (0.615, 0.788)` and centered points:

| Centered point | Calculation | Score on PC1 |
|---|---|---|
| (-1.5, -2) | -1.5(0.615) + -2(0.788) | **-2.50** |
| (-0.5, 0) | -0.5(0.615) | **-0.31** |
| (0.5, 0) | 0.5(0.615) | **0.31** |
| (1.5, 2) | 1.5(0.615) + 2(0.788) | **2.50** |

We've replaced 2 numbers per point with 1.

### Reconstruction (going back, approximately)

```
X_approx = Z W^T + μ
```

The information lost is the part of the data along the dropped components. **Reconstruction error = sum of the dropped eigenvalues.**

---

## 9. Code demo with visualization

### 9.1 PCA from scratch (NumPy)

```python
import numpy as np

def pca_from_scratch(X, k):
    mu = X.mean(axis=0)
    Xc = X - mu                                  # 1. center

    cov = np.cov(Xc, rowvar=False)               # 2. covariance matrix (d x d)

    eigvals, eigvecs = np.linalg.eigh(cov)       # 3. eigendecomposition

    order = np.argsort(eigvals)[::-1]            # 4. sort descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    W = eigvecs[:, :k]                           # 5. top-k components
    Z = Xc @ W                                   # 6. project

    explained_ratio = eigvals / eigvals.sum()
    return Z, W, mu, eigvals, explained_ratio
```

### 9.2 Test on the worked example

```python
X = np.array([[2, 1], [3, 3], [4, 3], [5, 5]], dtype=float)

Z, W, mu, eigvals, ratio = pca_from_scratch(X, k=1)

print("Eigenvalues:", eigvals)          # ~[4.228, 0.105]
print("PC1 direction:", W.ravel())      # ~[0.615, 0.788] (sign may be flipped)
print("Explained variance ratio:", ratio)   # ~[0.976, 0.024]
print("Projected scores:", Z.ravel())   # ~[-2.50, -0.31, 0.31, 2.50]
```

### 9.3 Visualize the principal axes

```python
import matplotlib.pyplot as plt

Xc = X - mu
plt.figure(figsize=(6, 6))
plt.scatter(Xc[:, 0], Xc[:, 1], label="centered data")

# principal axis
pc1 = W[:, 0]
t = np.linspace(-3, 3, 10)
plt.plot(t * pc1[0], t * pc1[1], "r--", label="PC1")

# projected points on PC1
proj = (Xc @ pc1)[:, None] * pc1[None, :]
plt.scatter(proj[:, 0], proj[:, 1], c="green", label="projections")
for p, q in zip(Xc, proj):
    plt.plot([p[0], q[0]], [p[1], q[1]], "gray", linewidth=0.8)

plt.axis("equal"); plt.grid(True); plt.legend()
plt.title("Projection of points onto PC1")
plt.show()
```

### 9.4 Verify against scikit-learn

```python
from sklearn.decomposition import PCA

sk = PCA(n_components=1).fit(X)
print(sk.components_)                  # same direction (up to sign)
print(sk.explained_variance_)          # matches the largest eigenvalue
print(sk.explained_variance_ratio_)
```

> Eigenvectors are only defined up to sign, so your result may be `-v` of scikit-learn's. This is normal.

---

## 10. Outro

**Key takeaways**

- PCA = find orthogonal directions of **maximum variance**.
- Those directions are the **eigenvectors of the covariance matrix**; the **eigenvalues** are the variances along them.
- Steps: **center, covariance, eigendecomposition, sort, pick top-k, project**.
- `Σ = V Λ V^T`: PCA rotates the data into an uncorrelated coordinate system.
- Reconstruction error equals the sum of the discarded eigenvalues.

**Next:** [PCA in Practice on MNIST](./PCA_MNIST_Practical_README.md)

### Practice questions

1. Why must data be centered before PCA?
2. Why are principal components orthogonal?
3. For `Σ = [[4,0],[0,1]]`, what are the principal components and the explained variance ratios?
4. Why is `np.linalg.eigh` preferred over `np.linalg.eig` here?
5. If the eigenvalues are `[5, 3, 1, 1]`, how much variance do the top 2 components capture?
6. Show that reconstruction error equals the sum of the dropped eigenvalues.
