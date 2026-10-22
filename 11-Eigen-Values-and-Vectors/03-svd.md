# Singular Value Decomposition (SVD)

> SVD factors **any** matrix (square or not) into three simple pieces: a rotation, a scaling, and another rotation. `A = U Σ Vᵀ`. It is the tool that makes PCA fast and works on matrices where eigen decomposition cannot.

**Prerequisite:** [Eigen Decomposition and Kernel PCA](./Eigen_Decomposition_and_Kernel_PCA_README.md)

---

## Table of Contents

1. [Recap: why SVD?](#1-recap-why-svd)
2. [Non-square matrices](#2-non-square-matrices)
3. [Rectangular diagonal matrix](#3-rectangular-diagonal-matrix)
4. [What is SVD?](#4-what-is-svd)
5. [Applications of SVD](#5-applications-of-svd)
6. [The SVD equation: shapes](#6-the-svd-equation-shapes)
7. [Relationship with eigen decomposition](#7-relationship-with-eigen-decomposition)
8. [Geometric intuition](#8-geometric-intuition)
9. [How to calculate SVD](#9-how-to-calculate-svd)
10. [SVD in PCA](#10-svd-in-pca)
11. [Summary and practice questions](#11-summary-and-practice-questions)

---

## 1. Recap: why SVD?

```
PCA  ──►  eigenvalues / eigenvectors
          └──►  eigen decomposition  ──►  matrix decomposition / factorization
                                              └──►  SVD   (important, but harder to grasp)
```

- PCA is usually *explained* with eigen decomposition of the covariance matrix.
- Eigen decomposition only works on **square** matrices (and needs diagonalizability).
- **SVD** is the more general decomposition, and it is what libraries (including scikit-learn) actually use to compute PCA.

---

## 2. Non-square matrices

A **square** matrix is `2 x 2`, `3 x 3`, and so on. A **non-square (rectangular)** matrix is `2 x 3`, `3 x 2`, `4 x 2`, and so on. Non-square matrices are also **linear transformations**, but they move vectors **between spaces of different dimension**.

### Shape rule: `m x n`

```
        A  is  m x n
              │   │
              │   └── n = input dimension   (vector A acts on lives in n-D)
              └────── m = output dimension  (result lives in m-D)
```

| Shape | Input space | Output space | Effect |
|---|---|---|---|
| 2 x 2 | 2D | 2D | Same dimension |
| **2 x 3** | 3D | 2D | Dimension **reduced** (3D to 2D) |
| **3 x 2** | 2D | 3D | Dimension **increased** (2D to 3D) |
| 4 x 2 | 2D | 4D | 2D plane embedded in 4D |

### Columns tell you where the basis vectors land

The columns of a matrix are the images of the standard basis vectors.

```
A = | 1  0 |      A (1,0) = (1, 2, 3)   <- first column
    | 2  0 |      A (0,1) = (0, 0, 1)   <- second column
    | 3  1 |

3 x 2:  2D input  ->  3D output
```

So a 3 x 2 matrix places a 2D plane inside 3D space; the 2D input *gets 3D coordinates*.

```
| 1  3  5 | | 1 |   | 1·1 + 3·2 + 5·3 |   | 22 |
| 2  4  6 | | 2 | = | 2·1 + 4·2 + 6·3 | = | 28 |      (2 x 3) times (3 x 1) = (2 x 1)
            | 3 |
```

A 2 x 3 matrix takes a 3D input and produces a 2D output (3D to 2D).

---

## 3. Rectangular diagonal matrix

> A matrix that would be diagonal if it were square, but is rectangular because of **extra rows or columns of zeros**.

```
2 x 3:  | a  0  0 |        3 x 2:  | a  0 |
        | 0  b  0 |                | 0  b |
                                     | 0  0 |   <- extra row of zeros
```

Compare with a normal square diagonal matrix `[[a, 0], [0, b]]`, which only scales.

### Two transformations in one

A rectangular diagonal matrix does **two** things: it **scales** and it **changes dimension**. It factors into a square diagonal matrix and a "dimension changer":

**2 x 3 (3D to 2D): drop a dimension, then scale**

```
| a  0  0 |   | a  0 | | 1  0  0 |
| 0  b  0 | = | 0  b | | 0  1  0 |
   2 x 3        2 x 2      2 x 3
               (scaling)  (drop the 3rd coordinate)
```

Geometrically a 3D box is flattened into a 2D (squished) parallelogram.

**3 x 2 (2D to 3D): scale, then add a dimension**

```
| a  0 |   | 1  0 |
| 0  b | = | 0  1 | | a  0 |
| 0  0 |   | 0  0 | | 0  b |
   3 x 2     3 x 2     2 x 2
          (add zero z-coordinate)  (scaling)
```

This is exactly the role of **Σ** in SVD.

---

## 4. What is SVD?

SVD is a **matrix decomposition (factorization)** that splits a matrix `A` into **three** matrices:

```
A = U Σ Vᵀ
```

| Matrix | Name | Type |
|---|---|---|
| `U` | **Left singular vectors** | Orthogonal matrix |
| `Σ` (Sigma) | **Singular values** on its diagonal | **Rectangular diagonal** matrix |
| `V` | **Right singular vectors** (`Vᵀ` is used in the formula) | Orthogonal matrix |

**Orthogonal** means the columns are perpendicular unit vectors. For columns `(a, b)` and `(c, d)`:

```
ac + bd = 0        (90° apart)
a² + b² = 1        (unit length)
```

and therefore `U⁻¹ = Uᵀ`, `V⁻¹ = Vᵀ`.

---

## 5. Applications of SVD

| Field | Use |
|---|---|
| **Machine learning / data science** | **PCA** (dimensionality reduction); **recommendation systems** such as collaborative filtering (e.g., Netflix movie recommendations) |
| **NLP** | **Latent Semantic Analysis (LSA)**: reduce a term-document matrix to find relationships between terms and documents |
| **Computer vision** | **Image compression**: keep only the largest singular values and their vectors |
| **Signal processing** | Separate useful signal from noise (mobile communications, audio) |
| **Numerical linear algebra** | Matrix inversion and solving linear systems; numerically stable for ill-conditioned problems |
| **Psychometrics** | Construction and scoring of tests; extracting latent traits |
| **Bioinformatics** | Gene expression analysis; finding underlying activity patterns |
| **Quantum computing** | Quantum state tomography |

### Mini demo: low-rank approximation (image compression idea)

Keeping only the top `k` singular values gives the best rank-`k` approximation of `A`:

```
A ≈ U[:, :k] · Σ[:k, :k] · Vᵀ[:k, :]
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").mean(axis=2)          # grayscale (427 x 640)
U, s, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, k in zip(axes, [5, 20, 50, len(s)]):
    approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    ax.imshow(approx, cmap="gray"); ax.set_title(f"k = {k}"); ax.axis("off")
plt.show()

# storage: full image = m*n numbers; rank-k = k*(m + n + 1) numbers
m, n = img.shape
print(m * n, 50 * (m + n + 1))
```

Small `k` already captures the main structure, with far fewer numbers stored.

---

## 6. The SVD equation: shapes

For **any** `m x n` matrix `A`:

```
A      =   U      Σ      Vᵀ
m x n    m x m   m x n   n x n
```

| Factor | Shape | Notes |
|---|---|---|
| `A` | `m x n` | Any matrix (the rectangular case is the interesting one) |
| `U` | `m x m` | Orthogonal; lives in the **output** space |
| `Σ` | `m x n` | **Rectangular diagonal**; singular values `σ1 >= σ2 >= ... >= 0` |
| `Vᵀ` | `n x n` | Orthogonal; lives in the **input** space |

Check: `(m x m)(m x n)(n x n) = (m x n)` ✓

**The big question:** what are `U`, `Σ`, and `V`? The next section answers it using eigen decomposition.

---

## 7. Relationship with eigen decomposition

### The problem with eigen decomposition

```
A = V Λ V⁻¹        needs A to be square (n x n), and diagonalizable
```

A non-square matrix has no eigenvalues. But we can **make** a square symmetric matrix from any `A`:

```
A Aᵀ    ->  m x m   (square, symmetric)
Aᵀ A    ->  n x n   (square, symmetric)
```

Symmetric matrices **do** have a nice eigen decomposition (orthogonal eigenvectors, real eigenvalues). Example:

```
A = | 1  0 |       Aᵀ = | 1  2  1 |       AᵀA = | 6  1 |   <- 2 x 2, symmetric
    | 2  0 |            | 0  0  1 |             | 1  1 |
    | 1  1 |
   (3 x 2)               (2 x 3)
```

### Derivation

Substitute `A = U Σ Vᵀ` and use `VᵀV = I`, `UᵀU = I`, `(UΣVᵀ)ᵀ = VΣᵀUᵀ`:

```
A Aᵀ = (U Σ Vᵀ)(V Σᵀ Uᵀ) = U Σ (VᵀV) Σᵀ Uᵀ = U (Σ Σᵀ) Uᵀ

Aᵀ A = (V Σᵀ Uᵀ)(U Σ Vᵀ) = V Σᵀ (UᵀU) Σ Vᵀ = V (Σᵀ Σ) Vᵀ
```

So:

```
A Aᵀ = U X Uᵀ      where X = Σ Σᵀ        <- this is an eigen decomposition of A Aᵀ
Aᵀ A = V Y Vᵀ      where Y = Σᵀ Σ        <- this is an eigen decomposition of Aᵀ A
```

(`X` and `Y` are diagonal, so they play the role of `Λ`.)

### What this tells us

| Part | How to get it |
|---|---|
| **`U`** (left singular vectors) | **Eigenvectors of `A Aᵀ`** (columns of `U`) |
| **`V`** (right singular vectors) | **Eigenvectors of `Aᵀ A`** (columns of `V`) |
| **`Σ`** (singular values) | **Square roots of the eigenvalues** of `A Aᵀ` (or `Aᵀ A`) |

### Why square roots? Worked on a 2 x 3 `Σ`

```
Σ = | a  0  0 |
    | 0  b  0 |

X = Σ Σᵀ = | a²  0  |           eigenvalues of A Aᵀ:  a², b²           (2 x 2)
           | 0   b² |

Y = Σᵀ Σ = | a²  0   0 |       eigenvalues of Aᵀ A:  a², b², 0         (3 x 3)
           | 0   b²  0 |
           | 0   0   0 |
```

- `A Aᵀ` and `Aᵀ A` share the **same non-zero eigenvalues** (`a²`, `b²`).
- Taking square roots gives `a`, `b`: the **singular values**.
- Extra zeros in the bigger matrix correspond to the "extra" dimension.

```
singular value  σ = √(eigenvalue of AᵀA)
```

**Summary picture:**

```
A = U Σ Vᵀ
    │ │  └─ eigenvectors of AᵀA
    │ └──── √(eigenvalues of AAᵀ or AᵀA)
    └────── eigenvectors of AAᵀ
```

### Eigen decomposition is the special case

For a **symmetric** matrix, `A = V Λ Vᵀ` is already an SVD-like form (rotate, scale, rotate back). So eigen decomposition of a symmetric positive semi-definite matrix is a special case of SVD, while SVD works for **every** matrix.

---

## 8. Geometric intuition

```
A = U Σ Vᵀ
```

Apply it to a vector `x` from **right to left**:

| Step | Matrix | Action | Space |
|---|---|---|---|
| 1 | `Vᵀ` | **Rotate** (or reflect) the input | Input space (`n`-D) |
| 2 | `Σ` | **Scale** each axis by a singular value, and **change dimension** (drop or add axes) | Input to output |
| 3 | `U` | **Rotate** the result | Output space (`m`-D) |

```
input (n-D)  ──Vᵀ──►  rotated input  ──Σ──►  scaled, dim changed (m-D)  ──U──►  output (m-D)
```

For a symmetric square matrix (eigen decomposition `A = V Λ V⁻¹`), the same idea holds: rotate, scale, rotate back.

### Σ splits into two sub-steps

As seen in section 3, `Σ` = scaling `x` dimension change:

- **Σ1** and **Σ2**: one is the scaling matrix `[[a, 0], [0, b]]`, the other changes dimension.
- 2 x 3 case: `n > m`, so dimension is **reduced** (3D to 2D).
- 3 x 2 case: `m > n`, so dimension is **increased** (2D to 3D).

### Demo 1: `A` is 2 x 3 (3D input, 2D output)

A 3D cube goes through:

```
3D cube ──Vᵀ──► rotated cube (3D) ──Σ1──► flattened to 2D ──Σ2──► scaled in 2D ──U──► rotated in 2D
```

The result is a parallelogram: the cube has been squashed from 3D into 2D, and stretched.

### Demo 2: `A` is 3 x 2 (2D input, 3D output)

A 2D square goes through:

```
2D square ──Vᵀ──► rotated square ──Σ1──► scaled ──Σ2──► placed in 3D (dimension added) ──U──► rotated in 3D
```

The result is a tilted parallelogram **lying in a plane inside 3D space**.

### Demo code (3 x 2 case)

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 0],
              [2, 0],
              [1, 1]], dtype=float)           # 3 x 2

# corners of the unit square (centered), as columns
sq = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]]).T

U, s, Vt = np.linalg.svd(A)                    # U: 3x3, s: (2,), Vt: 2x2
Sigma = np.zeros_like(A); Sigma[:2, :2] = np.diag(s)    # 3 x 2

step1 = Vt @ sq          # rotate in input space           (2 x 5)
step2 = Sigma @ step1    # scale + add a dimension (z = 0)  (3 x 5)
step3 = U @ step2        # rotate in 3D output space        (3 x 5)

print(np.allclose(step3, A @ sq))              # True: same as applying A directly

fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(1, 4, 1); ax1.plot(*sq, "b-");        ax1.set_title("Input square")
ax2 = fig.add_subplot(1, 4, 2); ax2.plot(*step1, "r-");     ax2.set_title("after Vᵀ (rotate)")
ax3 = fig.add_subplot(1, 4, 3); ax3.plot(*step2[:2], "g-"); ax3.set_title("after Σ (scale; z=0)")
for a in (ax1, ax2, ax3): a.set_aspect("equal"); a.grid(True)
ax4 = fig.add_subplot(1, 4, 4, projection="3d"); ax4.plot(*step3, "m-"); ax4.set_title("after U (rotate in 3D)")
plt.show()
```

---

## 9. How to calculate SVD

### 9.1 With NumPy (what you'll actually do)

```python
import numpy as np

A = np.array([[1, 0],
              [2, 0],
              [1, 1]], dtype=float)            # 3 x 2

U, S, Vt = np.linalg.svd(A)                    # full SVD

print(U.shape, S.shape, Vt.shape)              # (3, 3) (2,) (2, 2)
print(S)                                       # ~[2.489, 0.899]  (singular values, descending)

# S comes back as a 1-D vector; rebuild the 3 x 2 rectangular diagonal Σ
Sigma = np.zeros(A.shape)
Sigma[:len(S), :len(S)] = np.diag(S)

print(np.allclose(A, U @ Sigma @ Vt))          # True
```

> `np.linalg.svd(A)` returns **`U, S, Vt`**: note it gives `Vᵀ` directly (not `V`), and `S` as a 1-D array of singular values.
> Use `np.linalg.svd(A, full_matrices=False)` for the compact ("economy") form: `U` is `m x k`, `Σ` is `k x k`, `Vᵀ` is `k x n` with `k = min(m, n)`. This is what you usually want for large data.

### 9.2 Through eigen decomposition (what the theory says)

```
1. Compute AᵀA  and  AAᵀ
2. Eigen decomposition of AᵀA  ->  V (eigenvectors), λ (eigenvalues)
3. Singular values  σ_i = √λ_i   (largest first)  ->  build Σ
4. Eigenvectors of AAᵀ  ->  U
```

Worked on `A = [[1,0],[2,0],[1,1]]`:

```
AᵀA = | 6  1 |      trace = 7,  det = 5
      | 1  1 |

λ² - 7λ + 5 = 0   ->   λ1 = (7+√29)/2 ≈ 6.193,   λ2 = (7-√29)/2 ≈ 0.807

σ1 = √6.193 ≈ 2.489,   σ2 = √0.807 ≈ 0.899

V ≈ | 0.982  -0.189 |     (columns = eigenvectors of AᵀA)
    | 0.189   0.982 |
```

Sanity checks: `σ1·σ2 = √det(AᵀA) = √5 ≈ 2.236` ✓ and `σ1² + σ2² = 7 =` sum of squares of all entries of `A` ✓.

```python
vals, V = np.linalg.eigh(A.T @ A)
order = np.argsort(vals)[::-1]
vals, V = vals[order], V[:, order]
sigma = np.sqrt(vals)                    # singular values
U_cols = (A @ V) / sigma                 # u_i = A v_i / σ_i   (first 2 columns of U)
print(sigma)                             # [2.489 0.899]
print(U_cols.round(3))
```

> **Sign caution:** computing `U` and `V` from two *separate* eigen decompositions can give mismatched signs. The safe way is to get `V` from `AᵀA` and then compute `u_i = A v_i / σ_i`, as above. Libraries handle this for you.

### 9.3 Properties worth remembering

- Singular values are **non-negative** and sorted `σ1 >= σ2 >= ... >= 0`.
- Number of non-zero singular values = **rank** of `A`.
- Bigger singular value = more "important" direction (this is why truncating small ones compresses).

---

## 10. SVD in PCA

### Two ways to compute PCA

**Way 1: eigen decomposition of the covariance matrix**

```
X (n x d)  ->  covariance matrix (d x d)  ->  eigen decomposition  ->  principal components
```

Example: 3 features (`cgpa`, `iq`, `10th marks`) give a 3 x 3 covariance matrix.
Iris: `X` is `(150, 4)`, so the covariance matrix is `(4, 4)`.

```
1) mean-center all the columns:   X_c = X - mean(X)
2) Cov(X) = X_cᵀ X_c / (n - 1)
   (iris: X_cᵀ X_c / (150 - 1))
```

**Way 2: SVD of the centered data (no covariance matrix needed)**

```
X_c = U Σ Vᵀ
```

Then:

```
X_cᵀ X_c = (UΣVᵀ)ᵀ (UΣVᵀ) = V Σᵀ Σ Vᵀ = V Y Vᵀ
```

This is an eigen decomposition of `X_cᵀ X_c` (proportional to the covariance matrix), so:

| From the SVD of `X_c` | Meaning in PCA |
|---|---|
| Columns of `V` (rows of `Vᵀ`) | **Eigenvectors of the covariance matrix = principal components** |
| `σ_i² / (n - 1)` | **Eigenvalues of the covariance matrix = variance along each PC** |
| `X_c V_k` (equivalently `U_k Σ_k`) | **Projected data** (PC scores) |

### Why SVD is preferred

- **Faster for large data** and more **numerically stable**: no need to form `XᵀX`, which squares the condition number and is `d x d`.
- Works directly on the data matrix (rectangular `n x d`).
- scikit-learn's `PCA` uses SVD under the hood.

### Code: iris, eigen decomposition vs SVD

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data                       # (150, 4)
n = X.shape[0]
Xc = X - X.mean(axis=0)                    # 1) mean-center

# --- Way 1: eigen decomposition of the covariance matrix ---
cov = Xc.T @ Xc / (n - 1)                  # (4, 4)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

# --- Way 2: SVD of the centered data ---
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
vals_svd = s**2 / (n - 1)

print(np.allclose(vals, vals_svd))                      # True: same eigenvalues
print(np.allclose(np.abs(vecs.T), np.abs(Vt)))          # True: same PCs (up to sign)

# --- Way 3: scikit-learn (uses SVD) ---
pca = PCA(n_components=2).fit(X)
print(np.allclose(np.abs(pca.components_), np.abs(Vt[:2])))        # True
print(np.allclose(pca.explained_variance_, vals_svd[:2]))          # True

# Projected data: three equivalent ways
Z1 = Xc @ Vt[:2].T
Z2 = U[:, :2] * s[:2]
Z3 = pca.transform(X)
print(np.allclose(np.abs(Z1), np.abs(Z2)), np.allclose(np.abs(Z1), np.abs(Z3)))
```

---

## 11. Summary and practice questions

**Summary**

- A non-square `m x n` matrix is a linear map from `n`-D to `m`-D.
- A **rectangular diagonal** matrix scales **and** adds or drops dimensions.
- **SVD:** `A = U Σ Vᵀ` for **any** matrix. `U` (`m x m`) and `V` (`n x n`) are orthogonal; `Σ` (`m x n`) is rectangular diagonal.
- `U` = eigenvectors of `AAᵀ`; `V` = eigenvectors of `AᵀA`; singular values = **√ eigenvalues**.
- Geometrically: **rotate** (`Vᵀ`), **scale / change dimension** (`Σ`), **rotate** (`U`).
- Truncating small singular values gives low-rank approximations (compression, denoising).
- **In PCA:** SVD of the centered data gives the principal components (`V`) and variances (`σ²/(n-1)`) without ever forming the covariance matrix.

**Practice questions**

1. What are the shapes of `U`, `Σ`, and `Vᵀ` for a 5 x 3 matrix `A`?
2. Why can't we do an eigen decomposition of a 2 x 3 matrix, and how does SVD get around this?
3. How are the singular values of `A` related to the eigenvalues of `AᵀA`?
4. A 3 x 2 matrix maps from which space to which space? What does `Σ` do to the dimension?
5. For `A = [[3, 0], [0, 2]]`, what are `U`, `Σ`, `V`, and the singular values?
6. If a 100 x 100 image has singular values `[50, 20, 0.1, 0.01, ...]`, how many components would you keep for compression and why?
7. Why is SVD preferred over eigen decomposition of the covariance matrix when computing PCA on large datasets?
8. How do you get the explained variance of each principal component from the singular values?

**Previous:** [Eigen Decomposition and Kernel PCA](./Eigen_Decomposition_and_Kernel_PCA_README.md)
