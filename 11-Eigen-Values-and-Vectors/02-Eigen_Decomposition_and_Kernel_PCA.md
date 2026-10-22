# Eigen Decomposition and Kernel PCA

> Eigen decomposition breaks a square matrix into **eigenvectors** and **eigenvalues**: `A = V Λ V⁻¹`. It is the engine behind PCA, and extending it with a *kernel* gives **Kernel PCA** for nonlinear data.

**Prerequisite:** [Eigenvalues and Eigenvectors](./Eigenvalues_and_Eigenvectors_README.md)

---

## Table of Contents

1. [Roadmap of this session](#1-roadmap-of-this-session)
2. [Some special matrices](#2-some-special-matrices)
3. [Matrix composition and decomposition](#3-matrix-composition-and-decomposition)
4. [Eigen decomposition](#4-eigen-decomposition)
5. [Eigen decomposition of a symmetric matrix](#5-eigen-decomposition-of-a-symmetric-matrix)
6. [Advantages of eigen decomposition](#6-advantages-of-eigen-decomposition)
7. [Kernel PCA](#7-kernel-pca)
8. [Summary and practice questions](#8-summary-and-practice-questions)

---

## 1. Roadmap of this session

```
                    Topics
          ┌───────────┴────────────┐
   Eigen decomposition        PCA variants
                          ┌──────┼──────────────┐
                    Kernel PCA  Incremental PCA  Randomized PCA  (+ SVD)
```

| Topic | Status in this README |
|---|---|
| Eigen decomposition | Covered in sections 2-6 |
| **Kernel PCA** | Covered in section 7 |
| Incremental PCA, Randomized PCA, SVD | Next session (see the short preview at the end of section 7) |

---

## 2. Some special matrices

Three special types of square matrices show up everywhere in eigen decomposition.

### 2.1 Diagonal matrix

All entries outside the main diagonal (top-left to bottom-right) are **zero**.

```
D = | d1  0   0  |
    | 0   d2  0  |
    | 0   0   d3 |
```

| Property | Detail |
|---|---|
| **Powers** | `Dⁿ` = raise **each diagonal entry** to the power `n` |
| **Eigenvalues** | Just the diagonal values; eigenvectors are the **standard basis vectors** (1,0,0), (0,1,0), (0,0,1) |
| **Times a vector** | Scales each component by the matching diagonal entry |
| **Product of two diagonals** | Diagonal matrix with entries multiplied element by element |

```
| 2  0 |³   | 8   0 |           | 2  0 | | x |   | 2x |
| 0  3 |  = | 0  27 |           | 0  3 | | y | = | 3y |
```

Diagonal matrices are the **easiest** matrices to work with. The whole idea of eigen decomposition is to turn a complicated matrix into a diagonal one.

### 2.2 Orthogonal matrix

A square matrix whose **columns (and rows) are orthonormal**: each has length 1 and any two are at 90° to each other.

For a 2 x 2 matrix with columns `(a, c)` and `(b, d)`:

```
Q = | a  b |        column 1 · column 2 = ab + cd = 0     (perpendicular)
    | c  d |        each column has length 1
```

| Property | Detail |
|---|---|
| **Inverse = transpose** | `Qᵀ = Q⁻¹`, i.e. `QᵀQ = I`. Very cheap to invert. |
| **Geometry** | Pure **rotation** (or a reflection): lengths and angles are preserved, **no scaling or shearing** |

```python
import numpy as np
theta = np.pi / 6
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(np.allclose(Q.T @ Q, np.eye(2)))        # True
print(np.allclose(Q.T, np.linalg.inv(Q)))     # True
```

### 2.3 Symmetric matrix

A square matrix equal to its own transpose (`A = Aᵀ`): swapping rows and columns changes nothing.

```
A = | a  c |
    | c  b |
```

| Property | Detail |
|---|---|
| **Real eigenvalues** | A real symmetric matrix never has complex eigenvalues |
| **Orthogonal eigenvectors** | Eigenvectors of different eigenvalues are perpendicular (and can be chosen orthonormal) |

### How they connect

```
symmetric matrix  <──────>  orthogonal eigenvectors
```

A matrix is symmetric **exactly when** it has a full set of orthogonal eigenvectors (with real eigenvalues). This is why the **covariance matrix**, which is always symmetric, has perpendicular principal components.

---

## 3. Matrix composition and decomposition

### Composition: combining matrices

Multiplying matrices **composes** their transformations into one:

```
D = A B C
```

`D` does in a single step what `C`, then `B`, then `A` do one after another (the rightmost matrix acts first on a vector). A complex transformation `D` is built from simpler ones.

### Decomposition: the reverse

Going backwards, **decomposition** breaks a complex matrix into simpler factors:

```
Composition:     A · B · C = D
Decomposition:   D = A · B · C      (find A, B, C given D)
```

### Types of matrix decomposition

```
                Matrix Decomposition
     ┌──────────┬──────────┬──────────┬──────────┐
  LU decomp   QR decomp  Eigen decomp   SVD
                             └──── both widely used in ML ────┘
```

| Decomposition | Form | Typical use |
|---|---|---|
| **LU** | `A = L U` (lower x upper triangular) | Solving linear systems |
| **QR** | `A = Q R` (orthogonal x upper triangular) | Least squares, eigenvalue algorithms |
| **Eigen** | `A = V Λ V⁻¹` | PCA, dynamical systems, powers of a matrix |
| **SVD** | `A = U Σ Vᵀ` | PCA (in practice), recommender systems, compression |

---

## 4. Eigen decomposition

### Statement

```
A = V Λ V⁻¹
```

| Symbol | Meaning |
|---|---|
| `V` | Matrix whose **columns are the eigenvectors** of `A` |
| `Λ` (Lambda) | **Diagonal** matrix of the eigenvalues of `A` (same order as the columns of `V`) |
| `V⁻¹` | Inverse of `V` |

### Derivation (2 x 2)

Let `A` have eigenvectors `v1 = (x1, y1)` and `v2 = (x2, y2)` with eigenvalues `λ1`, `λ2`:

```
A v1 = λ1 v1
A v2 = λ2 v2
```

Stack the eigenvectors as columns of `V` and multiply `A` by `V`:

```
A V = | a  b | | x1  x2 |   | a x1 + b y1    a x2 + b y2 |
      | c  d | | y1  y2 | = | c x1 + d y1    c x2 + d y2 |
```

Since `a x1 + b y1 = λ1 x1`, and so on for every entry (that is just `A v = λ v` written out):

```
A V = | λ1 x1   λ2 x2 |   | x1  x2 | | λ1  0  |
      | λ1 y1   λ2 y2 | = | y1  y2 | | 0   λ2 |  =  V Λ
```

So:

```
A V = V Λ        ──►        A = V Λ V⁻¹
```

### Assumptions

| Requirement | Why |
|---|---|
| **Square matrix** | Eigenvalues are only defined for square matrices |
| **Diagonalizable** | An `n x n` matrix needs **n linearly independent eigenvectors** so that `V` is invertible |

(Not every square matrix qualifies. Symmetric matrices always do.)

### Worked example

Using `A = [[2, 3], [0, 1]]` from the previous README, with eigenpairs `λ = 2, v = (1, 0)` and `λ = 1, v = (-3, 1)`:

```
V = | 1  -3 |      Λ = | 2  0 |      V⁻¹ = | 1  3 |
    | 0   1 |          | 0  1 |            | 0  1 |

V Λ V⁻¹ = | 2  -3 | | 1  3 |   | 2  3 |
          | 0   1 | | 0  1 | = | 0  1 |  = A  ✓
```

```python
import numpy as np

A = np.array([[2, 3],
              [0, 1]], dtype=float)

vals, V = np.linalg.eig(A)
Lam = np.diag(vals)

A_rebuilt = V @ Lam @ np.linalg.inv(V)
print(np.allclose(A, A_rebuilt))      # True
```

---

## 5. Eigen decomposition of a symmetric matrix

For a symmetric matrix, the decomposition becomes simpler and nicer:

```
A = V Λ Vᵀ          (since V is orthogonal, V⁻¹ = Vᵀ)
```

This special case is called the **spectral decomposition**.

| Factor | Type | Nature |
|---|---|---|
| `V` | **Orthogonal** | Columns are orthonormal eigenvectors |
| `Λ` | **Diagonal** | Eigenvalues |

(Diagonal, orthogonal, and symmetric all appear together here.)

### Geometric meaning: rotate, scale, rotate back

`A` is a linear transformation. Reading `A x = V Λ Vᵀ x` right to left:

```
Vᵀ  →  rotate the axes so eigenvectors line up with x- and y-axes
Λ   →  scale along each axis by its eigenvalue
V   →  rotate back
```

So **any symmetric (linear) transformation = rotate, then scale, then rotate back.**

### Example

```
S = | 2  1 |        eigenvalues:  λ1 = 3,  λ2 = 1
    | 1  2 |        eigenvectors: v1 = (1, 1)/√2,  v2 = (1, -1)/√2

S = V Λ Vᵀ = (1/√2)| 1   1 | | 3  0 | (1/√2)| 1   1 |
                   | 1  -1 | | 0  1 |       | 1  -1 |
```

```python
S = np.array([[2, 1],
              [1, 2]], dtype=float)

vals, V = np.linalg.eigh(S)            # use eigh for symmetric matrices
print(vals)                              # [1. 3.]  (ascending)
print(np.allclose(V @ np.diag(vals) @ V.T, S))   # True
print(np.allclose(V.T @ V, np.eye(2)))           # True (orthogonal)
```

### Why this matters for PCA

The **covariance matrix is symmetric**, so:

```
Σ = V Λ Vᵀ
```

- columns of `V` = principal components (orthogonal directions)
- diagonal of `Λ` = variance along each principal component

That is PCA, expressed as an eigen decomposition.

---

## 6. Advantages of eigen decomposition

**Where it is used:** machine learning (PCA), physics, signal theory, quantum mechanics.

### Fast matrix powers

Computing `A¹⁰⁰⁰` by multiplying `A` by itself 999 times is expensive. With eigen decomposition the middle terms cancel (`V⁻¹V = I`):

```
A² = (V Λ V⁻¹)(V Λ V⁻¹) = V Λ² V⁻¹
A¹⁰⁰ = V Λ¹⁰⁰ V⁻¹
```

`Λ¹⁰⁰` is **diagonal**, so it just needs each eigenvalue raised to the 100th power. Total cost: one diagonal power plus two matrix multiplications, regardless of how large the exponent is.

### Example

For `A = [[2, 3], [0, 1]]`, `Λ¹⁰ = diag(2¹⁰, 1¹⁰) = diag(1024, 1)`:

```
A¹⁰ = V Λ¹⁰ V⁻¹ = | 1  -3 | | 1024  0 | | 1  3 |   | 1024  3069 |
                  | 0   1 | |  0    1 | | 0  1 | = |  0      1  |
```

```python
def matrix_power_eig(A, k):
    vals, V = np.linalg.eig(A)
    return V @ np.diag(vals ** k) @ np.linalg.inv(V)

A = np.array([[2, 3], [0, 1]], dtype=float)
print(matrix_power_eig(A, 10))
print(np.linalg.matrix_power(A, 10))     # same result
```

### Other benefits

| Benefit | Explanation |
|---|---|
| **Simplifies transformations** | Shows the transformation as pure scaling along eigen-directions |
| **Easy inverse** | `A⁻¹ = V Λ⁻¹ V⁻¹` (just invert the diagonal entries) |
| **Dimensionality reduction** | Keep only eigenvectors with large eigenvalues (PCA) |
| **Reveals structure** | Eigenvalues show which directions matter most |

---

## 7. Kernel PCA

### 7.1 The problem: PCA is linear

PCA assumes the principal components are **linear combinations of the original features**. It can't handle complex (curved or polynomial) relationships between features.

**Example: concentric circles**

```
Original data (2D):                After PCA to 1D:
    ●  ●  ●  ●                     ●●●●●○○○●●●●●
  ●   ○  ○  ○   ●                  (the two classes overlap on the line)
 ●   ○  ○ ○ ○    ●
  ●   ○  ○  ○   ●
    ●  ●  ●  ●
```

No straight line (PC1 or PC2) can separate the inner circle from the outer ring. Projecting onto PC1 piles both classes onto the same stretch of the line. **PCA fails.**

### 7.2 The idea: lift to a higher dimension

Use a **kernel** (a mathematical function) to map the data into a higher-dimensional space (e.g. 2D → 3D) where the classes become **linearly separable**, then apply PCA there.

```
2D: circles inside circles     ──kernel──►    3D: inner circle lifted up,
(not linearly separable)                       outer ring stays low
                                               (a flat plane can separate them)
```

This is the same **kernel trick** used by SVMs (Kernel SVM): never compute the high-dimensional coordinates explicitly, only use similarities between points.

**RBF (Gaussian) kernel**, as in the notes (`y = e^(-x²)`):

```
k(xᵢ, xⱼ) = exp( -γ · ‖xᵢ - xⱼ‖² )
```

- Close points: distance ≈ 0, so similarity ≈ 1
- Far points: large distance, so similarity ≈ 0
- `γ` (gamma) controls how fast similarity decays. **High γ** = only very near neighbours count as similar.

### 7.3 Algorithm (with `X` of shape 400 x 2)

```
Input: X (400 samples, 2 features)

1. Compute the pairwise squared Euclidean distance between every pair of rows
        row1 vs row1, row1 vs row2, ..., row400 vs row400
   ->  distance² matrix, shape (400 x 400)

2. Apply the RBF kernel:   K[i, j] = exp( -γ · distance²[i, j] )
   ->  similarity matrix K, shape (400 x 400)
       (entry (i, j) = how similar sample i is to sample j)

3. Center K  (equivalent to mean-centering the data in the lifted space)

4. K is symmetric (400 x 400)  ->  eigen decomposition
   ->  400 eigenvalues and 400 eigenvectors
       (the "dimension" is now 400, one per sample)

5. Keep the top-k eigenvectors (k = 2 or 1)
   ->  these are the new coordinates: 400 x 2  or  400 x 1
```

**Key difference from normal PCA:**

| | Regular PCA | Kernel PCA |
|---|---|---|
| Matrix decomposed | Covariance matrix (`d x d`, features x features) | Kernel matrix (`n x n`, samples x samples) |
| For 400 samples, 2 features | 2 x 2 | **400 x 400** |
| Can capture nonlinear structure | No | **Yes** |

### 7.4 Code demo

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

# 400 points, 2 features, 2 classes (concentric circles)
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
print(X.shape)                                        # (400, 2)

# Regular PCA -> 1D
X_pca = PCA(n_components=1).fit_transform(X)

# Kernel PCA with RBF kernel -> 2D and 1D
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10)
X_kpca2 = kpca.fit_transform(X)
X_kpca1 = X_kpca2[:, :1]

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=y);              ax[0].set_title("Original data")
ax[1].scatter(X_pca[:, 0], np.zeros_like(X_pca), c=y, s=10)
ax[1].set_title("PCA in 1D (classes overlap)")
ax[2].scatter(X_kpca2[:, 0], X_kpca2[:, 1], c=y);  ax[2].set_title("Kernel PCA in 2D")
ax[3].scatter(X_kpca1[:, 0], np.zeros_like(X_kpca1), c=y, s=10)
ax[3].set_title("Kernel PCA in 1D (classes separate)")
plt.show()
```

**What you'll see:**
- **PCA in 1D:** both classes mixed along the line.
- **Kernel PCA in 2D / 1D:** the two classes pull apart, and on a single axis they sit at separate ends (the inner circle clusters together).

> `gamma` is a hyperparameter. Too small and the kernel is nearly linear (like PCA); too large and every point only resembles itself. Tune it by looking at the plot or with cross-validation on a downstream task.

### 7.5 Kernel PCA from scratch (matches the steps above)

```python
from sklearn.metrics.pairwise import euclidean_distances

def kernel_pca_rbf(X, gamma, k):
    n = X.shape[0]

    D2 = euclidean_distances(X, squared=True)       # (n x n) squared distances
    K = np.exp(-gamma * D2)                          # (n x n) similarity matrix

    one = np.ones((n, n)) / n                        # center the kernel matrix
    Kc = K - one @ K - K @ one + one @ K @ one

    vals, vecs = np.linalg.eigh(Kc)                  # symmetric -> eigh
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    return vecs[:, :k] * np.sqrt(vals[:k])           # (n x k) new coordinates

Z = kernel_pca_rbf(X, gamma=10, k=2)
plt.scatter(Z[:, 0], Z[:, 1], c=y); plt.title("Kernel PCA from scratch"); plt.show()
# matches sklearn's KernelPCA output up to the sign of each component
```

### 7.6 Pros and cons

| Pros | Cons |
|---|---|
| Handles nonlinear structure (circles, spirals, manifolds) | Kernel matrix is `n x n`: memory grows as `n²`, eigen decomposition about `n³` (slow for large datasets) |
| Same kernel trick as Kernel SVM | Need to choose the kernel and tune `γ` |
| Reduces to regular PCA with a linear kernel | Components are less interpretable (no direct feature weights) |
| | No exact inverse transform (reconstruction is only approximate) |

### 7.7 Preview: other PCA variants (next session)

| Variant | Problem it addresses |
|---|---|
| **Incremental PCA** | Dataset too big for memory; process in mini-batches |
| **Randomized PCA** | Faster approximate PCA when you only need a few components |
| **SVD** | The numerically stable way PCA is actually computed in libraries |

---

## 8. Summary and practice questions

**Summary**

- **Diagonal** matrices are simple (powers and products are element by element); **orthogonal** matrices are pure rotations with `Qᵀ = Q⁻¹`; **symmetric** matrices have real eigenvalues and orthogonal eigenvectors.
- **Decomposition** breaks a matrix into simpler factors (LU, QR, Eigen, SVD).
- **Eigen decomposition:** `A = V Λ V⁻¹`, valid for square, diagonalizable matrices.
- For **symmetric** matrices: `A = V Λ Vᵀ`, interpreted as rotate, scale, rotate back. The covariance matrix is symmetric, which gives PCA.
- Eigen decomposition makes **matrix powers** cheap: `Aᵏ = V Λᵏ V⁻¹`.
- PCA is **linear** and fails on data like concentric circles. **Kernel PCA** applies a kernel (e.g. RBF) to build an `n x n` similarity matrix, eigen-decomposes it, and keeps the top components.

**Practice questions**

1. Why is `Qᵀ = Q⁻¹` for an orthogonal matrix, and why is that computationally useful?
2. For which matrices does `A = V Λ V⁻¹` exist? What extra property do symmetric matrices have?
3. Compute `A⁵` for `A = [[2, 3], [0, 1]]` using eigen decomposition.
4. Why does the symmetry of the covariance matrix guarantee perpendicular principal components?
5. Why does regular PCA fail on concentric circles, and how does Kernel PCA fix it?
6. For 400 samples with 2 features, what is the size of the matrix decomposed in (a) PCA and (b) Kernel PCA?
7. What does the RBF parameter `γ` control, and what happens if it is too large or too small?

**Previous:** [Eigenvalues and Eigenvectors](./Eigenvalues_and_Eigenvectors_README.md) | **Next:** [PCA: Mathematics and Implementation](./PCA_Math_and_Implementation_README.md)
