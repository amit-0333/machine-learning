# Eigenvalues and Eigenvectors

> An eigenvector of a matrix is a direction that the matrix only **stretches or flips**, never turns. The eigenvalue says by how much.

**Used in:** [PCA: Mathematics and Implementation](./PCA_Math_and_Implementation_README.md)

---

## Table of Contents

1. [What are matrices? (a transformation view)](#1-what-are-matrices-a-transformation-view)
2. [What are eigenvectors and eigenvalues?](#2-what-are-eigenvectors-and-eigenvalues)
3. [Intuition: axis of rotation](#3-intuition-axis-of-rotation)
4. [How to calculate eigenvalues and eigenvectors](#4-how-to-calculate-eigenvalues-and-eigenvectors)
5. [Worked example (step by step)](#5-worked-example-step-by-step)
6. [Properties](#6-properties)
7. [Eigenvectors in PCA](#7-eigenvectors-in-pca)
8. [Code: verify with NumPy](#8-code-verify-with-numpy)
9. [Summary and practice questions](#9-summary-and-practice-questions)

---

## 1. What are matrices? (a transformation view)

A matrix is not just a grid of numbers. Multiplying a vector by a matrix **transforms** it (stretches, rotates, shears, or flips it).

Take

```
A = | 2  0 |
    | 1  1 |
```

and apply it to a few vectors:

| Input vector `x` | Output `A x` | What happened |
|---|---|---|
| (1, 0) | (2, 1) | Direction changed |
| (0, 1) | (0, 1) | Unchanged |
| (1, 1) | (2, 2) | **Same direction, twice as long** |
| (-3, 1) | (-6, -2) | Direction changed |

Most vectors get **knocked off their original line**. A few special ones, like (0, 1) and (1, 1) above, stay on the same line. Those are the eigenvectors.

> Tip: try this in GeoGebra's linear transformation applet. Enter the matrix, drag a point around, and watch the output vector. Whenever the output lines up with the input, you've found an eigenvector.

---

## 2. What are eigenvectors and eigenvalues?

```
A v = λ v
```

| Symbol | Name | Meaning |
|---|---|---|
| `A` | matrix | The transformation |
| `v` | **eigenvector** | A (non-zero) vector whose direction doesn't change under `A` |
| `λ` (lambda) | **eigenvalue** | The scalar by which `v` is scaled |

**Reading it:** transforming `v` with `A` gives the same result as just multiplying `v` by a number `λ`.

### What different values of λ mean

| λ | Effect on the eigenvector |
|---|---|
| λ = 2 | Stretched to twice its length (e.g., (2, 0) becomes (4, 0)) |
| λ = 1 | Unchanged |
| λ = 0.5 | Shrunk to half (e.g., (6, 12) becomes (3, 6)) |
| λ = 0 | Collapsed to the zero vector |
| λ = -1 | **Flipped** to the opposite direction (still on the same line) |

### Eigenvectors and span

If `v` is an eigenvector, so is **every scalar multiple** of it. For example (1,1), (2,2), (3,3), ... all lie on the same line, called the **span** of `v`. They all share the same eigenvalue. So what really matters is the **direction** (the line), not the exact length of `v`. By convention we usually normalize eigenvectors to length 1.

For `A = [[2,0],[1,1]]` above:
- `v = (1, 1)` has `λ = 2` (since `A v = (2, 2) = 2 v`)
- `v = (0, 1)` has `λ = 1` (since `A v = (0, 1) = 1 v`)

---

## 3. Intuition: axis of rotation

Think of a 3D rotation (like spinning a globe). Every point moves **except those on the axis of rotation**. That axis stays fixed.

```
Rotation about the z-axis by angle θ:

R = | cos θ  -sin θ  0 |
    | sin θ   cos θ  0 |
    |   0       0    1 |

R (0, 0, 1) = (0, 0, 1)      ->  eigenvector with λ = 1
```

So the **axis of rotation is an eigenvector with eigenvalue 1**. In general, eigenvectors are the "special axes" of a transformation: the directions along which the transformation behaves in the simplest possible way (pure scaling).

---

## 4. How to calculate eigenvalues and eigenvectors

Start from the definition and rearrange:

```
A v = λ v
A v = λ I v              (insert the identity matrix I, so both sides are matrices times v)
A v - λ I v = 0
(A - λ I) v = 0
```

We want a **non-zero** `v` that satisfies `(A - λI) v = 0`.

If `(A - λI)` were invertible, the only solution would be `v = 0`. So for a non-zero `v` to exist, `(A - λI)` must be **non-invertible (singular)**, which means its determinant is zero:

```
det(A - λ I) = 0          <- characteristic equation
```

### Recipe

| Step | Action |
|---|---|
| 1 | Write `A - λI` |
| 2 | Compute `det(A - λI)` and set it to 0 (gives a polynomial in λ, the *characteristic polynomial*) |
| 3 | Solve for λ: these are the **eigenvalues** |
| 4 | For each λ, solve `(A - λI) v = 0`: these are the **eigenvectors** |

For an `n x n` matrix, the characteristic polynomial has degree `n`, so there are `n` eigenvalues (counting repeats, and possibly complex).

---

## 5. Worked example (step by step)

```
A = | 2  3 |
    | 0  1 |
```

### Step 1 to 3: find the eigenvalues

```
A - λI = | 2-λ   3  |
         |  0   1-λ |

det(A - λI) = (2-λ)(1-λ) - (3)(0) = 0
            = (2-λ)(1-λ) = 0
            = λ² - 3λ + 2 = 0

λ = 2   or   λ = 1
```

### Step 4a: eigenvector for λ = 2

```
(A - 2I) v = 0

| 0   3 | | x1 |   | 0 |
| 0  -1 | | x2 | = | 0 |

3 x2 = 0       ->  x2 = 0
x1 is free     ->  choose x1 = 1

v = (1, 0)
```

**Check:** `A v = [[2,3],[0,1]] (1,0) = (2, 0) = 2 · (1, 0)` ✓

Any multiple, like (2, 0) or (3, 0), also works (same span).

### Step 4b: eigenvector for λ = 1

```
(A - 1I) v = 0

| 1  3 | | x1 |   | 0 |
| 0  0 | | x2 | = | 0 |

x1 + 3 x2 = 0   ->   x1 = -3 x2
choose x2 = 1   ->   x1 = -3

v = (-3, 1)
```

**Check:** `A v = (2(-3) + 3(1), 0 + 1) = (-3, 1) = 1 · (-3, 1)` ✓

Also valid: (-6, 2), and so on.

### Result

| Eigenvalue | Eigenvector |
|---|---|
| λ = 2 | (1, 0) |
| λ = 1 | (-3, 1) |

> Note: in the second row of `A - λI` the whole row became `[0 0]`. That is expected: since `det = 0`, the rows are dependent, so there is one free variable, which is why there are infinitely many (parallel) eigenvectors per eigenvalue.

### How this scales

An `n x n` matrix gives `n` eigenvalues and `n` eigenvectors. For MNIST with 784 pixel features, the covariance matrix is 784 x 784, giving up to **784 eigenvectors**. PCA keeps only the top few.

When `λ = 0`, the matrix **collapses a dimension** (e.g., 3D to 2D, or 2D to 1D) because `A v = 0` sends that direction to the zero vector.

---

## 6. Properties

For an `n x n` (square) matrix `A` with eigenvalues `λ1, ..., λn`:

| # | Property | Check with `A = [[2,3],[0,1]]` |
|---|---|---|
| 1 | **Sum of eigenvalues = trace** (sum of diagonal elements) | 2 + 1 = 3 = 2 + 1 ✓ |
| 2 | **Product of eigenvalues = determinant** | 2 · 1 = 2 = det(A) = 2 ✓ |
| 3 | If `A` is **symmetric** (`A = Aᵀ`), eigenvectors for **distinct eigenvalues are orthogonal** | Covariance matrices are symmetric |
| 4 | Eigenvalues of the **identity matrix** are all 1 | `I v = 1 · v` |
| 5 | If `B = cA`, eigenvalues of `B` are `c λ` (eigenvectors unchanged) | Doubling `A` doubles every λ |
| 6 | Eigenvalues of a **diagonal matrix** are its diagonal entries | diag(4, 7) has λ = 4, 7 |
| 7 | `A` and `Aᵀ` have the **same eigenvalues** | |

> Correction to be aware of: eigenvalues are defined **only for square matrices**. The trace and determinant properties (1 and 2) apply to square matrices; they are not defined for non-square ones.

**Why property 1 matters for PCA:** for a covariance matrix, the trace is the total variance, so the eigenvalues add up to the total variance of the data. This is why `λ_i / sum(λ)` is the "explained variance ratio".

---

## 7. Eigenvectors in PCA

### Setup with a small example

Suppose each student has **2 features**: `CGPA` and `IQ` (so `n = 2`). We want to go from 2D to 1D. This is **feature extraction**: build a new feature (a combination of CGPA and IQ) that keeps as much information as possible.

### Step 1: build the covariance matrix

```
Σ = | Var(cgpa)       Cov(cgpa, iq) |
    | Cov(iq, cgpa)   Var(iq)       |
```

- Size is **2 x 2** (for `n` features it is `n x n`).
- It is **symmetric**: `Cov(cgpa, iq) = Cov(iq, cgpa)`.

### Step 2: take its eigenvectors

Because Σ is symmetric (property 3), its eigenvectors are **orthogonal** (perpendicular), so PCA gives perpendicular axes.

### Step 3: interpret

| Eigen-quantity | Meaning in PCA |
|---|---|
| Eigenvector | A new feature axis (principal component) |
| Eigenvalue | Variance of the data along that axis |
| Eigenvector with the **largest eigenvalue** | The direction that **captures maximum variance** |

Keep the top eigenvector(s) and project the data onto them to reduce dimensions with minimal information loss.

**Pipeline in one line:**

```
data -> covariance matrix (symmetric, n x n) -> eigenvectors (orthogonal)
     -> pick those with largest eigenvalues -> project data (max variance retained)
```

---

## 8. Code: verify with NumPy

```python
import numpy as np

A = np.array([[2, 3],
              [0, 1]])

eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)       # [2. 1.]
print("Eigenvectors (columns):\n", eigvecs)
# column 0 ~ [1, 0]            -> λ = 2
# column 1 ~ [-0.949, 0.316]   -> λ = 1  ( = (-3, 1) / sqrt(10) )

# Verify A v = λ v for each pair
for lam, v in zip(eigvals, eigvecs.T):
    print(np.allclose(A @ v, lam * v))   # True

# Properties
print(np.isclose(eigvals.sum(), np.trace(A)))          # trace = sum of eigenvalues
print(np.isclose(eigvals.prod(), np.linalg.det(A)))    # det = product of eigenvalues
```

> NumPy returns eigenvectors **normalized to length 1**, and the sign may differ from your hand calculation (`v` and `-v` are both valid).

### Symmetric matrices (like covariance): use `eigh`

```python
S = np.array([[1.667, 2.0],
              [2.0,   2.667]])

vals, vecs = np.linalg.eigh(S)          # sorted ascending, orthonormal eigenvectors
print(vals)                              # ~[0.105, 4.228]
print(np.allclose(vecs.T @ vecs, np.eye(2)))   # True -> orthogonal
```

### Visualize: eigenvectors keep their direction

```python
import matplotlib.pyplot as plt

A = np.array([[2, 0],
              [1, 1]])
vals, vecs = np.linalg.eig(A)

fig, ax = plt.subplots(figsize=(6, 6))
rng = np.random.RandomState(0)
for x in rng.uniform(-2, 2, (6, 2)):                 # ordinary vectors turn
    ax.arrow(0, 0, *x, color="lightgray", head_width=0.08)
    ax.arrow(0, 0, *(A @ x), color="gray", head_width=0.08)

for lam, v in zip(vals, vecs.T):                      # eigenvectors stay on their line
    ax.arrow(0, 0, *v, color="blue", head_width=0.08)
    ax.arrow(0, 0, *(A @ v), color="red", head_width=0.08)
    print(f"λ = {lam:.1f}, v = {v}")

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_aspect("equal"); ax.grid(True)
ax.set_title("Blue = eigenvector, Red = A·eigenvector (same line)")
plt.show()
```

---

## 9. Summary and practice questions

**Summary**

- `A v = λ v`: eigenvectors are directions a matrix only scales; eigenvalues are the scale factors.
- Find eigenvalues from `det(A - λI) = 0`, then eigenvectors from `(A - λI) v = 0`.
- Sum of eigenvalues = trace; product = determinant.
- Symmetric matrices (like covariance matrices) have **orthogonal** eigenvectors.
- In PCA, eigenvectors of the covariance matrix are the principal components, and the largest eigenvalue marks the direction of maximum variance.

**Practice questions**

1. Find the eigenvalues and eigenvectors of `[[3, 0], [0, 5]]`. What do you notice about diagonal matrices?
2. Find the eigenvalues of `[[4, 1], [2, 3]]` and verify the trace and determinant properties.
3. If `A` has eigenvalues 2 and 1, what are the eigenvalues of `3A`? Of `Aᵀ`?
4. Why must `(A - λI)` be singular for a non-zero eigenvector to exist?
5. What is the eigenvalue of a vector that gets flipped to point the opposite way?
6. Why are the principal components of a dataset always perpendicular to each other?

**Next:** [PCA: Mathematics and Implementation](./PCA_Math_and_Implementation_README.md)
