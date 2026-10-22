# Curse of Dimensionality

> As the number of features (dimensions) grows, data becomes sparse, distances lose meaning, and models need exponentially more data to work well.

**Why this matters for PCA:** the curse of dimensionality is the main motivation for dimensionality reduction techniques like PCA.

---

## Table of Contents

1. [What is it? (Basic)](#1-what-is-it-basic)
2. [Example 1: Data needed grows exponentially](#2-example-1-data-needed-grows-exponentially)
3. [Example 2: Sparsity of data](#3-example-2-sparsity-of-data)
4. [Example 3: Distances stop making sense](#4-example-3-distances-stop-making-sense)
5. [Example 4: Volume hides in the corners and the shell (Intermediate)](#5-example-4-volume-hides-in-the-corners-and-the-shell-intermediate)
6. [Impact on ML algorithms](#6-impact-on-ml-algorithms)
7. [Overfitting and the Hughes phenomenon](#7-overfitting-and-the-hughes-phenomenon)
8. [Advanced: the math behind it](#8-advanced-the-math-behind-it)
9. [How to fight the curse](#9-how-to-fight-the-curse)
10. [Summary](#10-summary)
11. [Practice questions](#11-practice-questions)

---

## 1. What is it? (Basic)

The term was coined by **Richard Bellman** (1961). It refers to problems that appear when working with **high-dimensional data** (many features) that **do not appear in low dimensions**.

**Simple analogy:**
- Finding your lost phone on a **line** (1D, a corridor): easy.
- Finding it on a **floor** (2D, a room): harder.
- Finding it in a **building** (3D): much harder.
- Now imagine 1000 dimensions. The "space" to search becomes astronomically large.

> **Dimension = number of features (columns).**
> A dataset with 500 columns is a 500-dimensional space.

---

## 2. Example 1: Data needed grows exponentially

Suppose each feature is split into **10 bins**, and we want **at least 1 data point per cell** to learn something useful.

| Features (d) | Cells to cover (10^d) |
|---|---|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 5 | 100,000 |
| 10 | 10,000,000,000 |
| 100 | 10^100 |

Adding just a few features means we need **exponentially more data** to get the same coverage. In real life we never have that much data.

---

## 3. Example 2: Sparsity of data

Say you have **1000 data points**.

| Dimension | What the data looks like |
|---|---|
| 1D | 1000 points on a line: **dense** |
| 2D | 1000 points on a square: still decent |
| 3D | 1000 points in a cube: getting thin |
| 100D | 1000 points in a huge space: each point is **alone**, far from everything else |

The same number of points gets **spread thinner and thinner** as dimensions are added. Neighbourhoods become empty, so there are no "nearby" points to learn from.

---

## 4. Example 3: Distances stop making sense

Many algorithms (KNN, K-Means, SVM with RBF) depend on **distance**. In high dimensions, the nearest and farthest points become almost the same distance away.

**Distance concentration:**

```
(dist_max - dist_min) / dist_min  -->  0   as d --> infinity
```

### Try it yourself

```python
import numpy as np

np.random.seed(42)
n_points = 1000

for d in [2, 10, 100, 1000, 10000]:
    X = np.random.rand(n_points, d)           # random points in [0,1]^d
    query = np.random.rand(d)                 # one query point
    dists = np.linalg.norm(X - query, axis=1)

    contrast = (dists.max() - dists.min()) / dists.min()
    print(f"d = {d:>5} | relative contrast = {contrast:.3f}")
```

**What you'll see:** the relative contrast shrinks steadily as `d` increases. "Nearest neighbour" stops being meaningfully nearer than anything else.

---

## 5. Example 4: Volume hides in the corners and the shell (Intermediate)

### 5a. Sphere inside a cube

Take a cube with side 2 and the sphere of radius 1 inscribed in it. What fraction of the cube does the sphere fill?

| Dimension | Sphere volume / Cube volume |
|---|---|
| 2 | 0.785 (78.5%) |
| 3 | 0.524 (52.4%) |
| 5 | 0.164 (16.4%) |
| 10 | 0.0025 (0.25%) |
| 20 | ~0.00000002 |

**Insight:** in high dimensions almost **all the volume is in the corners** of the cube, far from the centre. Data sampled uniformly will mostly sit in the corners, not near the centre.

### 5b. The "orange peel" effect

How much of a unit ball's volume lies within the **outer 5% shell** (between radius 0.95 and 1)?

```
fraction in shell = 1 - (1 - 0.05)^d
```

| Dimension | Volume in outer 5% shell |
|---|---|
| 1 | 5% |
| 10 | 40.1% |
| 50 | 92.3% |
| 100 | 99.4% |

**Insight:** in high dimensions, nearly **all points live on the surface** and almost none are "inside". Everything is an outlier-like point.

---

## 6. Impact on ML algorithms

| Algorithm | Problem in high dimensions |
|---|---|
| **KNN** | "Nearest" neighbours aren't really near; predictions become noisy |
| **K-Means / clustering** | Distances between points look the same, so clusters are poorly separated |
| **Linear / logistic regression** | More parameters than samples leads to overfitting and unstable coefficients |
| **Decision trees** | Many irrelevant features lead to spurious splits |
| **Density estimation** | Needs exponentially more samples |
| **Training cost** | Time and memory grow with the number of features |

---

## 7. Overfitting and the Hughes phenomenon

**Hughes phenomenon (peaking):** for a fixed number of training samples, model performance first **improves** as features are added, then **degrades** after an optimal point.

```
Accuracy
   |        ___
   |      /     \
   |    /         \___
   |  /                \___
   |/____________________________
            Number of features
        (optimum, then decline)
```

**Why?** More features give more chances to fit noise. With `d >> n` (features much greater than samples), a model can memorise the training data perfectly and still fail on new data.

**Rule of thumb:** keep samples per feature reasonably high (a commonly quoted guideline is at least 5-10 samples per feature, though this is only a rough heuristic).

---

## 8. Advanced: the math behind it

### 8.1 Volume of a unit ball

```
V_d = pi^(d/2) / Gamma(d/2 + 1)
```

The volume of the unit ball **peaks around d = 5** and then **goes to 0** as `d` grows, while the enclosing cube's volume (2^d) keeps growing. This is the formal reason behind Example 4a.

### 8.2 Concentration of norms (Gaussian data)

If `x ~ N(0, I_d)`, then

```
E[||x||^2] = d
```

and the norm `||x||` concentrates tightly around **sqrt(d)**, with a spread that stays roughly constant (about 1/sqrt(2)).

So a standard Gaussian in high dimensions does **not** look like a blob centred at the origin. It looks like a **thin shell of radius ~ sqrt(d)**. The origin, the "most likely" point, is never actually where samples land.

### 8.3 Orthogonality of random vectors

For two random vectors in `R^d`, the cosine of the angle between them has mean 0 and variance about `1/d`.

```
cos(theta) ~ N(0, 1/d)
```

As `d` grows, random vectors become **almost orthogonal** (~90 degrees). Cosine similarity then also loses discriminating power.

```python
import numpy as np

for d in [2, 10, 100, 1000, 10000]:
    a, b = np.random.randn(2, 5000, d)
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"d = {d:>5} | mean angle = {np.degrees(np.arccos(cos)).mean():.1f} deg | std of cos = {cos.std():.3f}")
```

### 8.4 Formal distance concentration

For i.i.d. features, Beyer et al. (1999) showed that under mild conditions:

```
lim (d -> inf)  Var(||x||) / E[||x||]^2 = 0
=> (D_max - D_min) / D_min -> 0   (in probability)
```

This is why "nearest neighbour" can become **meaningless** in high dimensions.

### 8.5 Minkowski distances (L_p)

Lower `p` (e.g., L1) retains **more contrast** in high dimensions than L2, and fractional norms (p < 1) retain even more (Aggarwal et al., 2001). Switching the distance metric is therefore one partial mitigation.

---

## 9. How to fight the curse

| Strategy | Idea |
|---|---|
| **Feature selection** | Keep only informative features (filter, wrapper, embedded methods such as Lasso) |
| **Dimensionality reduction** | Project to fewer dimensions: **PCA**, LDA, t-SNE, UMAP, autoencoders |
| **Regularization** | L1 / L2 penalties to control model complexity |
| **More data** | Expensive, and needs exponentially more, so rarely enough on its own |
| **Better distance metrics** | L1, cosine, or learned metrics |
| **Domain knowledge** | Engineer a few meaningful features instead of thousands of raw ones |

### Why dimensionality reduction works: the manifold hypothesis

Real-world high-dimensional data (images, text embeddings) usually lies close to a **much lower-dimensional manifold**. Its **intrinsic dimension** is small even though the **ambient dimension** is large.

> Example: a 28x28 image of a handwritten digit has 784 pixels, but the set of *valid digits* is governed by far fewer factors (stroke thickness, slant, shape, ...).

This is the key reason **PCA** and related techniques can compress data with little information loss. That brings us to the next topics: *What is PCA, geometric intuition, and variance*.

---

## 10. Summary

- Curse of dimensionality = problems that arise as the number of features grows.
- Data needed for good coverage grows **exponentially** with dimension.
- Data becomes **sparse**; the same number of points covers less and less space.
- **Distances concentrate**: nearest and farthest neighbours look alike.
- Volume moves to **corners and the outer shell**.
- Models **overfit** more easily (Hughes phenomenon).
- Remedy: **feature selection, regularization, and dimensionality reduction (PCA)**.

---

## 11. Practice questions

1. Why does the amount of data needed grow exponentially with dimension?
2. What happens to the ratio `(D_max - D_min) / D_min` as `d` increases, and why does this hurt KNN?
3. In 100 dimensions, roughly what percentage of a ball's volume lies in its outer 5% shell?
4. What is the Hughes phenomenon?
5. Why do random vectors in high dimensions tend to be nearly orthogonal?
6. Explain the manifold hypothesis and how it justifies PCA.

---

## References

- Bellman, R. (1961). *Adaptive Control Processes*.
- Beyer et al. (1999). *When Is "Nearest Neighbor" Meaningful?*
- Aggarwal, Hinneburg, Keim (2001). *On the Surprising Behavior of Distance Metrics in High Dimensional Space*.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, Ch. 2.
