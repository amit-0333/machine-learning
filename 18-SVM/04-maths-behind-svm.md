# Kernel SVM: From the Dual Formulation to Custom Kernels

A self-contained set of notes on **Support Vector Machines (SVM)** and the **kernel trick**. It starts from the dual formulation and builds up, step by step, to polynomial kernels, RBF kernels, the role of gamma, the connection to KNN, and how to write your own kernels.

> **Who is this for?** Anyone who knows basic ML (linear classifiers, dot products, overfitting) and wants to *really understand* why kernels work, not just call `SVC(kernel="rbf")`.

---

## Table of Contents

1. [Recap](#1-recap)
2. [SVM Dual Formulation](#2-svm-dual-formulation)
3. [The Similarity Perspective](#3-the-similarity-perspective)
4. [Kernel SVM](#4-kernel-svm)
5. [Polynomial Kernel](#5-polynomial-kernel)
6. [The Trick](#6-the-trick)
7. [What About the Other Polynomial Terms?](#7-what-about-the-other-polynomial-terms)
8. [RBF Kernel](#8-rbf-kernel)
9. [Local Decision Boundary](#9-local-decision-boundary)
10. [Effect of Gamma](#10-effect-of-gamma)
11. [Relationship Between RBF and Polynomial Kernel](#11-relationship-between-rbf-and-polynomial-kernel)
12. [Similarity with KNN](#12-similarity-with-knn)
13. [Custom Kernels](#13-custom-kernels)
14. [Cheat Sheet](#14-cheat-sheet)
15. [Quick Code Reference](#15-quick-code-reference)

---

## 1. Recap

A **linear SVM** finds the hyperplane that separates two classes with the **largest margin**.

- Hyperplane: $w^\top x + b = 0$
- Labels: $y_i \in \{-1, +1\}$
- Margin width: $\dfrac{2}{\lVert w \rVert}$, so maximizing the margin means minimizing $\lVert w \rVert$.

**Primal problem (soft margin):**

$$
\min_{w,b,\xi} \; \frac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i
\quad \text{s.t.} \quad y_i (w^\top x_i + b) \ge 1 - \xi_i, \;\; \xi_i \ge 0
$$

- $\xi_i$ are **slack variables** that allow some points to violate the margin.
- $C$ controls the trade-off: large $C$ means few violations allowed (risk of overfitting); small $C$ means a wider margin with more violations tolerated.

**Limitation:** this only works when the data is (almost) linearly separable. Kernels fix that, and the **dual formulation** is what makes kernels possible.

---

## 2. SVM Dual Formulation

Using Lagrange multipliers $\alpha_i$, the primal can be rewritten as the **dual problem**:

$$
\max_{\alpha} \; \sum_i \alpha_i \;-\; \frac{1}{2} \sum_{i}\sum_{j} \alpha_i \alpha_j \, y_i y_j \, (x_i^\top x_j)
$$

$$
\text{s.t.} \quad 0 \le \alpha_i \le C, \qquad \sum_i \alpha_i y_i = 0
$$

Key facts:

| Fact | Meaning |
|------|---------|
| $w = \sum_i \alpha_i y_i x_i$ | The weight vector is a combination of training points |
| $\alpha_i > 0$ only for **support vectors** | Points far from the boundary have $\alpha_i = 0$ and don't matter |
| Data appears **only as dot products** $x_i^\top x_j$ | **This is the door that kernels walk through** |

**Prediction for a new point $x$:**

$$
f(x) = \text{sign}\Big( \sum_{i \in SV} \alpha_i y_i \, (x_i^\top x) + b \Big)
$$

---

## 3. The Similarity Perspective

A dot product $x_i^\top x$ measures **how similar** two vectors are (large when they point in the same direction).

So the prediction formula can be read as a **weighted vote**:

> "Compare the new point to every support vector. Each support vector votes for its class ($y_i$), with strength $\alpha_i$ times how similar it is to the new point."

This is the key mindset shift. SVM is not really about hyperplanes. It's about **similarity to a few important training points**.

If we can define a better notion of similarity than the plain dot product, we get a more powerful classifier.

---

## 4. Kernel SVM

Replace the dot product with a **kernel function**:

$$
K(x_i, x_j) = \varphi(x_i)^\top \varphi(x_j)
$$

where $\varphi$ maps the data into a (possibly very high-dimensional) feature space.

**Dual problem becomes:**

$$
\max_{\alpha} \; \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \, K(x_i, x_j)
$$

**Prediction becomes:**

$$
f(x) = \text{sign}\Big( \sum_{i \in SV} \alpha_i y_i \, K(x_i, x) + b \Big)
$$

**Why this matters:** data that is not linearly separable in the original space (e.g., one class inside a circle of the other) **can** become linearly separable after mapping with $\varphi$. A linear boundary in feature space is a **curved boundary** in the original space.

---

## 5. Polynomial Kernel

$$
K(x, z) = (x^\top z + c)^d
$$

- $d$ = degree of the polynomial
- $c \ge 0$ = constant term

### Worked example (2D input, $d = 2$, $c = 1$)

Let $x = (x_1, x_2)$ and $z = (z_1, z_2)$.

$$
(x^\top z + 1)^2 = (x_1 z_1 + x_2 z_2 + 1)^2
$$

Expanding:

$$
= x_1^2 z_1^2 + x_2^2 z_2^2 + 2x_1x_2 z_1 z_2 + 2x_1 z_1 + 2x_2 z_2 + 1
$$

This equals $\varphi(x)^\top \varphi(z)$ with the explicit feature map:

$$
\varphi(x) = \big[\, x_1^2,\; x_2^2,\; \sqrt{2}\,x_1 x_2,\; \sqrt{2}\,x_1,\; \sqrt{2}\,x_2,\; 1 \,\big]
$$

So a **2D** input is implicitly lifted to a **6D** feature space.

---

## 6. The Trick

> **The kernel trick:** compute $K(x, z)$ directly in the original space, **without ever building** $\varphi(x)$.

For the example above:

- **Explicit route:** build 6D vectors, then take a dot product.
- **Kernel route:** compute $(x^\top z + 1)^2$ with one 2D dot product, one addition, one square.

The savings grow dramatically:

| Input dim $n$ | Degree $d$ | Explicit feature space size | Kernel cost |
|---|---|---|---|
| 2 | 2 | 6 | $O(n)$ |
| 100 | 3 | ~176,000 | $O(n)$ |
| 1000 | 5 | ~10¹³ | $O(n)$ |

(The explicit size is roughly $\binom{n+d}{d}$.)

This is why the dual formulation matters: since the algorithm only ever needs dot products, we can **swap in $K$ and never touch the huge feature space**.

---

## 7. What About the Other Polynomial Terms?

The expansion in Section 5 contains terms of *different* degrees (degree 2, degree 1, and a constant). Where do they come from?

- **The constant $c$ is what generates the lower-order terms.**
  - $c = 0$ gives the **homogeneous** polynomial kernel $(x^\top z)^d$, which contains **only degree-$d$ terms**.
  - $c > 0$ gives the **inhomogeneous** kernel, which contains **all terms from degree 0 up to $d$**.
- Increasing $c$ gives more weight to the lower-order terms relative to higher-order ones.
- The binomial coefficients (the $\sqrt{2}$ factors above) automatically weight each term. You never choose them manually.

A common generalized form (used by scikit-learn):

$$
K(x, z) = (\gamma \, x^\top z + r)^d
$$

with `gamma`, `coef0` ($r$), and `degree` ($d$) as hyperparameters.

---

## 8. RBF Kernel

The **Radial Basis Function** (Gaussian) kernel:

$$
K(x, z) = \exp\!\big(-\gamma \, \lVert x - z \rVert^2\big)
$$

Properties:

- Output is always in $(0, 1]$.
- $K(x, x) = 1$ (a point is perfectly similar to itself).
- $K(x, z) \to 0$ as the points move far apart.
- It depends only on the **distance** between points, not their direction.

So RBF is a very literal **similarity score**: close means near 1, far means near 0.

The feature space $\varphi$ for RBF is **infinite-dimensional** (see Section 11). The kernel trick is the only practical way to use it.

---

## 9. Local Decision Boundary

With an RBF kernel, the decision function is:

$$
f(x) = \sum_{i \in SV} \alpha_i y_i \exp\!\big(-\gamma \lVert x_i - x \rVert^2\big) + b
$$

Each support vector contributes a **Gaussian "bump"**:

- positive-class support vectors push the score **up** around themselves
- negative-class support vectors push it **down** around themselves

The decision boundary is where these bumps cancel out ($f(x) = 0$).

**Consequence:** the boundary is **local**. A point's classification is determined mainly by the *nearby* support vectors; faraway ones contribute almost nothing. This lets RBF-SVM draw very flexible, non-linear boundaries (islands, rings, wiggly curves).

Contrast with a polynomial kernel, where every point influences the global shape of the boundary.

---

## 10. Effect of Gamma

$\gamma$ controls **how far each support vector's influence reaches**. It is related to the Gaussian width by $\gamma = \dfrac{1}{2\sigma^2}$.

| | **Small $\gamma$** | **Large $\gamma$** |
|---|---|---|
| Bump width | Wide | Narrow |
| Influence of one point | Far reaching | Very local |
| Boundary | Smooth, simple (approaches linear) | Wiggly, hugs individual points |
| Risk | **Underfitting** (high bias) | **Overfitting** (high variance) |

Extreme cases:

- $\gamma \to 0$: every point looks similar to every other, so the model behaves almost like a linear classifier.
- $\gamma \to \infty$: each point is only similar to itself, so the model memorizes the training set.

**Tuning tip:** $\gamma$ and $C$ interact. Always tune them **together** with cross-validation (grid search over log-spaced values like $10^{-3}, \dots, 10^{3}$), and **scale your features** first.

---

## 11. Relationship Between RBF and Polynomial Kernel

Expand the RBF kernel:

$$
\exp(-\gamma\lVert x - z\rVert^2) = \exp(-\gamma\lVert x\rVert^2)\,\exp(-\gamma\lVert z\rVert^2)\,\exp(2\gamma\, x^\top z)
$$

Now use the Taylor series $e^{u} = \sum_{k=0}^{\infty} \dfrac{u^k}{k!}$ on the last factor:

$$
\exp(2\gamma\, x^\top z) = \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!}\,(x^\top z)^k
$$

Each term $(x^\top z)^k$ is a **homogeneous polynomial kernel of degree $k$**.

> **The RBF kernel is a weighted sum of polynomial kernels of *every* degree (0, 1, 2, 3, ...), with weights that shrink as $1/k!$.**

That is why:

- RBF has an **infinite-dimensional** feature space.
- RBF can approximate very complex functions: it contains all polynomial degrees at once.
- Higher-degree terms are down-weighted, so the model still prefers smooth solutions.

---

## 12. Similarity with KNN

An RBF-SVM looks a lot like a **K-Nearest Neighbors** classifier:

| | **KNN** | **RBF-SVM** |
|---|---|---|
| Idea | Look at nearby training points and vote | Look at nearby **support vectors** and vote |
| Who votes | The $k$ nearest points (hard cutoff) | All support vectors, weighted by $e^{-\gamma d^2}$ (soft cutoff) |
| Vote weight | Equal (or by distance) | Learned weight $\alpha_i$ times similarity |
| Memory at prediction time | **All** training data | Only support vectors (usually a small subset) |
| Hyperparameter for "locality" | $k$ | $\gamma$ |

Intuition: **RBF-SVM is a smarter, sparser, learned version of weighted KNN.** The SVM *learns* which points matter (support vectors) and how much each should count ($\alpha_i$), while KNN keeps everything and treats all neighbors almost equally.

Large $\gamma$ behaves like small $k$ (very local, jagged). Small $\gamma$ behaves like large $k$ (smooth, global).

---

## 13. Custom Kernels

You are not limited to the built-in kernels. You can design a kernel for your own data type: strings, graphs, histograms, time series, and so on.

### What makes a valid kernel? (Mercer's condition)

A function $K(x, z)$ is a valid kernel if it is **symmetric** and its **Gram matrix is positive semi-definite** for any set of points:

$$
G_{ij} = K(x_i, x_j), \qquad G = G^\top, \qquad v^\top G\, v \ge 0 \;\; \forall v
$$

This guarantees that $K$ really is a dot product in *some* feature space.

### Building new kernels from old ones

If $K_1, K_2$ are valid kernels and $a > 0$, then these are valid too:

- $a \cdot K_1$
- $K_1 + K_2$
- $K_1 \cdot K_2$
- $\exp(K_1)$

### Using a custom kernel in scikit-learn

```python
import numpy as np
from sklearn.svm import SVC

# Option 1: pass a function that returns the Gram matrix
def my_kernel(X, Y):
    return (X @ Y.T + 1) ** 2        # same as polynomial d=2, c=1

clf = SVC(kernel=my_kernel).fit(X_train, y_train)

# Option 2: precompute the Gram matrix yourself
K_train = my_kernel(X_train, X_train)
clf = SVC(kernel="precomputed").fit(K_train, y_train)

K_test = my_kernel(X_test, X_train)   # shape: (n_test, n_train)
preds = clf.predict(K_test)
```

> If your function is **not** positive semi-definite, the optimization may fail to converge or give meaningless results.

---

## 14. Cheat Sheet

| Kernel | Formula | Hyperparameters | Boundary type | Use when |
|---|---|---|---|---|
| Linear | $x^\top z$ | $C$ | Straight hyperplane | High-dim / text data, many features |
| Polynomial | $(\gamma x^\top z + r)^d$ | $d, \gamma, r, C$ | Global curves | Known feature interactions |
| RBF | $\exp(-\gamma\lVert x-z\rVert^2)$ | $\gamma, C$ | Local, flexible | **Good default** for non-linear data |
| Custom | Any valid PSD kernel | Depends | Depends | Structured data (strings, graphs, ...) |

**Remember:**

1. Dual form means data appears only as dot products.
2. Replace the dot product with $K(x, z)$ and you get a non-linear model for free.
3. Kernel = similarity; prediction = weighted vote of similarities to support vectors.
4. RBF = infinite sum of polynomial kernels; $\gamma$ sets how local it is.
5. Always scale features, and tune $C$ and $\gamma$ together.

---

## 15. Quick Code Reference

```python
from sklearn.datasets import make_circles
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X, y = make_circles(n_samples=500, factor=0.4, noise=0.1, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC())

grid = GridSearchCV(
    pipe,
    param_grid={
        "svc__kernel": ["rbf", "poly"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [0.01, 0.1, 1, 10],
    },
    cv=5,
)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
```

---

## Further Reading

- Cortes & Vapnik, *Support-Vector Networks* (1995)
- Schölkopf & Smola, *Learning with Kernels*
- scikit-learn docs: [SVM](https://scikit-learn.org/stable/modules/svm.html)
- Stanford CS229 lecture notes on SVMs and kernels
