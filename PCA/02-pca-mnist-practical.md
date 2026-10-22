# PCA in Practice: MNIST Walkthrough

> Apply PCA with scikit-learn on real data: visualize, measure explained variance, choose the number of components, and know when PCA fails.

**Prerequisite:** [PCA: Mathematics and Implementation](./PCA_Math_and_Implementation_README.md)

---

## Table of Contents

1. [Practical example on MNIST](#1-practical-example-on-mnist)
2. [PCA code demo](#2-pca-code-demo)
3. [2D visualization](#3-2d-visualization)
4. [Explained variance](#4-explained-variance)
5. [Finding the optimum number of principal components](#5-finding-the-optimum-number-of-principal-components)
6. [PCA code demo: compression and reconstruction](#6-pca-code-demo-compression-and-reconstruction)
7. [When does PCA not work?](#7-when-does-pca-not-work)
8. [Outro](#8-outro)

---

## 1. Practical example on MNIST

- **Data:** 70,000 handwritten digit images (0-9), each 28 x 28 = **784 features**, with pixel values 0-255.
- **Goal:** reduce 784 dimensions to a much smaller number while keeping the important structure.
- **Why:** faster training, less overfitting, visualization, noise removal, and compression.

---

## 2. PCA code demo

### Setup and load data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

mnist = fetch_openml("mnist_784", version=1, as_frame=False)   # may take a minute
X, y = mnist.data, mnist.target.astype(int)

print(X.shape)   # (70000, 784)
```

### Look at some images

```python
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, label in zip(axes.ravel(), X[:10], y[:10]):
    ax.imshow(img.reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.show()
```

### Preprocess

```python
X = X / 255.0     # scale pixels to [0, 1]
```

> All pixels share the same scale, so dividing by 255 is enough. Full standardization (`StandardScaler`) is usually **not** needed for pixel data and can amplify noise in near-constant border pixels.
> scikit-learn's `PCA` **centers the data automatically**, so you don't need to subtract the mean yourself.

### Fit PCA

```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print(X_2d.shape)   # (70000, 2)
```

---

## 3. 2D visualization

Plot all digits in 2D, coloured by label. Using a subset keeps the plot readable.

```python
idx = np.random.RandomState(42).choice(len(X), 5000, replace=False)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                      c=y[idx], cmap="tab10", s=8, alpha=0.7)
plt.colorbar(scatter, label="digit")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("MNIST projected onto the first 2 principal components")
plt.show()
```

**What to expect:**
- Some digits (like 0 and 1) form fairly distinct regions.
- Many digits **overlap heavily**.
- PC1 and PC2 together capture only a small fraction of the total variance (roughly 15-20%), so 2D is a very lossy view of MNIST.

This is a *linear* projection. Nonlinear methods (t-SNE, UMAP) usually separate digit clusters much better in 2D (see [section 7](#7-when-does-pca-not-work)).

### What do the components look like?

Each principal component is a 784-vector, so it can be shown as an image ("eigen-digits"):

```python
pca_20 = PCA(n_components=20).fit(X)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(pca_20.components_[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"PC{i+1}")
    ax.axis("off")
plt.show()
```

Early components are smooth, blob-like stroke patterns; later components show finer detail and noise.

---

## 4. Explained variance

Each component explains a fraction of the total variance.

| Attribute | Meaning |
|---|---|
| `pca.explained_variance_` | Eigenvalues (variance along each PC) |
| `pca.explained_variance_ratio_` | Eigenvalue / sum of all eigenvalues |
| `np.cumsum(explained_variance_ratio_)` | **Cumulative** variance retained |

```python
pca_full = PCA().fit(X)           # keep all 784 components

ratio = pca_full.explained_variance_ratio_
print("PC1:", ratio[0], "PC2:", ratio[1])
print("First 2 PCs together:", ratio[:2].sum())
```

### Plot 1: scree plot (variance per component)

```python
plt.figure(figsize=(8, 4))
plt.plot(range(1, 51), ratio[:50], marker="o", markersize=3)
plt.xlabel("Principal component"); plt.ylabel("Explained variance ratio")
plt.title("Scree plot (first 50 components)")
plt.grid(True); plt.show()
```

The curve **drops steeply and then flattens**, so the first few components matter most.

### Plot 2: cumulative explained variance

```python
cum = np.cumsum(ratio)

plt.figure(figsize=(8, 4))
plt.plot(cum)
plt.axhline(0.95, color="r", linestyle="--", label="95%")
plt.xlabel("Number of components"); plt.ylabel("Cumulative explained variance")
plt.grid(True); plt.legend(); plt.show()
```

---

## 5. Finding the optimum number of principal components

There is no single "correct" `k`; it is a trade-off between **compression** and **information retained**. Common methods:

### Method 1: variance threshold (most common)

Choose the smallest `k` that retains, say, 90%, 95%, or 99% of the variance.

```python
k_95 = np.argmax(cum >= 0.95) + 1
k_90 = np.argmax(cum >= 0.90) + 1
k_99 = np.argmax(cum >= 0.99) + 1
print(k_90, k_95, k_99)
```

On MNIST (scaled to [0, 1]), roughly **~90 components for 90%, ~150 for 95%, and ~330 for 99%** of the variance. This is a huge reduction from 784. (Run it to see exact values.)

scikit-learn shortcut: pass a float to `n_components`.

```python
pca_95 = PCA(n_components=0.95)       # keep enough PCs for 95% variance
X_95 = pca_95.fit_transform(X)
print(pca_95.n_components_, X_95.shape)
```

### Method 2: scree plot / elbow

Look for the "elbow" in the scree plot, the point after which extra components add little. This is subjective, so use it as a guide.

### Method 3: Kaiser criterion (for standardized data)

Keep components with eigenvalue > 1 (i.e., those explaining more variance than a single original standardized feature). Only meaningful when features are standardized.

### Method 4: downstream task performance (most practical)

Treat `k` as a **hyperparameter** and pick it using cross-validation on the model you actually care about.

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# subset for speed
Xs, ys = X[:10000], y[:10000]

for k in [10, 30, 50, 100, 200]:
    pipe = Pipeline([
        ("pca", PCA(n_components=k)),
        ("knn", KNeighborsClassifier()),
    ])
    score = cross_val_score(pipe, Xs, ys, cv=3).mean()
    print(f"k = {k:>3} | accuracy = {score:.4f}")
```

Accuracy typically **rises quickly, plateaus, and may even improve slightly** versus raw 784 features (noise removal), while training is much faster.

> Fit PCA **inside** the cross-validation pipeline (as above), never on the full dataset beforehand, to avoid data leakage.

### Summary

| Method | Pros | Cons |
|---|---|---|
| Variance threshold | Simple, interpretable | Threshold is arbitrary |
| Scree / elbow | Visual | Subjective |
| Kaiser (λ > 1) | Rule-based | Only for standardized data |
| CV on downstream task | Directly optimizes what you care about | More compute |

---

## 6. PCA code demo: compression and reconstruction

Reconstruct images from only `k` components and see how quality changes.

```python
def reconstruct(X, k):
    pca = PCA(n_components=k).fit(X)
    Z = pca.transform(X)
    return pca.inverse_transform(Z)       # Z @ components_ + mean

ks = [2, 10, 50, 150, 784]
sample = X[:5]

fig, axes = plt.subplots(len(ks) + 1, 5, figsize=(10, 12))

for j in range(5):
    axes[0, j].imshow(sample[j].reshape(28, 28), cmap="gray")
    axes[0, j].axis("off")
axes[0, 0].set_title("Original", loc="left")

for i, k in enumerate(ks, start=1):
    rec = reconstruct(X, k)[:5]
    for j in range(5):
        axes[i, j].imshow(rec[j].reshape(28, 28), cmap="gray")
        axes[i, j].axis("off")
    axes[i, 0].set_title(f"k = {k}", loc="left")

plt.tight_layout(); plt.show()
```

**What you'll see:**
- `k = 2`: blurry blobs, barely recognizable.
- `k = 10-50`: digit identity is clear.
- `k ~ 150`: nearly indistinguishable from the original.
- `k = 784`: perfect reconstruction (nothing dropped).

### Reconstruction error vs. k

```python
errors = []
for k in [2, 5, 10, 25, 50, 100, 200, 400]:
    p = PCA(n_components=k).fit(X[:10000])
    rec = p.inverse_transform(p.transform(X[:10000]))
    errors.append(np.mean((X[:10000] - rec) ** 2))

plt.plot([2, 5, 10, 25, 50, 100, 200, 400], errors, marker="o")
plt.xlabel("k"); plt.ylabel("Mean squared reconstruction error")
plt.grid(True); plt.show()
```

### Bonus: denoising

Add noise, project onto the top components, and reconstruct. Noise lives mostly in low-variance directions, so it gets discarded.

```python
rng = np.random.RandomState(0)
X_noisy = X[:2000] + rng.normal(0, 0.3, X[:2000].shape)

p = PCA(n_components=50).fit(X_noisy)
X_denoised = p.inverse_transform(p.transform(X_noisy))
```

---

## 7. When does PCA not work?

PCA is powerful but has clear limitations.

### 7.1 Nonlinear structure

PCA only finds **linear** projections. If the data lies on a curved manifold, PCA can't "unroll" it.

```python
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

Xc, yc = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)

linear = PCA(n_components=2).fit_transform(Xc)           # circles stay tangled
kernel = KernelPCA(n_components=2, kernel="rbf", gamma=10).fit_transform(Xc)  # separable

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(Xc[:, 0], Xc[:, 1], c=yc); ax[0].set_title("Original")
ax[1].scatter(linear[:, 0], linear[:, 1], c=yc); ax[1].set_title("PCA (fails)")
ax[2].scatter(kernel[:, 0], kernel[:, 1], c=yc); ax[2].set_title("Kernel PCA (works)")
plt.show()
```

**Alternatives:** Kernel PCA, t-SNE, UMAP, autoencoders, Isomap.

### 7.2 High variance does not mean important

PCA is **unsupervised**: it ignores labels. A feature that separates classes may have **low variance** and get dropped, while a noisy high-variance feature is kept.

**Alternative:** LDA (Linear Discriminant Analysis) or supervised feature selection when you have labels.

### 7.3 Sensitive to feature scale

Features with large numeric ranges dominate the variance. Example: salary (0-100,000) vs age (0-100).

**Fix:** standardize features first (`StandardScaler`) when features are on different scales.

### 7.4 Sensitive to outliers

Outliers inflate variance and can pull the principal axes toward them.

**Fix:** remove or clip outliers, or use Robust PCA.

### 7.5 Loss of interpretability

Each PC is a **mix of all original features**, so "PC1" has no direct real-world meaning (unlike "age" or "income").

If interpretability matters, prefer feature selection.

### 7.6 Other situations where PCA is a poor fit

| Situation | Why |
|---|---|
| Features are already uncorrelated | Nothing to compress; all PCs have similar variance |
| Categorical / binary data | Variance and covariance are not meaningful (use MCA or embeddings) |
| Very sparse data (e.g., text TF-IDF) | Centering destroys sparsity (use `TruncatedSVD` instead) |
| Need exact features for a regulatory or business reason | PCs are not original features |
| Dataset too large for memory | Use `IncrementalPCA` or randomized PCA |

### Quick decision guide

```
Is the structure roughly linear?          No  -> Kernel PCA / t-SNE / UMAP / autoencoder
Do you have labels and want separation?   Yes -> consider LDA
Features on different scales?             Yes -> standardize first
Need interpretable features?              Yes -> feature selection
Otherwise                                      -> PCA is a great first choice
```

---

## 8. Outro

**Key takeaways**

- MNIST: 784 dimensions can be compressed to ~100-150 with very little loss.
- `PCA(n_components=0.95)` automatically picks the number of components for 95% variance.
- Use the scree plot, cumulative variance, or cross-validation to choose `k`.
- 2D PCA plots are useful for a first look, but they lose a lot of information.
- PCA fails on nonlinear data, ignores labels, is scale- and outlier-sensitive, and reduces interpretability.

**Typical workflow**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scale", StandardScaler()),          # when features have different scales
    ("pca", PCA(n_components=0.95)),
    ("clf", LogisticRegression(max_iter=1000)),
])
```

### Practice questions

1. What does `explained_variance_ratio_` tell you, and how is it related to eigenvalues?
2. How would you choose `k` if your goal is classification accuracy?
3. Why should PCA be fit only on the training data?
4. Why does PCA fail on the concentric circles dataset, and what would you use instead?
5. Why might a low-variance component still be the most useful one for classification?
6. When would you use `TruncatedSVD` or `IncrementalPCA` instead of `PCA`?

---

## References

- scikit-learn docs: `sklearn.decomposition.PCA`, `KernelPCA`, `IncrementalPCA`
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, Ch. 14.5
- Jolliffe, I. T. *Principal Component Analysis* (2nd ed.)
