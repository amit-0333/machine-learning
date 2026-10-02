# K-Nearest Neighbors (KNN) — Intuition & Concepts

> Lecture outline: Intro · KNN Intuition · Breast-cancer code example · How to select K · Decision surface · Overfitting & underfitting in KNN · Limitations · Outro
>
> This README covers the **concepts**. The hands-on code lives in **README_KNN_2_Code_and_Practice.md**.

## Contents
1. [Intro](#1-intro)
2. [KNN intuition](#2-knn-intuition)
3. [The algorithm step by step](#3-the-algorithm-step-by-step)
4. [Distance metrics](#4-distance-metrics)
5. [How to select K](#5-how-to-select-k)
6. [Decision surface](#6-decision-surface)
7. [Overfitting and underfitting in KNN](#7-overfitting-and-underfitting-in-knn)
8. [Limitations of KNN](#8-limitations-of-knn)
9. [Outro / summary](#9-outro--summary)

---

## 1. Intro
KNN is one of the simplest supervised learning algorithms. It works for both **classification** and **regression**, makes **no assumption** about the shape of the data, and is a great first model to understand how "similarity" drives prediction.

Two key properties:
- **Lazy learner:** there is no training phase. It just stores the training data and does all the work at prediction time.
- **Non-parametric:** it doesn't learn a fixed set of weights; the stored data *is* the model.

## 2. KNN intuition
> "Tell me who your neighbours are, and I'll tell you who you are."

To predict the label of a new point:
1. Look at the **K closest** training points.
2. Let them **vote** (classification) or **average** their values (regression).

Example (breast cancer dataset): a new tumour whose 5 nearest known tumours are 4 benign and 1 malignant gets predicted **benign**.

## 3. The algorithm step by step
1. Choose **K** (number of neighbours).
2. Compute the distance from the query point to **every** training point.
3. Sort the distances and pick the **K smallest**.
4. **Classification:** majority vote. **Regression:** mean (or distance-weighted mean) of their targets.

## 4. Distance metrics

| Metric | Formula | Notes |
|---|---|---|
| Euclidean (default) | `√Σ(aᵢ − bᵢ)²` | straight-line distance |
| Manhattan | `Σ|aᵢ − bᵢ|` | grid-like distance; more robust to outliers |
| Minkowski | `(Σ|aᵢ − bᵢ|ᵖ)^(1/p)` | generalises both (p=1, p=2) |

**Feature scaling is essential.** Because KNN is distance-based, a feature measured in thousands (e.g. `area`) will swamp one measured in decimals (e.g. `smoothness`). Standardise features first.

## 5. How to select K
K is a **hyperparameter**: you choose it, the model doesn't learn it.

- **Small K** (e.g. 1): follows every point, including noise.
- **Large K**: averages over many points, smoothing away real structure.
- **Practical approach:** try a range of K values, measure **cross-validated accuracy**, and choose the K with the best score.
- **Odd K** for binary classification avoids tied votes.
- A common starting heuristic is K ≈ √n, but always validate.

## 6. Decision surface
The **decision surface (boundary)** is the border in feature space where the predicted class flips.

| K | Boundary looks like |
|---|---|
| K = 1 | Jagged, complex, with small "islands" around individual points |
| Moderate K | Smooth boundary that follows the real class separation |
| Very large K | Almost flat; tends toward always predicting the majority class |

## 7. Overfitting and underfitting in KNN
This is the bias–variance tradeoff again. Note that for KNN, **K behaves *opposite* to model complexity**: a *smaller* K means a *more complex* model.

| | Small K (≈1) | Large K |
|---|---|---|
| Boundary | Jagged | Very smooth |
| Bias | Low | High |
| Variance | High | Low |
| Failure mode | **Overfitting** | **Underfitting** |
| Train accuracy | ~100% (K=1 memorises) | Drops |
| Test accuracy | Lower than train | Low alongside train |

The sweet spot is a moderate K where train and test accuracy are both high and close together.

## 8. Limitations of KNN
- **Slow predictions:** every query computes distances to all training points (O(n·d)). Mitigate with KD-trees, ball trees or approximate nearest neighbours.
- **High memory use:** the whole training set must be stored.
- **Curse of dimensionality:** in many dimensions, distances become nearly equal and "nearest" loses meaning.
- **Sensitive to feature scale** and to **irrelevant features**.
- **Sensitive to class imbalance:** majority classes dominate the vote.
- **Sensitive to outliers/noise** when K is small.
- **Needs a good distance metric** for the problem.

## 9. Outro / summary
- KNN predicts using the **K most similar** training points.
- **Scale your features.**
- Choose K with **cross-validation**; small K → overfit, large K → underfit.
- Great baseline, but doesn't scale well to huge or high-dimensional data.

➡️ Next: **README_KNN_2_Code_and_Practice.md** for the breast cancer code, K selection and decision-surface plots.
