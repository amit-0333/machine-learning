# ROC Curve and AUC

> A classifier outputs a **probability**, and we pick a **threshold** to turn it into a yes/no decision. The **ROC curve** shows how the model behaves at *every* threshold, and **AUC** summarizes that into a single number.

---

## Table of Contents

1. [Why ROC curve?](#1-why-roc-curve)
2. [Confusion matrix](#2-confusion-matrix)
3. [True Positive Rate (TPR): the benefit](#3-true-positive-rate-tpr-the-benefit)
4. [False Positive Rate (FPR): the cost](#4-false-positive-rate-fpr-the-cost)
5. [ROC curve](#5-roc-curve)
6. [Different cases: what moving the threshold does](#6-different-cases-what-moving-the-threshold-does)
7. [AUC-ROC](#7-auc-roc)
8. [Code example](#8-code-example)
9. [Summary and practice questions](#9-summary-and-practice-questions)

---

## 1. Why ROC curve?

### The setting: classification

Supervised ML has two main problem types:

| Type | Target | Example |
|---|---|---|
| **Regression** | A number | `iq, cgpa -> lpa` (package in LPA) |
| **Classification** | A category | `iq, cgpa -> placed (Yes / No)` |

For **binary classification** (placed or not placed) we can use Logistic Regression, Decision Trees, and so on.

### Workflow

```
1000 rows (iq, cgpa, placed)
        │
   ┌────┴────┐
 training    test        ->  measure performance on the test set
  (800)      (200)
```

### Predictions are really probabilities

Logistic regression does not directly output 0 or 1. It outputs a **probability between 0 and 1**:

| iq | cgpa | placed (actual) | pred_prob | prediction (threshold 0.5) |
|---|---|---|---|---|
| 7 | 70 | 1 | 0.45 | 0 |
| 8 | 80 | 0 | 0.39 | 0 |
| 9 | 90 | 1 | 0.61 | 1 |

To turn the probability into a class we need a **threshold**:

```
probability >= threshold  ->  1 (positive)
probability <  threshold  ->  0 (negative)
```

The default is **0.5**, but is 0.5 always the right choice? No.

### Why the threshold matters: two kinds of mistake

Take an **email spam classifier**. A binary classifier can make **two different mistakes**:

| | Actual | Predicted | What happens |
|---|---|---|---|
| **Mistake 1** | Spam | Not spam | A spam email lands in your inbox (annoying) |
| **Mistake 2** | Not spam | Spam | A *real* email (job offer, bank alert) goes to the spam folder (costly) |

Which mistake is worse depends on the problem:

- Spam filter: mistake 2 is usually worse, so use a **high threshold** (e.g. 0.85 or 0.9, only flag when very sure).
- Disease screening: mistake 1 is worse (missing a sick patient), so use a **low threshold**.

```
        threshold 0.5:   prob 0.75 -> spam
        threshold 0.85:  prob 0.75 -> NOT spam
```

Different thresholds give different results, so how do we evaluate a model **without committing to one threshold**? We look at performance across **all thresholds** (0.1, 0.2, ..., 0.9). That is the **ROC curve**.

---

## 2. Confusion matrix

The confusion matrix is the **report card for binary classification**: a grid comparing actual and predicted labels.

```
                    Predicted
                   1          0
               ┌──────────┬──────────┐
          1    │    TP    │    FN    │
 Actual        │  (true   │  (false  │
               │ positive)│ negative)│
               ├──────────┼──────────┤
          0    │    FP    │    TN    │
               │  (false  │  (true   │
               │ positive)│ negative)│
               └──────────┴──────────┘
```

| Cell | Meaning | Correct? |
|---|---|---|
| **TP** (True Positive) | Actual 1, predicted 1 | Yes |
| **TN** (True Negative) | Actual 0, predicted 0 | Yes |
| **FP** (False Positive) | Actual 0, predicted 1 | **No** (mistake 2) |
| **FN** (False Negative) | Actual 1, predicted 0 | **No** (mistake 1) |

The diagonal (TP and TN) holds the correct predictions; the off-diagonal (FP and FN) holds the mistakes.

### Example: 200 test samples (from the 1000-row dataset)

```
               Predicted
               1      0
Actual   1  | 100  |   5  |
         0  |  15  |  80  |
```

- Correct predictions: `100 + 80 = 180`
- Wrong predictions: `5 + 15 = 20`
- Accuracy = `180 / 200 = 90%`

Accuracy alone hides *which* kind of mistake the model makes. TPR and FPR separate them.

---

## 3. True Positive Rate (TPR): the benefit

```
TPR = TP / (TP + FN)
```

"Out of all the **actual positives**, how many did we catch?" (Also called **recall** or **sensitivity**.)

### Example: email spam classifier

`1 = spam`, `0 = not spam`. Test set: **200 emails**, of which 100 are spam and 100 are not spam.

```
               Predicted
               spam   not spam
Actual spam  |  80  |   20  |
   not spam  |  20  |   80  |

TPR = 80 / (80 + 20) = 80 / 100 = 80%
```

The model detected 80 out of 100 spam emails. **If it caught all 100, TPR would be 100%.**

### Example: churn prediction (Netflix)

`1 = customer leaves`, `0 = customer stays`. Of 100 customers who actually leave, the model detects 80:

```
TPR = 80 / (80 + 20) = 80%
```

> **TPR is the benefit.** It measures how much of what you wanted to catch you actually caught. We want it **high**.

---

## 4. False Positive Rate (FPR): the cost

```
FPR = FP / (FP + TN)
```

"Out of all the **actual negatives**, how many did we wrongly flag as positive?"

### Example: email spam classifier

Of 100 emails that are **not spam**, the model wrongly marks 20 as spam:

```
FPR = 20 / (20 + 80) = 20 / 100 = 20%
```

Those 20 real emails end up in the spam folder, which is the **cost** of the model's aggressiveness.

### Example: churn prediction

Customers who were **not** going to leave, but the model predicted "will leave", get a retention discount they didn't need. Each wasted discount is a **real cost** to the platform. The more of them there are, the higher the FPR.

> **FPR is the cost.** We want it **low**.

```
TPR  ->  Benefit  ->  want it HIGH ↑
FPR  ->  Cost     ->  want it LOW  ↓
```

### The best case

```
TPR = 100% (= 1)   ->   FN = 0
FPR = 0%   (= 0)   ->   FP = 0

               Predicted
               1      0
Actual   1  | 100  |   0  |
         0  |   0  | 100  |       <- all mistakes are zero
```

A perfect classifier has TPR = 1 and FPR = 0.

---

## 5. ROC curve

**ROC** = Receiver Operating Characteristic. It is a plot of:

```
y-axis:  TPR (benefit)
x-axis:  FPR (cost)
```

drawn for **many different thresholds**.

### How to build it

```
1. Train a model (e.g. Logistic Regression on cgpa, iq -> placement)
2. Get predicted probabilities on the test set
3. For each threshold (0.1, 0.2, 0.3, ..., 0.9):
       a. Convert probabilities to 0/1 using that threshold
       b. Build the confusion matrix
       c. Compute TPR and FPR
       d. Plot the point (FPR, TPR)
4. Connect the points -> ROC curve
```

```
TPR
 1 │ ●  ← ideal point (FPR = 0, TPR = 1)
   │   ╲ ● 0.5
   │      ● 0.3     each dot = one threshold
   │   ● 0.2
   │ ● 0.1
   └──────────────── FPR
 0                  1
```

### How to read it

- The **top-left corner (FPR = 0, TPR = 1)** is the ideal state: all benefit, no cost.
- A model whose curve **hugs the top-left** is better.
- Each point on the curve is one choice of threshold, so the ROC curve also helps you **pick a threshold** that matches your cost/benefit trade-off.

---

## 6. Different cases: what moving the threshold does

### Lowering the threshold ("relaxing" it)

Example threshold `0.8 -> 0.5`: more emails are labeled spam.

```
Lower threshold  ->  more samples predicted positive
                 ->  TP ↑  (catch more real positives)   -> TPR ↑
                 ->  FP ↑  (but also flag more negatives) -> FPR ↑
                 ->  FN ↓,  TN ↓
```

**Moves you up and to the right** on the ROC curve.

### Raising the threshold ("strict")

Example: only flag spam if `prob > 0.95`.

```
Higher threshold ->  fewer samples predicted positive
                 ->  TP ↓, FP ↓
                 ->  TPR ↓ and FPR ↓
                 ->  FN ↑,  TN ↑
```

**Moves you down and to the left** on the ROC curve.

### Summary table

| Change | TP | FP | TPR | FPR | Position on ROC |
|---|---|---|---|---|---|
| **Decrease** threshold | ↑ | ↑ | ↑ | ↑ | Toward top-right |
| **Increase** threshold | ↓ | ↓ | ↓ | ↓ | Toward bottom-left |

**TPR and FPR move together.** You can't raise TPR for free; the price is a higher FPR. The ROC curve shows how steep that trade-off is.

### Extreme thresholds

| Threshold | Prediction | Point |
|---|---|---|
| 0 | Everything is positive | (FPR = 1, TPR = 1), top-right corner |
| 1 | Everything is negative | (FPR = 0, TPR = 0), bottom-left corner |

### Different quality models

```
TPR
 1 │ ●───────────●        <- ideal point (top-left)
   │ │   ╱‾‾‾‾‾‾
   │ │ ╱     good model (curve bulges toward top-left)
   │ ╱   ╱   ╱     average model
   │╱  ╱   ╱ <-- diagonal = random guessing (TPR = FPR)
   │ ╱
   └───────────────● FPR     <- worst state (bottom-right: TPR = 0, FPR = 1)
```

| Curve | Meaning |
|---|---|
| Hugs top-left corner | Strong model |
| Along the **diagonal** | Random guessing (no skill) |
| **Below** the diagonal | Worse than random: predictions are inverted (flipping them would help) |
| Bottom-right corner | Worst state (TPR = 0, FPR = 1) |

---

## 7. AUC-ROC

**AUC** = **A**rea **U**nder the (ROC) **C**urve.

> It measures the entire two-dimensional area under the ROC curve from (0, 0) to (1, 1). It gives an **aggregate measure of performance across all possible classification thresholds**.

| AUC | Meaning |
|---|---|
| **1** | **Perfect discrimination**: classifies all positives and negatives correctly |
| **0.5** | **No discrimination**: as good as random guessing |
| **0** | **Perfectly wrong**: classifies all positives as negative and all negatives as positive (flip the predictions and you get a perfect model) |

In practice AUC falls between **0.5 (random) and 1 (perfect)**; higher is better.

### Use of AUC: comparing models

A single number lets you **compare models**, independent of any one threshold:

```
Model A: AUC = 0.91
Model B: AUC = 0.84     ->  Model A ranks positives above negatives better
```

### Probabilistic meaning

AUC equals the probability that a **randomly chosen positive** sample gets a **higher predicted score** than a **randomly chosen negative** sample. AUC = 0.9 means that in 90% of such pairs the model ranks the positive one higher.

### Things to keep in mind

- AUC doesn't tell you *which threshold to use*. For that, read the ROC curve (see the Youden's J example below) or use your actual cost of FP vs FN.
- With **heavily imbalanced** data (e.g. 1% positives), ROC-AUC can look overly optimistic; also check the **precision-recall curve**.

---

## 8. Code example

### 8.1 Hand calculation from a confusion matrix

```python
TP, FN, FP, TN = 100, 5, 15, 80       # from section 2

tpr = TP / (TP + FN)
fpr = FP / (FP + TN)
acc = (TP + TN) / (TP + TN + FP + FN)
print(f"TPR = {tpr:.3f}, FPR = {fpr:.3f}, Accuracy = {acc:.3f}")
# TPR = 0.952, FPR = 0.158, Accuracy = 0.900
```

### 8.2 Build a placement dataset and train a model

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

# synthetic data: 1000 students (iq, cgpa) -> placed (1/0)
rng = np.random.RandomState(42)
n = 1000
iq = rng.normal(100, 15, n)
cgpa = np.clip(rng.normal(7, 1, n), 4, 10)
score = 0.05 * (iq - 100) + 1.0 * (cgpa - 7) + rng.normal(0, 0.8, n)
placed = (score > 0).astype(int)

df = pd.DataFrame({"iq": iq, "cgpa": cgpa, "placed": placed})
X, y = df[["iq", "cgpa"]], df["placed"]

# 800 train / 200 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]          # P(placed = 1)
print(probs[:5].round(2))
```

### 8.3 Confusion matrix at one threshold

```python
threshold = 0.5
y_pred = (probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()   # sklearn order: TN FP FN TP
print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print("TPR =", tp / (tp + fn))
print("FPR =", fp / (fp + tn))
```

> sklearn's `confusion_matrix` has rows = actual, columns = predicted, in the order `[0, 1]`, so the layout is `[[TN, FP], [FN, TP]]`, which is **different** from the `[[TP, FN], [FP, TN]]` grid drawn in your notes. Same numbers, different arrangement.

### 8.4 ROC curve from scratch (loop over thresholds)

This is exactly the recipe from section 5.

```python
def roc_from_scratch(y_true, probs, thresholds):
    points = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        points.append((fp / (fp + tn), tp / (tp + fn)))      # (FPR, TPR)
    return np.array(points)

pts = roc_from_scratch(y_test.values, probs, np.linspace(0, 1, 101))
fpr_s, tpr_s = pts[:, 0], pts[:, 1]

order = np.argsort(fpr_s)
print("AUC (from scratch):", auc(fpr_s[order], tpr_s[order]))
```

### 8.5 With scikit-learn (the usual way)

```python
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)
print("AUC =", roc_auc)          # ~0.886 with the seed above

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
plt.scatter([0], [1], c="green", s=80, zorder=5, label="Ideal point")
plt.xlabel("False Positive Rate (cost)")
plt.ylabel("True Positive Rate (benefit)")
plt.title("ROC curve")
plt.legend(loc="lower right"); plt.grid(True)
plt.show()
```

### 8.6 Look at TPR / FPR across thresholds

```python
table = pd.DataFrame({"threshold": thresholds, "FPR": fpr, "TPR": tpr})
print(table.iloc[::10].round(3))
# as threshold goes down: FPR and TPR both go up (section 6)
```

### 8.7 Choose a threshold (Youden's J)

One simple rule picks the point that maximizes `TPR - FPR` (farthest above the diagonal):

```python
J = tpr - fpr
best = J.argmax()
print(f"best threshold = {thresholds[best]:.3f}, TPR = {tpr[best]:.3f}, FPR = {fpr[best]:.3f}")
```

If false positives are very costly (spam filter), pick a point further left (lower FPR); if false negatives are very costly (disease screening), pick a point further up (higher TPR).

### 8.8 Compare models using AUC

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
}

plt.figure(figsize=(6, 6))
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    f, t, _ = roc_curve(y_test, p)
    plt.plot(f, t, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right"); plt.grid(True)
plt.title("Comparing models with ROC-AUC")
plt.show()
# with this seed: Logistic ~0.886, Random Forest ~0.846, Tree ~0.819
```

### 8.9 AUC = 0 means "perfectly wrong"

```python
print(roc_auc_score(y_test, probs))        # ~0.886
print(roc_auc_score(y_test, 1 - probs))    # ~0.114 = 1 - 0.886 (inverted predictions)
```

> Use `predict_proba` (probabilities), **not** `predict` (hard 0/1 labels), for ROC-AUC. Hard labels give just one threshold, so one point and a poor, misleading curve.

---

## 9. Summary and practice questions

**Summary**

- Classifiers output probabilities; a **threshold** converts them to classes. The best threshold depends on the cost of each kind of mistake.
- **Confusion matrix:** TP, FN, FP, TN.
- **TPR = TP / (TP + FN)** is the **benefit** (want high). **FPR = FP / (FP + TN)** is the **cost** (want low).
- **ROC curve:** plot (FPR, TPR) for many thresholds. Top-left is ideal; the diagonal is random.
- Lower threshold: both TPR and FPR go **up**. Higher threshold: both go **down**.
- **AUC:** area under the ROC curve. 1 = perfect, 0.5 = random, 0 = perfectly wrong. It is a threshold-independent number to **compare models**.

**Practice questions**

1. In the confusion matrix `[[100, 5], [15, 80]]` (rows = actual 1 and 0; columns = predicted 1 and 0), compute accuracy, TPR, and FPR.
2. Why is accuracy alone not enough to judge a spam filter?
3. What happens to TP, FP, TPR, and FPR when you decrease the threshold from 0.8 to 0.3?
4. Why is the point (FPR = 0, TPR = 1) the ideal point?
5. What does a ROC curve that lies below the diagonal tell you? How could you fix it?
6. What does a threshold of 0 give? What about a threshold of 1? Where do they sit on the ROC curve?
7. Two models have AUC 0.92 and 0.85. Does the first one always do better at every threshold? Why or why not?
8. For a cancer-screening model, would you pick a threshold toward the top-right or the bottom-left of the ROC curve? Why?
9. Why shouldn't you pass `model.predict(X_test)` to `roc_auc_score`?
10. For heavily imbalanced data, what other curve should you look at besides ROC?
