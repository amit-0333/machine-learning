# Data Leakage in Machine Learning

> Data leakage is when the model gets to see information during training or validation that it **will not have in the real world**. The result: great scores in the notebook, disappointing results in production.

**Related:** [Cross-Validation](./Cross_Validation_README.md) | [ROC and AUC](./ROC_AUC_Curve_README.md)

---

## Table of Contents

1. [The problem](#1-the-problem)
2. [What is data leakage?](#2-what-is-data-leakage)
3. [Ways in which data leakage can occur](#3-ways-in-which-data-leakage-can-occur)
4. [How to detect data leakage](#4-how-to-detect-data-leakage)
5. [How to remove data leakage](#5-how-to-remove-data-leakage)
6. [The validation set](#6-the-validation-set)
7. [Code demos](#7-code-demos)
8. [Checklist, summary and practice questions](#8-checklist-summary-and-practice-questions)

---

## 1. The problem

```
Dataset D ──► split ──► train ──► model
                  └───► test  ──► test score = 95%   (looks great!)
                                      │
                                      ▼
                                   deploy (e.g. on a website)
                                      │
                                      ▼
                                 track real performance = 70%   (what happened?)
```

A **25-point gap** between test performance and real-world performance is a red flag. One major cause is **data leakage**: something in the experiment design let the model "cheat", so the 95% was never real.

---

## 2. What is data leakage?

> Data leakage refers to a problem where **information from outside the training dataset** is used to create the model. This extra information is something the model **wouldn't have access to when it is used for prediction** in a real-world scenario.

| Phase | What happens |
|---|---|
| **Training / validation** | The model has access to extra information, so performance estimates are **overly optimistic** |
| **Production** | The extra information is **no longer available**, so performance **drops significantly** |

This discrepancy is typically the result of **mistakes in the experiment design**, not of a bad algorithm.

---

## 3. Ways in which data leakage can occur

| # | Type |
|---|---|
| 1 | Target leakage |
| 2 | Multicollinearity with the target column |
| 3 | Duplicated data |
| 4 | Preprocessing leakage (train-test contamination and improper cross-validation) |
| 5 | Hyperparameter tuning |

### 3.1 Target leakage

> Target leakage occurs when your **predictors include data that will not be available at the time you make predictions.**

The test: for every input feature, ask **"will this value exist at the moment I need to predict?"** (the input must be available in the future too).

**Example 1: credit card fraud detection**

| value | date | website | ... | `reversed_transaction` | **`fraud`** (target) |
|---|---|---|---|---|---|
| 500 | | | | 1 | 1 |
| 400 | | | | 0 | 0 |

- `reversed_transaction` is recorded **after** a transaction has been investigated and found fraudulent.
- At the moment a new transaction arrives, we cannot know if it will be reversed.
- The model learns *"reversed means fraud"*, which gives very high training performance. In production this feature is missing or empty, so the prediction quality drops (e.g. from 95% in training to 80% in practice).

**Example 2: smartphone price prediction**

| brand | ram | ... | `rating` | **`price`** (target) |
|---|---|---|---|---|

- `rating` is partly a *consequence* of the price (and doesn't exist for a phone that hasn't been released or reviewed yet).
- It leaks information about the target, so accuracy looks great but the model is useless for pricing a new phone.

**Rule of thumb:** features must be known **at prediction time**, and must not be *caused by* or *computed from* the target.

### 3.2 Multicollinearity with the target column

A feature that is **highly correlated with, or directly derived from, the target** acts as a giveaway of the answer.

Examples:
- Predicting `price_in_USD` while including `price_in_INR` as a feature.
- Predicting `total_bill` while including `tax_amount` (computed as a fixed % of the bill).
- Predicting a churn flag while including `cancellation_date`.

A very high correlation between a feature and the target is a hint to **investigate** (it may be a legitimately strong predictor, or it may be leakage). Check where the column comes from.

### 3.3 Duplicated data

If the dataset contains duplicate (or near-duplicate) rows, after a random split the **same row can land in both train and test**.

```
1000 rows, of which 200 are duplicates
          │
   random split
     ┌────┴────┐
   train      test      <- many test rows have an exact twin in train
     │          │
     └─ model ──┘        the model has "seen" the test answers = extra info
                         -> inflated accuracy
```

The model effectively **memorizes** answers for those rows (worst for models like KNN or deep trees). **Remove duplicates before splitting.**

### 3.4 Preprocessing leakage

Preprocessing steps (handling missing values, scaling, encoding, feature selection, PCA, etc.) *learn* statistics from data. If they learn from the **whole dataset before the split**, the test data influences training.

#### a) Train-test contamination

```
1000 rows  ->  mean(Age) = 27   <- computed on ALL rows (includes the test rows!)
                  │
          split into train / test
                  │
   fill missing Age with 27 / standardize using μ, σ of all 1000 rows
```

The mean (or μ and σ for standardization) already contains information about the **test rows**, which is **extra information** the model shouldn't have. The test set no longer represents truly unseen data.

**Correct order:**

```
1000 rows ──► split first ──► train (800)        test (200)
                               │                   │
                      compute mean/μ/σ on          apply the SAME train
                      TRAIN only (e.g. 27)  ─────► statistics to test
                      fit the model                (test's own average, e.g. 26,
                                                    is never used)
```

#### b) Improper cross-validation

The same mistake inside CV: if you preprocess the whole dataset first and then run cross-validation, **each validation fold has already influenced the preprocessing** applied to its training folds.

```python
# WRONG: normalize the whole dataset, then cross-validate
X_normalized = normalize(X)
cross_val_score(model, X_normalized, y, cv=5)

# RIGHT: make preprocessing part of the model (a pipeline), so it is
#        re-fitted on the training folds of every split
pipeline = make_pipeline(normalizer, model)
cross_val_score(pipeline, X, y, cv=5)
```

A **pipeline** chains operations (preprocessing, then model) so that each step is fitted only on the data it is allowed to see.

### 3.5 Hyperparameter tuning

Suppose you tune KNN's `n_neighbors` (1, 2, 3, 4, ...) or the distance metric by checking the **test accuracy** each time and picking the best.

```
train ──► model(k=1) ──► test score
train ──► model(k=2) ──► test score      ->  pick the k with the best TEST score
train ──► model(k=3) ──► test score
```

The test set was used to *make a decision*, so information about it leaked into the model's configuration (the test set became a "north star" for your choices). The final test score is now optimistic: it's no longer measuring performance on unseen data.

**Fix:** tune on a separate **validation set** (or by cross-validation on the training data) and touch the test set only once at the very end (see [section 6](#6-the-validation-set)).

---

## 4. How to detect data leakage

| # | Check | What to look for |
|---|---|---|
| 1 | **Review your features** | Does any feature contain data not available at prediction time, or that directly or indirectly reveals the target? These are the most common sources of leakage. |
| 2 | **Unexpectedly high performance** | Most real predictive tasks are hard. Near-perfect scores are suspicious (the model may have access to information it shouldn't). |
| 3 | **Inconsistent performance between training/validation and unseen data** | Much better on train/validation data than on new data after deployment suggests leakage. |
| 4 | **Model interpretability** | Check feature importances/coefficients. If one feature dominates but makes no sense for the target (or is available only after the fact), suspect leakage. |

Quick practical checks:

```python
# correlation of each numeric feature with the target
df.corr(numeric_only=True)["target"].sort_values(ascending=False)

# feature importances from a tree model
pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# duplicates
df.duplicated().sum()
```

---

## 5. How to remove data leakage

| # | Practice | Details |
|---|---|---|
| 1 | **Understand the data and the task** | Before any processing or modelling, understand the problem, how the data was collected, what each feature means, and whether it will be available at prediction time. |
| 2 | **Careful feature selection** | Review every feature. If one contains information not available at prediction time, or directly/indirectly gives away the target, **remove or modify it**. |
| 3 | **Proper data splitting** | Split into train / validation / test **early**, before any preprocessing or feature extraction. |
| 4 | **Preprocess inside the cross-validation loop** | Do preprocessing separately on each fold so the validation fold never influences training. |
| 5 | **Use a pipeline** (the "correct way" in 4) | Chain preprocessing and model with `make_pipeline`, then cross-validate the pipeline. |
| 6 | **Avoid overlapping data** | The same individuals, or the same time periods, must not appear in both train and test. **Remove duplicates.** |

```python
# Incorrect way
X_normalized = normalize(X)                              # normalizes the whole dataset
cross_val_score(model, X_normalized, y, cv=5)            # then cross-validation

# Correct way
pipeline = make_pipeline(normalizer, model)
cross_val_score(pipeline, X, y, cv=5)
```

**Overlap also happens with grouped data:** if one customer has 10 rows, all 10 should go to the same side of the split (use `GroupKFold` / `GroupShuffleSplit`). For time-ordered data, train on the past and test on the future (`TimeSeriesSplit`).

---

## 6. The validation set

Using only **train** and **test** is risky when you also make choices (hyperparameters, model type), because those choices are made by looking at the test set.

Use **three** sets:

```
all data
 ├── train        ──► fit the model
 ├── validation   ──► tune hyperparameters / compare models   (e.g. 85%)
 └── test         ──► ONE final evaluation on unseen data     (e.g. 79%)  <- the realistic number
```

| Set | Used for | How often |
|---|---|---|
| **Train** | Learning the model's parameters | Many times |
| **Validation** | Choosing hyperparameters, comparing models | Many times (it is "burned" by this) |
| **Test** | Final, honest performance estimate | **Once** |

The validation score (85%) is a bit optimistic because we selected the model *because* it did well there; the test score (79%) is the real-world estimate. This train / validation / test pattern is standard practice in deep learning.

If data is scarce, replace the fixed validation set with **cross-validation on the training data** (e.g. `GridSearchCV`) and keep the test set for the end.

---

## 7. Code demos

All demos use scikit-learn.

### 7.1 Target leakage: a fraud-detection example

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(0)
n = 2000
amount  = rng.exponential(100, n)
hour    = rng.randint(0, 24, n)
foreign = rng.binomial(1, 0.2, n)

logit = -4 + 0.012 * amount + 1.5 * foreign + 1.0 * (hour < 5)
fraud = rng.binomial(1, 1 / (1 + np.exp(-logit)))

# reversed_transaction is only known AFTER fraud is confirmed -> leaky feature
reversed_tx = np.where(fraud == 1, rng.binomial(1, 0.97, n), rng.binomial(1, 0.02, n))

df = pd.DataFrame(dict(amount=amount, hour=hour, foreign=foreign,
                       reversed_transaction=reversed_tx, fraud=fraud))
print(df.corr()["fraud"].round(2))        # reversed_transaction ~ 0.9: suspicious!

for cols in (["amount", "hour", "foreign", "reversed_transaction"],   # with leakage
             ["amount", "hour", "foreign"]):                          # leakage removed
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cols], df.fraud, test_size=0.25, stratify=df.fraud, random_state=0)
    m = RandomForestClassifier(random_state=0, min_samples_leaf=5).fit(X_tr, y_tr)
    print(cols, "AUC =", round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 3))
# with leakage    -> AUC ~ 0.996 (too good to be true)
# without leakage -> AUC ~ 0.80  (honest estimate)
```

(AUC is used rather than accuracy because fraud is rare. See the [ROC-AUC README](./ROC_AUC_Curve_README.md).)

The leaky model looks nearly perfect on the test set, but in production `reversed_transaction` wouldn't exist yet, so it would perform far worse.

### 7.2 Duplicated data

```python
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=800, n_features=10, n_informative=3,
                           flip_y=0.25, random_state=1)

X_dup = np.vstack([X, X[:200]])                      # 200 duplicated rows
y_dup = np.concatenate([y, y[:200]])

def knn_score(X_, y_):
    X_tr, X_te, y_tr, y_te = train_test_split(X_, y_, test_size=0.25, random_state=0)
    return KNeighborsClassifier(n_neighbors=1).fit(X_tr, y_tr).score(X_te, y_te)

print("with duplicates   :", knn_score(X_dup, y_dup))   # ~0.76 (inflated)
print("without duplicates:", knn_score(X, y))           # ~0.66 (honest)

# In real data: find and drop duplicates BEFORE splitting
df = pd.DataFrame(X_dup)
print(df.duplicated().sum())                         # 200
df = df.drop_duplicates()
```

### 7.3 Preprocessing leakage: wrong vs right

**(a) Dramatic example: feature selection on random data**

The labels are pure noise, so the honest accuracy is about 50%.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(0)
X_rand = rng.randn(100, 5000)                 # 100 samples, 5000 random features
y_rand = rng.randint(0, 2, 100)               # random labels: nothing to learn!

# WRONG: select the "best" 20 features using ALL the data (incl. validation folds)
X_sel = SelectKBest(f_classif, k=20).fit_transform(X_rand, y_rand)
print("wrong:", cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_rand, cv=5).mean())
# ~0.90  <- looks like a great model, but it's learning nothing

# RIGHT: selection happens inside each training fold, via a pipeline
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
print("right:", cross_val_score(pipe, X_rand, y_rand, cv=5).mean())
# ~0.57 (close to the true 0.5)
```

**(b) Scaling: same mistake, usually smaller effect**

```python
from sklearn.preprocessing import StandardScaler

# WRONG
X_scaled = StandardScaler().fit_transform(X)            # μ, σ from ALL rows
score_wrong = cross_val_score(KNeighborsClassifier(), X_scaled, y, cv=5).mean()

# RIGHT
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
score_right = cross_val_score(pipe, X, y, cv=5).mean()
```

For plain scaling the numeric difference is often small, but the principle is the same, and for steps that use the **target** (feature selection, target encoding) or small/noisy data the difference can be huge. The pipeline costs nothing, so always use it.

**(c) Train/test version (no CV)**

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_tr)        # learn μ, σ from TRAIN only
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)            # apply the train μ, σ to test (never fit on test)
```

### 7.4 Hyperparameter tuning with a validation set

```python
Xa, ya = make_classification(n_samples=1000, n_informative=5, flip_y=0.1, random_state=2)

# 1) split off the TEST set first and don't touch it
X_tmp, X_test, y_tmp, y_test = train_test_split(Xa, ya, test_size=0.2, stratify=ya, random_state=0)

# 2) split the rest into TRAIN and VALIDATION  (60% / 20% / 20% overall)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)

# 3) choose k using the VALIDATION set
best_k, best_score = None, 0
for k in [1, 3, 5, 7, 9]:
    s = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).score(X_val, y_val)
    print(k, round(s, 3))
    if s > best_score:
        best_k, best_score = k, s

# 4) retrain with the chosen k, then evaluate on the test set ONCE
final = KNeighborsClassifier(n_neighbors=best_k).fit(X_tmp, y_tmp)
print("validation:", best_score, "| test:", final.score(X_test, y_test))
# validation score is optimistic (we picked the winner); the test score is the realistic one
```

### 7.5 Tuning with cross-validation (no fixed validation set)

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9]}, cv=5)
grid.fit(X_tmp, y_tmp)                     # only the training portion

print("best params:", grid.best_params_)
print("final test score (used once):", grid.score(X_test, y_test))
```

### 7.6 Grouped and time-ordered data

```python
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

# customer_id: all rows of one customer stay together (never split across train/test)
# scores = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5), groups=customer_id)

# time series: always train on the past, test on the future
# scores = cross_val_score(pipe, X, y, cv=TimeSeriesSplit(n_splits=5))
```

---

## 8. Checklist, summary and practice questions

### Pre-flight checklist

- [ ] Does every feature exist **at prediction time**?
- [ ] Is any feature derived from (or a consequence of) the target?
- [ ] Are duplicates removed **before** splitting?
- [ ] Is the data split **before** preprocessing?
- [ ] Is all preprocessing inside a **pipeline** (fit on training folds only)?
- [ ] Are hyperparameters tuned on **validation / CV**, not on the test set?
- [ ] Is the test set used **only once**?
- [ ] Do grouped or time-ordered rows stay on the correct side of the split?
- [ ] Is the performance suspiciously good? If so, investigate.

### Summary

| Leakage type | Cause | Fix |
|---|---|---|
| **Target leakage** | Feature unavailable at prediction time or derived from target | Remove or rebuild the feature |
| **Multicollinearity with target** | Feature is almost a copy of the target | Check the source; drop |
| **Duplicates** | Same row in train and test | `drop_duplicates()` before splitting |
| **Preprocessing leakage** | Statistics (mean, σ, selected features) learned from all data | Split first; use a pipeline; fit on train only |
| **Hyperparameter tuning** | Test set used to choose settings | Validation set or CV; test once |

**Key idea:** the test set must simulate the future. Anything you do using information from the test set, directly or indirectly, breaks that simulation.

### Practice questions

1. A model scores 95% on the test set but only 70% after deployment. Name two possible causes related to data leakage.
2. Why is `reversed_transaction` a leaky feature for fraud detection?
3. Why does data leakage make the *training/validation* performance look **better**, not worse?
4. You fill missing ages with the mean of the full dataset before splitting. What is wrong with this?
5. Why does `make_pipeline(scaler, model)` inside `cross_val_score` fix preprocessing leakage?
6. How can duplicated rows inflate the accuracy of a KNN model?
7. You try `n_neighbors = 1, 2, 3, 4`, pick the best on the test set, and report that test score. What's wrong?
8. What are the roles of the train, validation, and test sets? Which one can be used only once?
9. A customer has 10 rows in the dataset. Why might a random row-wise split cause leakage, and how can you avoid it?
10. A feature has a 0.9 correlation with the target. Is it automatically leakage? What would you check?
