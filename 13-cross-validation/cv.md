# Cross-Validation

> Cross-validation estimates how well a model will perform on **unseen data** by repeatedly training on some parts of the data and testing on the rest, then averaging the results. It is the foundation of **model selection** and **hyperparameter tuning**.

---

## Table of Contents

1. [Recap and motivation](#1-recap-and-motivation)
2. [The problem: how do we evaluate a model?](#2-the-problem-how-do-we-evaluate-a-model)
3. [The hold-out approach](#3-the-hold-out-approach)
4. [Problems with the hold-out approach](#4-problems-with-the-hold-out-approach)
5. [Why is hold-out still used?](#5-why-is-hold-out-still-used)
6. [Cross-validation: the idea and its family](#6-cross-validation-the-idea-and-its-family)
7. [Leave-One-Out CV (LOOCV)](#7-leave-one-out-cv-loocv)
8. [K-Fold CV](#8-k-fold-cv)
9. [Stratified K-Fold CV](#9-stratified-k-fold-cv)
10. [Other variants (overview)](#10-other-variants-overview)
11. [Code examples](#11-code-examples)
12. [Which method should I use?](#12-which-method-should-i-use)
13. [Summary and practice questions](#13-summary-and-practice-questions)

---

## 1. Recap and motivation

From the linear regression case study, two ideas drive **model selection**:

```
Linear regression case study
        ├── Cross-validation   ─┐
        └── Hyperparameter     ─┴──►  Model selection
```

Cross-validation is used for three related purposes:

| # | Purpose | Meaning |
|---|---|---|
| 1 | **Reliable evaluation** | Get a trustworthy estimate of performance, not one lucky or unlucky number |
| 2 | **Avoid data leakage** | Don't let information from the test data influence the model |
| 3 | **Hyperparameter tuning** | Compare settings (e.g. `k` in KNN, regularization strength) fairly |

---

## 2. The problem: how do we evaluate a model?

In supervised ML we have a dataset with features `X` and a target `y`:

```
  ←── X (features) ──→ | y (target)
  ┌────┬────┬────┬────┬────┐
  │    │    │    │    │    │      ──►  train a model  ──►  use on NEW data
  └────┴────┴────┴────┴────┘
```

Example: a ride-hailing company (Ola / Uber) trains a model on historical trips and then **deploys** it to make predictions on **new, unseen** trips.

```
X, y  ──► model  ──► evaluate  ──► deploy  ──► new data
                        ▲
                  How good will it be
                  on data it has never seen?
```

If we evaluate the model on the **same data it trained on**, the score is over-optimistic (the model has already seen those answers). We need to estimate performance on data the model **hasn't** seen **before** deploying it. That is the job of model evaluation, and cross-validation is the most robust way to do it.

---

## 3. The hold-out approach

The simplest method is `train_test_split`:

```
1000 customers
        │
   1) Shuffle
   2) Split into train / test
        │
  ┌─────────────────────────┬────────────┐
  │   Training set (0.75)   │ Test (0.25)│
  └─────────────────────────┴────────────┘
          │                        │
     train the model ───────► predict  ──► y_pred
                                            │
                                  compare with y_true  ──►  performance metric
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# compare y_pred with y_test -> accuracy, MSE, etc.
```

This is simple, but it has some serious weaknesses.

---

## 4. Problems with the hold-out approach

| # | Problem | Explanation |
|---|---|---|
| 1 | **Variability** | The score depends heavily on *which* rows land in the test set. An unlucky split can give a training set that isn't representative, or a test set that is unusually easy or hard. The performance estimate has **high variance**. |
| 2 | **Data inefficiency** | The model trains on only part of the data (and is tested on a different part), so it never learns from everything. Especially painful for small datasets. |
| 3 | **Biased estimate** | If some classes or patterns are over- or under-represented in the train or test portion by chance, the estimate is **biased**. |
| 4 | **Unreliable for hyperparameter tuning** | If you tune hyperparameters by repeatedly checking the test set, information **leaks** from the test set into your choices. You overfit to the test set, and the score looks better than reality (**data leakage**). |

### See problem 1 in action

Same model, same data, only the random split changes:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

accs = []
for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    accs.append(LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te))

print(np.round(accs, 3))
print(min(accs), max(accs))     # ~0.895 to 1.0 on this data
```

The "accuracy" ranges from about **89% to 100%** purely because of the split. Which of these would you trust?

---

## 5. Why is hold-out still used?

Despite the problems, it has real advantages:

| # | Advantage | Explanation |
|---|---|---|
| 1 | **Simplicity** | Easy to understand and implement: two sets, one model. Great for initial exploration and simple projects. |
| 2 | **Computational efficiency** | The model is trained **once**. K-fold needs `K` trainings, which is costly for big data or complex models. |
| 3 | **Large datasets** | With a huge dataset, even a small test portion is a representative sample, so hold-out works well. |

---

## 6. Cross-validation: the idea and its family

> Divide the data into several subsets (**folds**). Train the model on some folds and test it on the remaining ones. **Repeat** with different folds playing the role of the test set each time. **Average** the results to estimate overall performance.

### Where it fits: resampling

Cross-validation is a **resampling** technique: we repeatedly re-use the one sample of data we have to learn how a model would behave on the wider population.

```
                        Resampling
                 ┌──────────┴───────────┐
          Cross-validation           Bootstrap
        ┌──────┴────────┐
 Leave-One-Out CV     K-Fold CV family
 Leave-p-Out CV       ├── K-Fold
                      ├── Stratified K-Fold
                      ├── Nested K-Fold
                      └── Repeated K-Fold

 (plus time-based CV for time-series data)
```

| Family | Idea |
|---|---|
| **Cross-validation** | Split into folds; each row is used for testing in a systematic way |
| **Bootstrap** | Repeatedly draw random samples **with replacement** from the dataset (e.g. from the 150 iris rows) and train a model on each sample |

In scikit-learn, the one-line API is `cross_val_score`.

### Key terms

| Term | Meaning |
|---|---|
| **Fold** | One chunk of the dataset |
| **Split** | One train/test arrangement |
| **CV score** | Average of the metric over all splits |

---

## 7. Leave-One-Out CV (LOOCV)

### How it works

With `n` rows: train on `n - 1` rows, test on the **1 left-out row**. Repeat for every row: `n` models, `n` tests. Average the `n` results.

```
n = 5 rows:

 split 1:  [TEST] train train train train
 split 2:  train [TEST] train train train
 split 3:  train train [TEST] train train
 split 4:  train train train [TEST] train
 split 5:  train train train train [TEST]

 -> 5 models, average the 5 scores
```

### Advantages

| # | Advantage | Explanation |
|---|---|---|
| 1 | **Uses almost all data for training** | Valuable when the dataset is small and every point matters |
| 2 | **Low bias** | Each model trains on `n - 1` rows, almost the full dataset, so the estimate is much less dependent on the partition |
| 3 | **No randomness** | No random split, so the result is the same every time (stable) |

### Disadvantages

| # | Disadvantage | Explanation |
|---|---|---|
| 1 | **Computationally expensive** | Needs `n` model fits (10,000 rows = 10,000 models) |
| 2 | **High variance** | The `n` training sets are nearly identical, so the models are highly correlated and the average can fluctuate |
| 3 | **Some metrics don't work** | Metrics like **R²** are undefined for a single-sample validation set (use e.g. MSE/MAE per prediction and average) |
| 4 | **Not ideal for imbalanced data** | One validation sample can't represent the class distribution |

### When to use

- **Small datasets** (maximize training data)
- **Balanced datasets** (class imbalance can leave a class missing from training)
- When you need a **low-bias** estimate

---

## 8. K-Fold CV

The **most used** CV technique, a general and practical variant of the idea.

### How it works (`K = 5`, 100 rows)

```
1. Split the data into K equal folds   ->  5 folds x 20 rows
2. Each round, hold out ONE fold as the test set,
   and train on the other K-1 folds
3. Repeat K times (every fold is the test set exactly once)
4. Average the K scores
```

```
            Fold 1  Fold 2  Fold 3  Fold 4  Fold 5
 Split 1:   TEST    train   train   train   train   -> model m1 -> score 1
 Split 2:   train   TEST    train   train   train   -> model m2 -> score 2
 Split 3:   train   train   TEST    train   train   -> model m3 -> score 3
 Split 4:   train   train   train   TEST    train   -> model m4 -> score 4
 Split 5:   train   train   train   train   TEST    -> model m5 -> score 5

                       CV score = average(score 1 ... score 5)
```

Typical choices: **K = 5 or K = 10**.
Example: 1000 rows with 5 folds = **200 rows per fold**, 5 models.

### Role in the full workflow

```
 All data
   ├── Training data  ──►  K-fold CV  (find parameters / compare models)
   └── Test data      ──►  FINAL evaluation on truly unseen data (used once)
```

Cross-validation is used to **choose** the model or hyperparameters. The held-out **test data** stays untouched until the end, which prevents data leakage.

### K and bias/variance

| Choice | Effect |
|---|---|
| **LOOCV** (`K = n`) | Each model trains on `n - 1` rows: **bias is low**, but cost is high (`n` models) |
| **Small K** (e.g. 2 or 3) | Each model trains on much less data than the final model: **higher bias** |
| **K = 5 or 10** | Good compromise: modest cost (5 or 10 models) and low bias |

### Advantages

| # | Advantage |
|---|---|
| 1 | **Reduces variance**: averaging over `K` different partitions makes the estimate less sensitive to any single random split |
| 2 | **Computationally cheaper** than LOOCV: fewer models (`K = 5, 10`) means less time and memory |

### Disadvantages

| # | Disadvantage |
|---|---|
| 1 | **Potential for high bias** if `K` is too small (models train on too little data / folds not representative) |
| 2 | **Imbalanced classes**: some folds may contain no minority-class samples, giving misleading metrics (fix: stratified K-fold) |

### When to use

- **Reasonably large datasets** (the `K` trainings are worth it for a more reliable estimate)
- **Evenly distributed** (balanced) data. For imbalanced data, use **Stratified K-Fold**.

---

## 9. Stratified K-Fold CV

### The problem

Imbalanced **classification**: say 95% of the rows are class **Y** and only 5% are **N**.

With ordinary K-fold, a fold might have *no* N samples at all. A model that always predicts "Y" then scores **100%** on that fold, and the average is misleadingly high.

```
Plain K-fold (random / unlucky split):          Stratified K-fold:
 Fold 1: 100% Y,   0% N   <- no minority!       Fold 1: 95% Y, 5% N
 Fold 2:  98% Y,   2% N                         Fold 2: 95% Y, 5% N
 Fold 3:  90% Y,  10% N                         Fold 3: 95% Y, 5% N
        ...                                            ...
```

### The idea

**Stratified K-fold keeps the class proportions in every fold the same as in the whole dataset.** Example: 1000 rows with 950 Y and 50 N, 5 folds: each fold gets **190 Y and 10 N**.

```
whole data:    ■■■■■■■■■■■■■■■■■■■□   (95% Y, 5% N)
every fold:    ■■■■■■■■■■■■■■■■■■■□   (95% Y, 5% N)   same ratio each time
```

Each model is then trained and tested on data that looks like the full dataset, so the scores (e.g. 70%, 72%, ... in your notes' example) are **realistic**, and their **average is a realistic value**.

### Works for multi-class too

If there are 3 classes with counts 2, 4 and 3 in a small group, stratification splits **each class** across the folds proportionally rather than shuffling everything together.

### When to use

- **Classification with imbalanced classes** (use it as the default for classification)
- Small datasets where a random split could accidentally drop a class
- Not applicable to regression (no classes), unless you stratify on binned target values

---

## 10. Other variants (overview)

Your notes list these under the CV family. Brief descriptions:

| Variant | Idea | Notes |
|---|---|---|
| **Leave-p-Out (LPO)** | Test on every possible subset of `p` rows | Number of splits explodes (`C(n, p)`), rarely practical |
| **Repeated K-Fold** | Run K-fold several times with different shuffles and average everything | More stable estimate; costs `K x repeats` fits |
| **Nested K-Fold** | **Inner** loop tunes hyperparameters; **outer** loop evaluates the whole tuning procedure | Gives an honest estimate when you tune hyperparameters |
| **Time-based / Time-series CV** | Always train on the **past** and test on the **future** | Don't shuffle time-ordered data! |

```python
# Time-series split: training window grows, test is always later
#   train [0 1]            test [2 3]
#   train [0 1 2 3]        test [4 5]
#   train [0 1 2 3 4 5]    test [6 7]
#   train [0 ... 7]        test [8 9]
```

---

## 11. Code examples

### 11.1 K-Fold CV with `cross_val_score`

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print("Fold scores:", scores.round(3))               # [1.    1.    0.933 0.967 0.967]
print("Mean = %.3f, Std = %.3f" % (scores.mean(), scores.std()))   # ~0.973, ~0.025
```

- The **mean** is the CV estimate; the **std** shows how much it varies across folds (something hold-out can't tell you).
- `cv=5` on a classifier automatically uses stratified folds.

### 11.2 What happens inside (manual loop)

```python
scores = []
for train_idx, test_idx in kf.split(X):
    m = LogisticRegression(max_iter=1000)
    m.fit(X[train_idx], y[train_idx])                 # train on K-1 folds
    scores.append(m.score(X[test_idx], y[test_idx]))  # test on the held-out fold

print(np.mean(scores))
```

### 11.3 Leave-One-Out CV

```python
from sklearn.model_selection import LeaveOneOut

scores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(len(scores))         # 150 models for 150 rows
print(scores.mean())       # ~0.967  (each score is 0 or 1 for one row)
```

For regression, use an error metric rather than R²:

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

Xd, yd = load_diabetes(return_X_y=True)
mse = -cross_val_score(LinearRegression(), Xd, yd,
                       cv=LeaveOneOut(), scoring="neg_mean_squared_error").mean()
print("LOOCV MSE:", mse)
```

(sklearn reports errors as *negative* scores so that "higher is better" holds everywhere, hence the minus sign.)

### 11.4 Why stratification matters (imbalanced data)

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold

Xi, yi = make_classification(n_samples=1000, weights=[0.95, 0.05],
                             flip_y=0, random_state=0)
print(np.bincount(yi))                       # [950  50]

def class_counts(cv, y):
    return [np.bincount(y[te], minlength=2).tolist() for _, te in cv.split(Xi, y)]

print("KFold      :", class_counts(KFold(5, shuffle=True, random_state=0), yi))
print("Stratified :", class_counts(StratifiedKFold(5), yi))
# Stratified -> [[190, 10], [190, 10], [190, 10], [190, 10], [190, 10]]

# If the data happens to be ordered (e.g. sorted by class) and not shuffled:
order = np.argsort(yi)
print("KFold on sorted data:", [np.bincount(yi[order][te], minlength=2).tolist()
                                for _, te in KFold(5).split(Xi)])
# [[200, 0], [200, 0], [200, 0], [200, 0], [150, 50]]  <- 4 folds have NO minority samples!
```

Plain random K-fold gives slightly uneven counts; with ordered data it can be disastrous. Stratified K-fold gives exactly **190 / 10** in every fold.

```python
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
```

### 11.5 Hyperparameter tuning with CV (`GridSearchCV`)

Choose `n_neighbors` for KNN using 5-fold CV **on the training data only**, then touch the test set **once**:

```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11]}

gs = GridSearchCV(pipe, grid, cv=5)            # 6 settings x 5 folds = 30 fits
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV score:", gs.best_score_)         # estimate used for choosing
print("Test score:", gs.score(X_test, y_test))  # final check on unseen data, used once
```

> **Data-leakage rule:** put preprocessing (scaling, imputation, feature selection, PCA, ...) **inside the pipeline**, so it is fitted only on the training folds each round. If you scale the whole dataset *before* cross-validating, information from the validation fold leaks into training.

### 11.6 Several metrics and repeated K-fold

```python
from sklearn.model_selection import cross_validate, RepeatedKFold

res = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1_macro"])
print(res["test_accuracy"].mean(), res["test_f1_macro"].mean())

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)    # 15 fits
print(cross_val_score(model, X, y, cv=rkf).mean())
```

### 11.7 Time-series split

```python
from sklearn.model_selection import TimeSeriesSplit

for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(np.arange(10)):
    print("train", train_idx, "test", test_idx)
```

---

## 12. Which method should I use?

| Situation | Recommended |
|---|---|
| Huge dataset, quick experiment | **Hold-out** (a single split) |
| Small dataset, balanced | **LOOCV** (or K-fold with a large K) |
| General case, moderate to large data | **K-Fold, K = 5 or 10** |
| Classification, imbalanced classes | **Stratified K-Fold** |
| Want a more stable estimate | **Repeated (Stratified) K-Fold** |
| Tuning hyperparameters and reporting performance | **Nested CV**, or CV on train data plus one final test set |
| Time-ordered data | **Time-series split** (never shuffle) |

**Comparison**

| | Hold-out | K-Fold | LOOCV |
|---|---|---|---|
| Models trained | 1 | K | n |
| Cost | Lowest | Moderate | Highest |
| Training size per model | Partial | (K-1)/K of data | n - 1 rows |
| Bias of estimate | Higher | Low to moderate | Lowest |
| Variance of estimate | High (depends on the split) | Lower | Can be high (similar training sets) |
| Randomness | Yes | Yes (if shuffled) | None |

---

## 13. Summary and practice questions

**Summary**

- We must evaluate a model on **unseen data** before deploying it.
- **Hold-out** (`train_test_split`) is simple and fast, but its score depends on the split (variance, data inefficiency, bias, leakage when tuning).
- **Cross-validation** trains and tests on multiple splits and **averages** the scores for a more reliable estimate.
- **LOOCV:** `n` models, 1 test row each. Low bias, expensive, not for imbalanced data, and R² is undefined.
- **K-Fold (K = 5 or 10):** every fold is the test set once. Lower variance and much cheaper than LOOCV.
- **Stratified K-Fold:** keeps class proportions in every fold, which is essential for imbalanced classification.
- Use CV on the training data to **tune hyperparameters** and keep a final **test set** for one last unbiased check. Put preprocessing inside a **pipeline** to avoid leakage.

**Practice questions**

1. Why is evaluating a model on its own training data misleading?
2. Give two reasons why a hold-out score can be unreliable. How does K-fold address them?
3. With 1000 rows and 5-fold CV, how many models are trained, and how many rows are in each fold and each training set?
4. A dataset has 10,000 rows. How many models does LOOCV train? Why might you prefer K-fold here?
5. Why can't R² be computed in each LOOCV iteration?
6. Why can plain K-fold give a misleadingly high accuracy on a dataset with 95% / 5% class balance, and how does stratification fix it?
7. What is data leakage, and why does scaling the entire dataset before cross-validation cause it?
8. Why shouldn't you shuffle time-series data before cross-validating?
9. What is the difference between the CV score you use to pick a hyperparameter and the final test score?
10. When might plain hold-out be perfectly acceptable?
