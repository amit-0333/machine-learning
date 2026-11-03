# Hyperparameter Tuning in Machine Learning

A complete, beginner-to-intermediate guide: what hyperparameters are, why tuning is needed, and how to do it with `GridSearchCV`, `RandomizedSearchCV`, and smarter modern methods.

## Table of Contents

1. [Parameters vs Hyperparameters](#1-parameters-vs-hyperparameters)
2. [Requirement: Why Do We Need Hyperparameter Tuning?](#2-requirement-why-do-we-need-hyperparameter-tuning)
3. [Prerequisite: Cross-Validation](#3-prerequisite-cross-validation)
4. [GridSearchCV](#4-gridsearchcv)
5. [RandomizedSearchCV](#5-randomizedsearchcv)
6. [GridSearchCV vs RandomizedSearchCV](#6-gridsearchcv-vs-randomizedsearchcv)
7. [Can This Be Improved?](#7-can-this-be-improved)
8. [Best Practices & Common Mistakes](#8-best-practices--common-mistakes)
9. [Cheat Sheet](#9-cheat-sheet)
10. [Further Reading](#10-further-reading)

---

## 1. Parameters vs Hyperparameters

### 1.1 Parameters

Parameters are the **internal variables of a model that are learned from the data** during training. They define the model's representation of the underlying patterns in the data.

| Model | Parameters |
|---|---|
| Linear regression | Coefficients (slopes) and intercept |
| Neural network | Weights and biases of the nodes |
| Decision tree | Split points and split criteria at each node |

The goal of training is to find the values of these parameters that **minimize the discrepancy between the model's predictions and the actual outcomes** (i.e., minimize the loss function).

> You never set parameters by hand. The learning algorithm (e.g., gradient descent, least squares) finds them.

### 1.2 Hyperparameters

Hyperparameters are values that are **set before the learning process begins**. They are **not learned from the data** and must be predefined. They control *how* the learning happens and can significantly influence model performance.

| Model | Hyperparameters |
|---|---|
| Neural network | Learning rate, number of layers, number of nodes per layer, batch size, epochs, dropout rate |
| Support Vector Machine | Regularization parameter `C`, kernel type, `gamma` |
| Decision tree | Maximum depth, minimum samples per split, minimum samples per leaf |
| Random forest | Number of trees (`n_estimators`), `max_features`, `max_depth` |
| k-Nearest Neighbours | Number of neighbours `k`, distance metric, weighting |
| Ridge / Lasso | Regularization strength `alpha` |

The best values for hyperparameters **often cannot be determined in advance** and must be found through **trial and error**. Automating that trial and error is exactly what hyperparameter tuning is.

### 1.3 Why the word "hyper"?

The choice of the word is primarily a naming convention to differentiate between the two types of values (internal parameters and guiding parameters) that influence the behaviour of a machine learning model. It is also a nod to the fact that the role they play is a *meta* one: they control the structural aspects of the learning process itself, rather than being part of the direct pattern-finding mission of the model.

### 1.4 Side-by-side comparison

| | Parameters | Hyperparameters |
|---|---|---|
| **Who sets them?** | The learning algorithm | The data scientist (you) |
| **When?** | During training | Before training |
| **Learned from data?** | Yes | No |
| **Purpose** | Capture patterns in the data | Control the learning process / model complexity |
| **Stored in the final model?** | Yes, they *are* the model | Only as configuration |
| **Examples** | Weights, coefficients, split points | Learning rate, `C`, `max_depth`, `k` |

### 1.5 Quick example in scikit-learn

```python
from sklearn.linear_model import Ridge
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = Ridge(alpha=1.0)   # alpha  -> HYPERPARAMETER (you chose it)
model.fit(X, y)

print(model.coef_)         # coef_      -> PARAMETER (learned from data)
print(model.intercept_)    # intercept_ -> PARAMETER (learned from data)
```

A handy rule in scikit-learn: attributes that **end with an underscore** (`coef_`, `intercept_`, `feature_importances_`) are learned parameters. Arguments passed to the constructor (`alpha`, `max_depth`) are hyperparameters.

---

## 2. Requirement: Why Do We Need Hyperparameter Tuning?

### 2.1 The problem

Every algorithm ships with **default hyperparameter values**, but defaults are generic. They are not tailored to *your* dataset. Poorly chosen hyperparameters lead to:

- **Underfitting (high bias):** model is too simple (e.g., `max_depth=1`, very strong regularization). It misses real patterns and performs badly on both training and test data.
- **Overfitting (high variance):** model is too complex (e.g., unlimited `max_depth`, huge `C`). It memorizes noise, scoring great on training data but poorly on unseen data.

Hyperparameters are the knobs that move a model along this **bias-variance trade-off**. The aim of tuning is to find the setting that **generalizes best to unseen data**.

```
Model complexity  --->  low                                    high
Training error          high  ............................     low
Validation error        high  ......... (sweet spot) .........  high
                        UNDERFIT                               OVERFIT
```

### 2.2 Why not tune by hand?

Manual tuning (change a value, retrain, compare, repeat) breaks down quickly:

1. **Too many combinations.** A model with 4 hyperparameters and 5 candidate values each has 5⁴ = 625 combinations.
2. **Hyperparameters interact.** The best `C` for an SVM depends on `gamma` and the kernel, so you can't tune them one at a time reliably.
3. **Not reproducible or fair.** Ad-hoc experiments are hard to track and easy to bias.
4. **Evaluation noise.** A single train/test split can be lucky or unlucky. You need a robust estimate, which means cross-validation.

### 2.3 What we actually require

A good tuning procedure needs:

| Requirement | Solved by |
|---|---|
| A list of candidate hyperparameter values to try | A *search space* (`param_grid` / `param_distributions`) |
| A way to try them systematically | A *search strategy* (grid, random, Bayesian, ...) |
| A reliable measure of how good each combination is | *Cross-validation* with a *scoring metric* |
| Pick the best and retrain on all training data | `refit=True` -> `best_estimator_` |
| An honest final performance number | A held-out **test set** never touched during tuning |

`GridSearchCV` and `RandomizedSearchCV` bundle all of this into a single tool.

---

## 3. Prerequisite: Cross-Validation

Both search tools use **cross-validation (CV)** to score each hyperparameter combination, so understand it first.

### 3.1 k-Fold CV

1. Split the training data into **k equal folds** (commonly k = 5).
2. For each fold *i*: train on the other k-1 folds, validate on fold *i*.
3. Average the k validation scores. That average is the CV score.

```
Fold 1: [VAL][ T ][ T ][ T ][ T ]
Fold 2: [ T ][VAL][ T ][ T ][ T ]
Fold 3: [ T ][ T ][VAL][ T ][ T ]
Fold 4: [ T ][ T ][ T ][VAL][ T ]
Fold 5: [ T ][ T ][ T ][ T ][VAL]      -> mean of 5 scores
```

### 3.2 Why use it?

- Every data point is used for both training and validation, so you get a **more stable estimate** than a single split.
- It reduces the chance that a hyperparameter looks good just because of one lucky split.

### 3.3 Variants

- **Stratified k-Fold:** keeps class proportions in every fold. This is the default for classifiers in scikit-learn and is important for imbalanced data.
- **Repeated k-Fold:** repeats k-fold with different shuffles for even more stable estimates.
- **TimeSeriesSplit:** for time-ordered data. Never shuffle time series!

### 3.4 The three-way split

```
Full data
 ├── Training set  ──► used by GridSearchCV / RandomizedSearchCV
 │                     (internally split into CV train/validation folds)
 └── Test set      ──► touched ONCE at the very end for the final score
```

> **Golden rule:** never use the test set to choose hyperparameters. Otherwise your final score is optimistically biased (a form of data leakage).

---

## 4. GridSearchCV

### 4.1 Definition

`GridSearchCV` is an algorithm that performs an **exhaustive search over a specified grid of hyperparameters**, using **cross-validation** to determine which hyperparameter combination gives the best model performance.

### 4.2 How it works

1. You define a **grid**: for each hyperparameter, a list of values to try.
2. It builds every possible combination (the Cartesian product).
3. For each combination, it runs k-fold CV and records the mean score.
4. It picks the combination with the best mean CV score.
5. (If `refit=True`) it retrains the model with the best combination on the **entire** training set.

**Example:** with

```python
param_grid = {
    "C": [0.1, 1, 10],          # 3 values
    "kernel": ["linear", "rbf"],# 2 values
    "gamma": [0.01, 0.1, 1],    # 3 values
}
```

there are 3 x 2 x 3 = **18 combinations**. With `cv=5`, the model is trained **18 x 5 = 90 times** (plus 1 final refit).

### 4.3 Basic code example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Pipeline (scaling happens INSIDE each CV fold -> no leakage)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

# 3. Grid  (note the "<step>__<param>" naming for pipelines)
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", 0.01, 0.1],
}

# 4. Search
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,      # use all CPU cores
    verbose=1,
    refit=True,
)
grid.fit(X_train, y_train)

# 5. Results
print("Best params :", grid.best_params_)
print("Best CV score:", grid.best_score_)

# 6. Final, honest evaluation on the untouched test set
print("Test score  :", grid.score(X_test, y_test))
```

### 4.4 Important arguments

| Argument | Meaning |
|---|---|
| `estimator` | The model or pipeline to tune |
| `param_grid` | Dict (or list of dicts) of hyperparameter name -> list of values |
| `scoring` | Metric to optimize: `"accuracy"`, `"f1"`, `"roc_auc"`, `"neg_mean_squared_error"`, `"r2"`, or a custom scorer. For multiple metrics, pass a list/dict and set `refit` to the one to select by |
| `cv` | Number of folds (int) or a CV splitter object (`StratifiedKFold(...)`, `TimeSeriesSplit(...)`) |
| `n_jobs` | Parallel jobs. `-1` = all cores |
| `refit` | If `True`, retrain on the full training set using the best params; exposes `best_estimator_` |
| `verbose` | Amount of progress output |
| `return_train_score` | Also compute training scores (helps diagnose overfitting; slower) |
| `error_score` | What to do when a combination fails (`np.nan` or `"raise"`) |

### 4.5 Important attributes (available after `.fit`)

| Attribute | Meaning |
|---|---|
| `best_params_` | Best hyperparameter combination |
| `best_score_` | Mean CV score of the best combination |
| `best_estimator_` | The refitted model with the best params |
| `cv_results_` | Dict with every combination's scores, timings, ranks (turn into a DataFrame) |
| `best_index_` | Index of the best combination in `cv_results_` |

```python
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
print(results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
      .sort_values("rank_test_score").head())
```

### 4.6 Searching different model families / conditional spaces

Pass a **list of dicts** when some hyperparameters only make sense together (e.g., `gamma` is irrelevant for a linear kernel):

```python
param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
    {"svc__kernel": ["rbf"],    "svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]},
]
```

This avoids wasting time on meaningless combinations.

### 4.7 Advantages

- **Exhaustive:** within the grid, it is guaranteed to find the best combination.
- **Simple and deterministic:** easy to understand, explain, and reproduce.
- **Great for small search spaces** (few hyperparameters, few values).
- Built-in CV, parallelism, and refit.

### 4.8 Disadvantages

- **Combinatorial explosion ("curse of dimensionality"):** adding one more hyperparameter multiplies the total cost.

  `total fits = (product of number of values per hyperparameter) x cv folds`

- **Wastes effort** on unimportant hyperparameters: it tries every value of a useless hyperparameter against every value of the useful ones.
- **Limited to the grid:** the true optimum may lie *between* grid points.
- **Only works well for discrete or pre-discretized values:** continuous hyperparameters must be manually discretized.

---

## 5. RandomizedSearchCV

### 5.1 Definition

`RandomizedSearchCV` **randomly samples a fixed number of combinations** from the hyperparameter space (`n_iter`) instead of trying all of them. Each sampled combination is scored with cross-validation, exactly as in `GridSearchCV`. Hyperparameters can be given as **lists** (sampled uniformly) or as **probability distributions** (e.g., from `scipy.stats`).

### 5.2 How it works

1. You define a **search space**: lists or distributions per hyperparameter.
2. You choose a **budget**: `n_iter` (how many random combinations to try).
3. For each of the `n_iter` iterations, it draws one random value per hyperparameter.
4. Each combination is evaluated by k-fold CV.
5. Best combination -> refit on the full training set.

**Cost:** `n_iter x cv` fits, **regardless of how large the space is**. You control the budget directly.

### 5.3 Code example

```python
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "n_estimators": randint(100, 1000),        # integers 100..999
    "max_depth": [None, 5, 10, 20, 30],        # list -> uniform choice
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.5],
}

rand = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,            # only 50 random combinations
    scoring="f1",
    cv=5,
    n_jobs=-1,
    random_state=42,      # reproducible sampling
    verbose=1,
)
rand.fit(X_train, y_train)

print("Best params :", rand.best_params_)
print("Best CV score:", rand.best_score_)
print("Test score  :", rand.score(X_test, y_test))
```

For scale-type hyperparameters, use a **log-uniform** distribution:

```python
from scipy.stats import loguniform

param_distributions = {
    "svc__C": loguniform(1e-3, 1e3),      # samples evenly across orders of magnitude
    "svc__gamma": loguniform(1e-4, 1e1),
    "svc__kernel": ["rbf"],
}
```

> **Why log scale?** The difference between C = 0.001 and 0.01 matters as much as between 100 and 1000. A plain uniform distribution over [0.001, 1000] would almost never sample small values.

### 5.4 Extra argument vs GridSearchCV

| Argument | Meaning |
|---|---|
| `param_distributions` | Dict of name -> list **or** distribution object with an `.rvs()` method |
| `n_iter` | Number of random combinations to try (your compute budget) |
| `random_state` | Seed for reproducible sampling |

All other arguments (`scoring`, `cv`, `n_jobs`, `refit`, ...) and attributes (`best_params_`, `cv_results_`, ...) are the same as for `GridSearchCV`.

### 5.5 Why random search works so well

A well-known study (Bergstra & Bengio, 2012) showed that in most problems **only a few hyperparameters really matter**, and which ones matter varies by dataset.

- A grid of 9 points on 2 hyperparameters only tests **3 distinct values** of each.
- 9 random points test **9 distinct values** of each hyperparameter.

So random search explores the important dimensions far more thoroughly for the same budget.

```
Grid (9 trials)              Random (9 trials)
 x   x   x                     x      x
 x   x   x                  x     x     x
 x   x   x                     x   x  x
 -> only 3 unique values    -> 9 unique values on each axis
    per axis
```

### 5.6 Advantages

- **Budget control:** you decide how long it runs via `n_iter`.
- **Scales to high-dimensional spaces** much better than grid search.
- **Handles continuous hyperparameters naturally** through distributions.
- Often finds a **near-optimal** result in a fraction of the time.
- Can be stopped/extended easily (just increase `n_iter`).

### 5.7 Disadvantages

- **No guarantee** of finding the true best combination. It is stochastic.
- Results can vary between runs unless `random_state` is fixed.
- Still **uninformed**: it does not learn from previous trials, so it can keep sampling bad regions.
- Choosing sensible distributions/ranges requires some domain knowledge.

---

## 6. GridSearchCV vs RandomizedSearchCV

| Aspect | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Search strategy | Exhaustive over all combinations | Random sampling of `n_iter` combinations |
| Total fits | (product of all value counts) x cv | `n_iter` x cv |
| Cost grows with | Number of hyperparameters (exponentially) | Only `n_iter` (you choose) |
| Continuous hyperparameters | Must be discretized manually | Sampled from distributions |
| Finds the best in the space? | Best **in the grid** (guaranteed) | Not guaranteed, usually near-best |
| Reproducibility | Deterministic | Needs `random_state` |
| Best for | Small spaces, final fine-tuning | Large spaces, first exploration |
| Typical use | 2-3 hyperparameters, few values | 4+ hyperparameters, wide ranges |

**Rule of thumb:** start with `RandomizedSearchCV` over a wide range to find the promising region, then (optionally) run a small `GridSearchCV` around the best values to polish.

---

## 7. Can This Be Improved?

Yes. Both methods share two weaknesses: they are **computationally expensive** (every trial is a full CV run) and **uninformed** (a new trial doesn't learn from earlier ones). Below are ways to improve.

### 7.1 Coarse-to-fine search (manual, easy win)

1. Run `RandomizedSearchCV` on a **wide** space.
2. Inspect `cv_results_` to see where the best scores cluster.
3. Run a second, **narrower** search (random or grid) around that region.

```python
# Stage 1: wide
# C in loguniform(1e-3, 1e3)  ->  best C ~ 8

# Stage 2: narrow
param_grid = {"svc__C": [4, 6, 8, 10, 12, 16]}
```

### 7.2 Successive Halving (`HalvingGridSearchCV`, `HalvingRandomSearchCV`)

Idea: **don't give every candidate the full budget.** Start all candidates on a small amount of data/resources, keep only the best fraction (e.g., top 1/3), give the survivors more resources, and repeat.

```python
from sklearn.experimental import enable_halving_search_cv  # noqa  (required)
from sklearn.model_selection import HalvingRandomSearchCV

halving = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    factor=3,                 # keep 1/3 of candidates each round
    resource="n_samples",     # grow the training-set size each round
    cv=5,
    random_state=42,
    n_jobs=-1,
)
halving.fit(X_train, y_train)
```

- Much **faster** on large datasets / many candidates.
- Still lives in scikit-learn and has the same API. (It is flagged *experimental*, hence the special import.)

### 7.3 Bayesian Optimization (Optuna, scikit-optimize, Hyperopt)

Idea: build a **probabilistic model of the objective** (score as a function of hyperparameters) from previous trials, then choose the *next* trial where improvement is most likely. It balances **exploration** (try uncertain regions) and **exploitation** (refine good regions).

- Typically needs **far fewer trials** than random search to reach the same quality.
- Great when each trial is expensive (deep learning, big ensembles).

Example with **Optuna**:

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(study.best_params, study.best_value)
```

(Install with `pip install optuna`.)

### 7.4 Pruning & early stopping

Stop unpromising trials **early** instead of letting them finish (Optuna's pruners, Hyperband / ASHA, early stopping in XGBoost/LightGBM/Keras). This can save a large share of compute.

### 7.5 Other practical speed-ups

| Technique | How it helps |
|---|---|
| `n_jobs=-1` | Run folds/candidates in parallel |
| Fewer folds (`cv=3`) during exploration | Cuts cost by ~40% vs 5-fold |
| Tune on a **subsample** first | Quickly narrow the range, then refine on full data |
| Use faster models/solvers | e.g., `HistGradientBoosting` instead of plain `GradientBoosting` |
| Cache expensive pipeline steps | `Pipeline(memory=...)` |
| Smaller, smarter search space | Remove irrelevant hyperparameters; use log scales |

### 7.6 Better evaluation

- Pick a **scoring metric that matches the business goal** (e.g., `recall` for fraud detection, `roc_auc`/`average_precision` for imbalanced data), not just accuracy.
- Use **nested cross-validation** (an outer CV around the search) for an unbiased estimate of tuned-model performance, especially on small datasets.
- Apply **the one-standard-error rule:** among near-equal scores, pick the *simpler* model to reduce overfitting risk.

### 7.7 Method comparison

| Method | Informed? | Cost | Best when |
|---|---|---|---|
| Grid search | No | Very high (explodes) | Tiny search spaces |
| Random search | No | Controlled (`n_iter`) | Medium/large spaces, baseline |
| Successive halving | Partly (drops losers early) | Lower | Large datasets, many candidates |
| Bayesian optimization | **Yes** | Low number of trials | Expensive models, limited budget |
| Hyperband / ASHA | Partly | Low | Deep learning, iterative models |

---

## 8. Best Practices & Common Mistakes

**Do**

- Split off a **test set first** and leave it alone until the end.
- Put preprocessing (scaling, imputation, encoding) **inside a `Pipeline`** so it is fit only on each CV training fold.
- Use **stratified** CV for classification, **time-based** CV for time series.
- Use **log scales** for hyperparameters like `C`, `gamma`, `alpha`, learning rate.
- Set `random_state` for reproducibility; set `n_jobs=-1` for speed.
- Inspect `cv_results_` (look at `std_test_score` and train vs validation gap).
- Retrain the final model on **all available data** (train + validation) if appropriate, after you've chosen hyperparameters.

**Don't**

- Don't tune on the test set (data leakage -> inflated results).
- Don't scale/impute the whole dataset *before* cross-validation (leakage).
- Don't use huge grids "just in case". Think about which hyperparameters matter.
- Don't trust a tiny score difference. Check the standard deviation across folds.
- Don't tune endlessly: after a point, **better features/data** beat more tuning.

---

## 9. Cheat Sheet

```python
# ---------- Grid ----------
GridSearchCV(estimator, param_grid, scoring="f1", cv=5, n_jobs=-1, refit=True)

# ---------- Random ----------
RandomizedSearchCV(estimator, param_distributions, n_iter=50, scoring="f1",
                   cv=5, n_jobs=-1, random_state=42)

# ---------- After fitting ----------
search.best_params_      # best hyperparameters
search.best_score_       # best mean CV score
search.best_estimator_   # refitted best model
search.cv_results_       # full results table
search.score(X_test, y_test)   # final test evaluation
```

| Question | Answer |
|---|---|
| Parameter or hyperparameter? | Learned from data -> parameter. Set by you before training -> hyperparameter |
| Few hyperparameters, few values? | `GridSearchCV` |
| Many hyperparameters / wide ranges? | `RandomizedSearchCV` |
| Expensive model, limited budget? | Bayesian optimization (Optuna) / Halving |
| Where do I evaluate the final model? | On the untouched **test set** |

---

## 10. Further Reading

- scikit-learn User Guide: *Tuning the hyper-parameters of an estimator*: https://scikit-learn.org/stable/modules/grid_search.html
- scikit-learn: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`, `HalvingRandomSearchCV` API docs
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR
- Optuna documentation: https://optuna.org
- Li et al. (2018), *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*
