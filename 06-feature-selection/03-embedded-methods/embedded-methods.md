# Embedded-Based Feature Selection

A comprehensive guide to understanding and applying embedded feature selection methods in machine learning pipelines.

---

## Table of Contents

- [Recap: All Three Feature Selection Types](#recap-all-three-feature-selection-types)
- [What are Embedded Methods?](#what-are-embedded-methods)
- [Linear Regression Assumptions](#linear-regression-assumptions)
- [Techniques](#techniques)
  - [1. Regularized Models](#1-regularized-models)
    - [Lasso Regression (L1)](#lasso-regression-l1)
    - [Ridge Regression (L2)](#ridge-regression-l2)
    - [Elastic Net](#elastic-net)
  - [2. Tree-Based Models](#2-tree-based-models)
    - [Random Forest Feature Importance](#random-forest-feature-importance)
  - [3. Recursive Feature Elimination (RFE)](#3-recursive-feature-elimination-rfe)
- [Advantages & Disadvantages](#advantages--disadvantages)
- [Complete Cheatsheet: All Feature Selection Methods](#complete-cheatsheet-all-feature-selection-methods)
- [Choosing the Right Method](#choosing-the-right-method)
- [Quick Reference](#quick-reference)

---

## Recap: All Three Feature Selection Types

| Type | How It Works | Model Required? | Speed | Overfitting Risk | Feature Interactions |
|------|-------------|-----------------|-------|-----------------|----------------------|
| **Filter** | Statistical scores per feature | No | Fast | Low | Not considered |
| **Wrapper** | Train model on feature subsets iteratively | Yes (external) | Slow | Higher | Considered |
| **Embedded** | Feature selection happens during model training | Yes (built-in) | Moderate | Low–Moderate | Considered |

> **Embedded methods aim to combine the best of both worlds** — they consider feature interactions like wrapper methods, but are more computationally efficient since the model is fit only once.

---

## What are Embedded Methods?

Embedded methods are feature selection techniques that **perform feature selection as part of the model construction process**. They are called "embedded" because feature selection is **embedded within** the training of the machine learning model itself.

**Key characteristics:**
- Feature selection and model training happen **simultaneously**
- They solve key limitations of filter and wrapper methods:
  - More accurate than filter methods (consider feature interactions)
  - More efficient than wrapper methods (model is trained only once, not per subset)
- Often use **regularization** to shrink or eliminate less important feature coefficients
- Result is a model that has inherently performed its own feature selection

---

## Linear Regression Assumptions

Embedded methods built on linear models rely on the following assumptions. Violating them can affect both the model and the quality of feature selection.

| # | Assumption | Description |
|---|-----------|-------------|
| 1 | **Linearity** | The relationship between independent and dependent variables is linear — the change in the output per unit change in input is constant |
| 2 | **Independence** | Observations are independent of each other — residuals (observed minus predicted) must be independent |
| 3 | **Homoscedasticity** | The variance of residuals is constant across all levels of the independent variables |
| 4 | **Normality** | Residuals are normally distributed |
| 5 | **No Multicollinearity** | Independent variables are not highly correlated with each other — especially critical when interpreting regression coefficients |

> **Note:** Regularization methods like Ridge are specifically designed to handle multicollinearity (Assumption 5), making them useful even when this assumption is violated.

---

## Techniques

### 1. Regularized Models

Regularized linear models include a **penalty term in the loss function** during training. This penalty discourages the model from learning overly complex relationships, helping prevent overfitting and performing implicit feature selection.

**General regularized loss function:**
```
Loss = RSS + λ · Penalty

Where:
  RSS = Residual Sum of Squares (standard linear regression loss)
  λ   = Regularization strength (hyperparameter to tune)
  Penalty = depends on the method (L1, L2, or both)
```

---

#### Lasso Regression (L1)

**Lasso** (Least Absolute Shrinkage and Selection Operator) adds an **L1 penalty** — the sum of the absolute values of the coefficients.

**Loss Function:**
```
Loss = RSS + λ · Σ|βᵢ|
```

**Key property:** Lasso can shrink coefficients **all the way to zero**, effectively removing features from the model entirely. This makes it a true embedded feature selection method.

**When to use:**
- You want a **sparse model** with only the most important features retained
- You want a **simple, interpretable** model
- You suspect only a **few features** are truly relevant

**Usage:**
```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Features with non-zero coefficients are selected
selector = SelectFromModel(lasso, prefit=True)
X_reduced = selector.transform(X_train)

# View selected features
import pandas as pd
coef_df = pd.Series(lasso.coef_, index=feature_names)
print(coef_df[coef_df != 0])  # Non-zero = selected
```

**Effect of λ (alpha):**
```
Low λ  → Less penalty → More features retained → Risk of overfitting
High λ → More penalty → Fewer features retained → Risk of underfitting
```

---

#### Ridge Regression (L2)

**Ridge** regression adds an **L2 penalty** — the sum of the squared values of the coefficients.

**Loss Function:**
```
Loss = RSS + λ · Σβᵢ²
```

**Key property:** Ridge **shrinks coefficients toward zero but never exactly to zero**. It does not perform feature selection per se, but it reduces the impact of less important features and handles **multicollinearity** effectively.

**When to use:**
- You have **multicollinearity** in your dataset (correlated features)
- You want to **minimize model complexity** without eliminating features
- You want to keep **all features** but reduce their influence

**Usage:**
```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Coefficients are shrunk but not zeroed out
coef_df = pd.Series(ridge.coef_, index=feature_names)
print(coef_df.sort_values(ascending=False))
```

**Lasso vs Ridge:**

| | Lasso (L1) | Ridge (L2) |
|---|---|---|
| **Penalty** | Σ\|βᵢ\| | Σβᵢ² |
| **Zeroes out features?** | Yes | No |
| **Feature selection?** | Yes (true selection) | No (shrinkage only) |
| **Best for** | Sparse models, few relevant features | Multicollinearity, all features matter |

---

#### Elastic Net

**Elastic Net** combines both L1 and L2 penalties, making it a hybrid of Lasso and Ridge.

**Loss Function:**
```
Loss = RSS + λ₁ · Σ|βᵢ| + λ₂ · Σβᵢ²
```

**Key property:** Particularly useful when there are **multiple correlated features** — Lasso tends to arbitrarily pick one from a group of correlated features and zero the rest, while Elastic Net tends to keep them all with shrinkage.

**When to use:**
- You have **multiple correlated features** and want balanced selection
- You want the **feature elimination of Lasso** with the **stability of Ridge**
- Your data has both sparse and dense groups of relevant features

**Usage:**
```python
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.01, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
enet.fit(X_train, y_train)
```

---

### 2. Tree-Based Models

Tree-based models provide **built-in feature importance scores** as a natural byproduct of their training process. These scores can be used to rank and select the most important features.

#### Random Forest Feature Importance

Random forests calculate feature importance using **Mean Decrease in Impurity (MDI)** — how much each feature reduces impurity (e.g., Gini impurity or entropy) across all trees in the forest.

**How it works:**
1. Train a Random Forest on the full feature set
2. For each feature, measure the total reduction in impurity it contributes across all trees
3. Normalize the scores so they sum to 1
4. Rank and select features above a chosen threshold

**Usage:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values(ascending=False, inplace=True)

# Plot
importances.plot(kind='bar', figsize=(12, 4))
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.show()

# Select features above mean importance
selector = SelectFromModel(rf, prefit=True)
X_reduced = selector.transform(X_train)
```

**Advantages of tree-based importance:**
- Captures **non-linear relationships** between features and target
- Handles **mixed data types** (numerical and categorical)
- Robust to **outliers** and does not require feature scaling
- Works for both **classification** and **regression**

---

### 3. Recursive Feature Elimination (RFE)

**RFE** is a hybrid method that sits between wrapper and embedded approaches. It recursively removes the least important features by repeatedly training a model and using its internal feature importance scores.

**How it works:**
1. Train the model on all features
2. Rank features by their importance (e.g., coefficients or feature importances)
3. Remove the least important feature(s)
4. Retrain the model on the remaining features
5. Repeat until the desired number of features remains

```
All Features
    ↓
Train model → rank features → remove weakest
    ↓
Train model → rank features → remove weakest
    ↓
... (repeat)
    ↓
k features remain → done
```

**Usage:**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_train, y_train)

print('Selected features:', [f for f, s in zip(feature_names, rfe.support_) if s])
print('Feature ranking:', rfe.ranking_)
```

**With Cross-Validation (RFECV) — recommended:**
```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=LogisticRegression(), cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

print('Optimal number of features:', rfecv.n_features_)
print('Selected features:', [f for f, s in zip(feature_names, rfecv.support_) if s])
```

**When to use RFE:**
- You want to use the model's own importance scores for selection
- You want a systematic, ranked elimination rather than threshold-based filtering
- You have enough computational budget for multiple model fits

---

## Advantages & Disadvantages

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Performance** | Generally more accurate than filter methods — takes feature interactions into account |
| **Efficiency** | More computationally efficient than wrapper methods — fits the model only once (or a small number of times) |
| **Less Prone to Overfitting** | Regularization (Lasso, Ridge) introduces a penalty that shrinks coefficients, reducing overfitting risk |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Model Specific** | Selected features are tied to the specific model used — may not be optimal for other model types |
| **Complexity** | Harder to interpret than filter methods — understanding why Lasso zeroes a coefficient can be non-trivial |
| **Tuning Required** | Hyperparameters (e.g., regularization strength `λ`) must be tuned, often via cross-validation |
| **Stability** | Small changes in data can yield different selected features — especially with complex models like decision trees |

---

## Complete Cheatsheet: All Feature Selection Methods

### 1. Filter Methods

| Method | Use When | Feature Type | Target Type |
|--------|----------|-------------|-------------|
| **Variance Threshold** | Remove constants/near-constants from many features | Numerical | Any |
| **Correlation Coefficient** | Suspect highly correlated features | Numerical | Numerical |
| **Chi-Square Test** | Find dependency between categorical feature and target | Categorical | Categorical |
| **Mutual Information** | Measure both linear AND non-linear dependencies | Any | Any |
| **ANOVA** | Categorical independent variables, continuous dependent variable | Numerical | Categorical |

### 2. Wrapper Methods

| Method | Use When | Notes |
|--------|----------|-------|
| **Recursive Feature Elimination (RFE)** | Want model to identify best features | Uses model's feature importance iteratively |
| **Sequential Feature Selection (SFS)** | Computational cost is not an issue, want optimal subset | Forward or Backward |
| **Exhaustive Feature Selection** | Small number of features, want absolute best subset | Computationally expensive: O(2ⁿ) |

### 3. Embedded Methods

| Method | Use When | Feature Selection? |
|--------|----------|-------------------|
| **Lasso (L1)** | Want sparse, interpretable model | Yes — zeroes out features |
| **Ridge (L2)** | Multicollinearity present, want coefficient shrinkage | No — shrinks but retains all |
| **Elastic Net** | Multiple correlated features, want balanced Lasso+Ridge | Yes — partial |
| **Random Forest Importance** | Want non-linear, robust feature importance scores | Yes — by threshold |

---

## Choosing the Right Method

```
START
  │
  ├─ Do you have very many features (1000+)?
  │     └─ YES → Start with Filter Methods first
  │
  ├─ Is speed / computational cost a concern?
  │     └─ YES → Use Filter or Embedded methods
  │     └─ NO  → Wrapper methods are fine
  │
  ├─ Do you need feature interactions captured?
  │     └─ YES → Use Wrapper or Embedded methods
  │     └─ NO  → Filter methods are sufficient
  │
  ├─ Do you need a sparse/interpretable model?
  │     └─ YES → Lasso or Elastic Net
  │
  ├─ Do you have multicollinearity?
  │     └─ YES → Ridge or Elastic Net
  │
  ├─ Do you want non-linear feature importance?
  │     └─ YES → Random Forest Importance
  │
  └─ Best general practice:
        Filter → reduce noise first
        Embedded → fine-tune with model-aware selection
        Wrapper → final optimization if budget allows
```

---

## Quick Reference

```
EMBEDDED FEATURE SELECTION CHEAT SHEET
========================================

REGULARIZED MODELS:
  Lasso (L1)
    Loss = RSS + λ·Σ|βᵢ|
    → Zeroes out features → true feature selection
    → Use: sparse models, few relevant features

  Ridge (L2)
    Loss = RSS + λ·Σβᵢ²
    → Shrinks but never zeroes → handles multicollinearity
    → Use: correlated features, keep all variables

  Elastic Net (L1 + L2)
    Loss = RSS + λ₁·Σ|βᵢ| + λ₂·Σβᵢ²
    → Hybrid of Lasso + Ridge
    → Use: multiple correlated features

TREE-BASED:
  Random Forest → feature_importances_ (MDI)
  → Non-linear, robust, no scaling needed

RFE:
  from sklearn.feature_selection import RFE, RFECV
  → Iteratively removes weakest features
  → Use RFECV to auto-select optimal k

KEY HYPERPARAMETER:
  λ (alpha): controls regularization strength
  → Tune via cross-validation (GridSearchCV, LassoCV)

SKLEARN HELPERS:
  SelectFromModel(estimator) → threshold-based selection
  RFE(estimator, n_features_to_select=k)
  RFECV(estimator, cv=5) → auto-selects k
```

---

## References

- Session Notes: *Session on Embedded-Based Feature Selection* (May 2023)
- Scikit-learn Documentation: `sklearn.linear_model`, `sklearn.feature_selection`, `sklearn.ensemble`
- Standard ML literature on regularization and tree-based feature importance
