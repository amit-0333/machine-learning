# Filter-Based Feature Selection

A comprehensive guide to understanding and applying filter-based feature selection techniques in machine learning pipelines.

---

## Table of Contents

- [What is Feature Selection?](#what-is-feature-selection)
- [Types of Feature Selection](#types-of-feature-selection)
- [What is Filter-Based Feature Selection?](#what-is-filter-based-feature-selection)
- [Plan of Action](#plan-of-action)
- [Techniques](#techniques)
  - [1. Duplicate Features](#1-duplicate-features)
  - [2. Variance Threshold](#2-variance-threshold)
  - [3. Correlation](#3-correlation)
  - [4. ANOVA](#4-anova)
  - [5. Chi-Square Test](#5-chi-square-test)
- [Advantages & Disadvantages of Filter Methods](#advantages--disadvantages-of-filter-methods)
- [When to Use Which Technique](#when-to-use-which-technique)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## What is Feature Selection?

Feature selection is the process of selecting a subset of relevant features (variables, predictors) from the original feature set to use in model construction. It helps to:

- **Reduce overfitting** — fewer redundant features means less noise for the model to learn from
- **Improve accuracy** — removing irrelevant data reduces misleading signals
- **Speed up training** — fewer features means faster model training and inference
- **Improve interpretability** — simpler models are easier to understand and explain

---

## Types of Feature Selection

There are three broad categories of feature selection methods:

| Type | Description | Examples |
|------|-------------|---------|
| **Filter Methods** | Use statistical measures to score each feature independently, without involving a ML model | Variance Threshold, Correlation, ANOVA, Chi-Square |
| **Wrapper Methods** | Use a predictive model to evaluate feature subsets by training and testing different combinations | Forward Selection, Backward Elimination, RFE |
| **Embedded Methods** | Feature selection happens as part of the model training process itself | Lasso (L1 regularization), Ridge, Decision Trees |

---

## What is Filter-Based Feature Selection?

Filter-based feature selection techniques use **statistical measures** to score each feature independently, then select a subset of features based on these scores. They are called "filter" methods because they essentially **filter out** features that do not meet some criterion — before any machine learning model is trained.

Key characteristics:
- **Model-agnostic** — do not depend on any specific ML algorithm
- **Fast and scalable** — evaluate features without training a model
- **Univariate** — typically assess each feature individually against the target
- **Pre-processing step** — commonly used to clean and reduce the feature space before applying wrapper or embedded methods

---

## Plan of Action

A typical pipeline for filter-based feature selection:

```
Raw Dataset
    │
    ▼
1. Remove Duplicate Features
    │
    ▼
2. Remove Low-Variance Features (Variance Threshold)
    │
    ▼
3. Remove Highly Correlated Features (Correlation Analysis)
    │
    ▼
4. Select Features Based on Relationship with Target
    ├── Numerical Feature → Numerical Target  →  Correlation
    ├── Categorical Feature → Numerical Target →  ANOVA
    └── Categorical Feature → Categorical Target → Chi-Square
    │
    ▼
Final Selected Feature Set → Model Training
```

---

## Techniques

### 1. Duplicate Features

The first and simplest step — identify and remove features that are **exact copies** of other features in the dataset.

**Why it matters:**
- Duplicate features add no new information
- They inflate the feature space unnecessarily
- They can distort distance-based algorithms and correlation measures

**How to detect:**
```python
# Detect duplicate columns
duplicates = df.T.duplicated()
df = df.loc[:, ~duplicates]
```

**Things to watch for:**
- Features may be duplicates even if column names differ
- Near-duplicates (very high correlation) are handled in the correlation step

---

### 2. Variance Threshold

Variance Threshold removes features whose **variance falls below a specified threshold**. The idea is that features with very low variance carry little information — a feature that is almost constant across all samples is unlikely to be a useful predictor.

**Formula:**
```
Var(X) = (1/n) * Σ(Xᵢ - X̄)²
```

**Usage in sklearn:**
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_reduced = selector.fit_transform(X)
```

**Points to Consider:**

| Limitation | Explanation |
|-----------|-------------|
| **Ignores Target Variable** | Evaluates each feature independently — may keep irrelevant high-variance features or discard low-variance but predictive ones |
| **Ignores Feature Interactions** | A low-variance feature may become informative in combination with another feature |
| **Sensitive to Data Scaling** | Features on larger scales will naturally have higher variance — always **standardize** before applying |
| **Arbitrary Threshold** | No universal "correct" threshold — it is dataset and problem dependent |

> **Best Practice:** Always standardize (z-score or min-max scale) your features before applying Variance Threshold to avoid scale bias.

---

### 3. Correlation

Correlation measures the **linear relationship** between two variables. In feature selection, it is used in two ways:
1. **Feature–Target correlation** — keep features that are highly correlated with the target variable
2. **Feature–Feature correlation** — remove one of any pair of features that are highly correlated with each other (to reduce redundancy)

**Pearson Correlation Formula:**
```
r = Σ((Xᵢ - X̄)(Yᵢ - Ȳ)) / √[Σ(Xᵢ - X̄)² * Σ(Yᵢ - Ȳ)²]
```

**Interpretation:**
| Value | Strength |
|-------|----------|
| 0.0 – 0.2 | Very weak |
| 0.2 – 0.4 | Weak |
| 0.4 – 0.6 | Moderate |
| 0.6 – 0.8 | Strong |
| 0.8 – 1.0 | Very strong |

**Usage:**
```python
import pandas as pd

# Correlation with target
corr_with_target = df.corr()['target'].abs().sort_values(ascending=False)

# Correlation matrix for feature-feature relationships
corr_matrix = df.corr().abs()
```

**Disadvantages:**

| Limitation | Explanation |
|-----------|-------------|
| **Linearity Assumption** | Only captures linear relationships — non-linear relationships can be missed entirely |
| **Doesn't Capture Complex Relationships** | Only measures pairwise relationships between two variables at a time |
| **Threshold Determination** | Defining what counts as "high" correlation is subjective and context-dependent |
| **Sensitive to Outliers** | A few extreme values can significantly skew the correlation coefficient |

> **Use Case:** Best suited for **numerical features** and **numerical targets**. For categorical targets or features, consider ANOVA or Chi-Square instead.

---

### 4. ANOVA

**Analysis of Variance (ANOVA)** tests whether the **means of a numerical feature differ significantly across the categories of a categorical target variable**. It uses the F-statistic to determine if a feature is informative.

**When to use:** Numerical feature → Categorical target (classification problems)

**ANOVA Table:**

| Source of Variation | Sum of Squares (SS) | Degrees of Freedom | Mean Square (MS) | F-ratio |
|---|---|---|---|---|
| Between groups (categories) | SS_between = Σ nₖ(X̄ₖ - X̄)² | k - 1 | SS_between / (k-1) | MS_between / MS_within |
| Within groups | SS_within = Σ Σ (Xᵢⱼ - X̄ₖ)² | n - k | SS_within / (n-k) | — |
| Total | SS_total = Σ (Xᵢⱼ - X̄)² | n - 1 | — | — |

Where:
- `k` = number of categories
- `n` = total number of observations
- `X̄` = grand mean
- `X̄ₖ` = mean of group k

**The F-statistic → p-value:** A low p-value (typically < 0.05) indicates the feature significantly differs across target classes and is likely useful.

**Usage:**
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_reduced = selector.fit_transform(X, y)
```

**Disadvantages:**

| Limitation | Explanation |
|-----------|-------------|
| **Assumption of Normality** | Assumes data within each group follows a normal distribution — may not hold for skewed data |
| **Homogeneity of Variance** | Assumes equal variances across groups (homoscedasticity) — violations lead to incorrect results |
| **Independence of Observations** | Assumes observations are independent — problematic for time series or nested data |
| **Effect of Outliers** | A single outlier can significantly affect the F-statistic |
| **Doesn't Account for Interactions** | Like other univariate methods, ANOVA does not consider feature interactions |

> **Note:** ANOVA is essentially a **one-way hypothesis test**. It tells you *whether* a feature is significant, not *how* it relates to the target.

---

### 5. Chi-Square Test

The **Chi-Square (χ²) test** measures the **dependence between a categorical feature and a categorical target**. It tests whether the observed frequency distribution of two categorical variables differs from what would be expected if they were independent.

**When to use:** Categorical feature → Categorical target

**Chi-Square Formula:**
```
χ² = Σ [(Observed - Expected)² / Expected]
```

**Interpretation:**
- High χ² value → large difference between observed and expected → feature is likely dependent on the target → **keep the feature**
- Low χ² value → feature is likely independent of the target → **consider dropping it**

**Usage:**
```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2, k=10)
X_reduced = selector.fit_transform(X, y)
```

**Disadvantages:**

| Limitation | Explanation |
|-----------|-------------|
| **Categorical Data Only** | Cannot be used with continuous variables unless discretized — discretization can cause information loss |
| **Independence of Observations** | Assumes observations are independent — problematic for time series or clustered data |
| **Sufficient Sample Size** | Unreliable if expected frequency in any cell is < 5 |
| **No Variable Interactions** | Does not consider combinations of features — may miss features that are significant in combination |

> **Important:** Features must be **non-negative** for Chi-Square to work in sklearn. Ensure all values are ≥ 0 before applying.

---

## Advantages & Disadvantages of Filter Methods

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Simplicity** | Straightforward to understand — score each feature, select the top ones |
| **Speed** | Computationally efficient — no model needs to be trained |
| **Scalability** | Handles high-dimensional datasets well — suitable as a first pass on large feature spaces |
| **Pre-processing Step** | Can serve as a first pass before more expensive wrapper or embedded methods |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Lack of Feature Interaction** | Treats features individually — may miss features that are insignificant alone but powerful in combination |
| **Model Agnostic** | Selected features may not optimally serve the specific model you intend to use |
| **Statistical Measures Limitation** | Each statistical test has its own assumptions and limitations (e.g., correlation misses non-linear relationships) |
| **Threshold Determination** | Defining what counts as "low" variance or "high" correlation is subjective and varies by dataset |

---

## When to Use Which Technique

| Feature Type | Target Type | Recommended Method |
|---|---|---|
| Any | Any | Duplicate Feature Removal (always first) |
| Numerical | Any | Variance Threshold |
| Numerical | Numerical | Pearson Correlation |
| Numerical | Categorical | ANOVA (F-test) |
| Categorical | Categorical | Chi-Square Test |
| Numerical | Categorical | also Point Biserial Correlation |
| Mixed | Any | Combine methods in a pipeline |

---

## Quick Reference Cheat Sheet

```
FILTER-BASED FEATURE SELECTION CHEAT SHEET
============================================

STEP 1: Remove Duplicates
  → df.T.duplicated()

STEP 2: Variance Threshold (Numerical)
  → VarianceThreshold(threshold=0.1)
  → Standardize first!

STEP 3: Correlation (Numerical → Numerical)
  → df.corr() — drop if |r| > 0.8 between features
  → Keep features with high |r| with target

STEP 4a: ANOVA (Numerical feature → Categorical target)
  → SelectKBest(f_classif, k=K)
  → Low p-value = important feature

STEP 4b: Chi-Square (Categorical → Categorical)
  → SelectKBest(chi2, k=K)
  → High χ² = dependent on target = keep

THRESHOLDS (rules of thumb):
  Correlation: |r| > 0.8 → consider dropping one
  VIF: > 10 → multicollinearity concern
  ANOVA p-value: < 0.05 → significant feature
  Chi-Square p-value: < 0.05 → significant feature
  Condition No.: > 30 → multicollinearity warning
```

---

## References

- Session Notes: *Session on Filter-Based Feature Selection* (May 2023)
- Scikit-learn Feature Selection Documentation
- Standard ML literature on statistical feature selection methods
