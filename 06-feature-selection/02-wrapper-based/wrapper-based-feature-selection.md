# Wrapper-Based Feature Selection

A comprehensive guide to understanding and applying wrapper-based feature selection methods in machine learning pipelines.

---

## Table of Contents

- [Recap: Filter vs Wrapper vs Embedded](#recap-filter-vs-wrapper-vs-embedded)
- [What are Wrapper Methods?](#what-are-wrapper-methods)
- [How Wrapper Methods Work](#how-wrapper-methods-work)
- [Techniques](#techniques)
  - [1. Exhaustive Feature Selection (Best Subset Selection)](#1-exhaustive-feature-selection-best-subset-selection)
  - [2. Sequential Backward Selection (SBS)](#2-sequential-backward-selection-sbs)
  - [3. Sequential Forward Selection (SFS)](#3-sequential-forward-selection-sfs)
- [Advantages & Disadvantages](#advantages--disadvantages)
- [Filter vs Wrapper: Side-by-Side Comparison](#filter-vs-wrapper-side-by-side-comparison)
- [When to Use Wrapper Methods](#when-to-use-wrapper-methods)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## Recap: Filter vs Wrapper vs Embedded

| Type | How It Works | Model Required? | Speed | Risk of Overfitting |
|------|-------------|-----------------|-------|---------------------|
| **Filter** | Statistical scores per feature, independent of any model | No | Fast | Low |
| **Wrapper** | Train a model on feature subsets and evaluate performance | Yes | Slow | Higher |
| **Embedded** | Feature selection happens during model training | Yes (built-in) | Moderate | Low–Moderate |

> Wrapper methods sit between filter and embedded methods — they are more accurate than filter methods but more computationally expensive.

---

## What are Wrapper Methods?

Wrapper methods are a category of feature selection techniques that **use a predictive model** to score combinations of features. They are called "wrapper" methods because they **wrap** a model-based evaluation around the feature selection process.

Unlike filter methods (which assess each feature independently using statistics), wrapper methods:
- Evaluate **subsets** of features together
- Use **actual model performance** (e.g., accuracy, RMSE) as the selection criterion
- Account for **feature interactions** that filter methods miss
- Are **model-specific** — the selected subset is optimized for the model used during selection

---

## How Wrapper Methods Work

All wrapper methods follow the same general three-step loop:

```
┌─────────────────────────────────────────────┐
│                                             │
│   1. SUBSET GENERATION                      │
│      Generate a candidate subset of         │
│      features (add, remove, or random)      │
│                                             │
│              ↓                              │
│                                             │
│   2. SUBSET EVALUATION                      │
│      Train a model on the subset            │
│      Evaluate via cross-validation          │
│      Record the performance score           │
│                                             │
│              ↓                              │
│                                             │
│   3. STOPPING CRITERION                     │
│      Stop if:                               │
│      - Max iterations reached               │
│      - No improvement in N iterations       │
│      - Time limit exceeded                  │
│      - All subsets evaluated                │
│                                             │
│   Otherwise → go back to Step 1            │
└─────────────────────────────────────────────┘
```

**Key design choices:**
- **Subset generation strategy** → defines which type of wrapper method (exhaustive, forward, backward, etc.)
- **Model choice** → any ML model can be used (logistic regression, random forest, SVM, etc.)
- **Evaluation metric** → accuracy, F1, AUC-ROC for classification; RMSE, R² for regression
- **Cross-validation** → typically k-fold CV is used to get a reliable performance estimate

---

## Techniques

### 1. Exhaustive Feature Selection (Best Subset Selection)

Exhaustive Feature Selection evaluates **every possible combination** of features and selects the subset that produces the best model performance.

**How it works:**
- For `n` features, generate all `2ⁿ` possible subsets
- Train and evaluate the model on each subset
- Return the subset with the best performance score

**Example (n=3 features: A, B, C):**
```
Subsets evaluated:
{A}, {B}, {C},
{A,B}, {A,C}, {B,C},
{A,B,C}
→ Total: 2³ - 1 = 7 subsets
```

**Usage:**
```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
efs = ExhaustiveFeatureSelector(model, min_features=1, max_features=5,
                                 scoring='accuracy', cv=5)
efs.fit(X, y)
print('Best features:', efs.best_feature_names_)
```

**Disadvantages:**

| Limitation | Explanation |
|-----------|-------------|
| **Computational Complexity** | Number of subsets = 2ⁿ — grows exponentially with number of features. Impractical for large feature spaces |
| **Risk of Overfitting** | The best-performing subset on training data may not generalize well to unseen data |
| **Requires a Good Evaluation Metric** | Poor choice of metric leads to suboptimal feature selection regardless of thoroughness |

> **Rule of thumb:** Only practical for datasets with a small number of features (typically n ≤ 20).

---

### 2. Sequential Backward Selection (SBS)

Sequential Backward Selection (also called **Sequential Backward Elimination**) starts with **all features** and iteratively removes the least useful one at each step until a stopping criterion is met.

**How it works:**
1. Start with the full feature set (all `n` features)
2. Train the model and record performance
3. Try removing each feature one at a time
4. Remove the feature whose removal causes the **least drop** (or greatest improvement) in performance
5. Repeat until the desired number of features is reached or performance degrades

**Pseudocode:**
```
features = all features
while len(features) > k:
    scores = []
    for each feature f in features:
        score = cross_val_score(model, features - {f})
        scores.append(score)
    remove feature with highest score (least impact on removal)
```

**Usage:**
```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sbs = SequentialFeatureSelector(model, k_features=5, forward=False,
                                 scoring='accuracy', cv=5)
sbs.fit(X, y)
print('Selected features:', sbs.k_feature_names_)
```

**When to use:**
- You suspect most features are relevant and only a few are noise
- You want to start from a complete model and prune it
- Faster than exhaustive search: O(n²) evaluations vs O(2ⁿ)

---

### 3. Sequential Forward Selection (SFS)

Sequential Forward Selection starts with **no features** and iteratively adds the most useful one at each step until a stopping criterion is met. It is the greedy opposite of SBS.

**How it works:**
1. Start with an empty feature set
2. Try adding each remaining feature one at a time
3. Add the feature that **most improves** model performance
4. Repeat until the desired number of features is reached or performance stops improving

**Pseudocode:**
```
features = {}
while len(features) < k:
    scores = []
    for each remaining feature f:
        score = cross_val_score(model, features + {f})
        scores.append(score)
    add feature with the best score
```

**Usage:**
```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sfs = SequentialFeatureSelector(model, k_features=5, forward=True,
                                 scoring='accuracy', cv=5)
sfs.fit(X, y)
print('Selected features:', sfs.k_feature_names_)
```

**When to use:**
- You suspect only a small number of features are truly relevant
- You want to build up a minimal, high-performing feature set
- Faster than exhaustive search: O(n²) evaluations vs O(2ⁿ)

**SFS vs SBS at a glance:**

| | SFS (Forward) | SBS (Backward) |
|---|---|---|
| **Starts with** | Empty set | Full set |
| **At each step** | Adds best feature | Removes worst feature |
| **Better when** | Few features are relevant | Most features are relevant |
| **Risk** | May miss feature interactions early on | May retain redundant features longer |

---

## Advantages & Disadvantages

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Accuracy** | Generally yields the best-performing feature subset for a given algorithm — uses the model's own predictive power as the selection criterion |
| **Feature Interaction** | Evaluates subsets of features together, capturing synergistic relationships that filter methods miss |
| **Model-Aligned** | Selected features are optimized for the specific model you intend to use, leading to better final model performance |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Computational Complexity** | Generating and evaluating many feature subsets is expensive — especially with large feature counts |
| **Risk of Overfitting** | Optimizing for training performance can lead to a subset that doesn't generalize well to unseen data |
| **Model Specific** | The selected subset is tailored to the model used during selection — may not transfer well to a different model type |

---

## Filter vs Wrapper: Side-by-Side Comparison

| Aspect | Filter Methods | Wrapper Methods |
|--------|---------------|-----------------|
| **Feature evaluation** | Individual features independently | Subsets of features together |
| **Model involvement** | None | Required (trained at each step) |
| **Speed** | Fast | Slow |
| **Feature interactions** | Not considered | Considered |
| **Risk of overfitting** | Low | Higher |
| **Result quality** | Good baseline | Usually better accuracy |
| **Scalability** | High-dimensional datasets | Better for smaller feature sets |
| **Use case** | Quick pre-processing pass | Final feature tuning for a specific model |

---

## When to Use Wrapper Methods

✅ **Use wrapper methods when:**
- You have a **moderate number of features** (not thousands)
- You need the **best possible feature subset** for a specific model
- You have **computational resources** and time available
- You want to capture **feature interactions**
- You are doing a **final optimization pass** after filter methods have reduced the feature space

❌ **Avoid wrapper methods when:**
- You have **thousands of features** (exponential cost)
- You need a **quick, model-agnostic** feature selection step
- You are in the **exploratory phase** of your analysis
- **Interpretability** of the selection process matters

> **Best Practice:** Use filter methods first to eliminate obvious noise, then apply wrapper methods on the reduced feature set for fine-tuned selection.

---

## Quick Reference Cheat Sheet

```
WRAPPER-BASED FEATURE SELECTION CHEAT SHEET
=============================================

THREE-STEP LOOP:
  1. Generate subset
  2. Train model + cross-validate
  3. Check stopping criterion → repeat or stop

METHODS:
  Exhaustive (Best Subset)
    → Tries all 2ⁿ combinations
    → Best accuracy, worst scalability
    → Use only for n ≤ 20 features

  Sequential Forward Selection (SFS)
    → Starts empty, adds best feature each step
    → forward=True in SequentialFeatureSelector
    → Good when few features are relevant

  Sequential Backward Selection (SBS)
    → Starts full, removes worst feature each step
    → forward=False in SequentialFeatureSelector
    → Good when most features are relevant

COMPLEXITY:
  Exhaustive  → O(2ⁿ)   ← avoid for large n
  SFS / SBS   → O(n²)   ← feasible for moderate n

SKLEARN / MLXTEND:
  from mlxtend.feature_selection import SequentialFeatureSelector
  from mlxtend.feature_selection import ExhaustiveFeatureSelector

EVALUATION METRICS:
  Classification  → accuracy, f1, roc_auc
  Regression      → r2, neg_mean_squared_error

RECOMMENDED PIPELINE:
  Filter Methods (remove noise)
      → Wrapper Methods (optimize for model)
          → Final Model Training
```

---

## References

- Session Notes: *Session on Wrapper-Based Feature Selection* (May 2023)
- mlxtend Documentation: `mlxtend.feature_selection`
- Scikit-learn Feature Selection Documentation
- Standard ML literature on greedy feature selection strategies
