# Multicollinearity in Machine Learning

A comprehensive guide to understanding, detecting, and handling multicollinearity in multiple regression models.

---

## Table of Contents

- [What is Multicollinearity?](#what-is-multicollinearity)
- [Inference vs Prediction](#inference-vs-prediction)
- [When is Multicollinearity Bad?](#when-is-multicollinearity-bad)
- [What Happens Mathematically?](#what-happens-mathematically)
- [Perfect Multicollinearity](#perfect-multicollinearity)
- [Types of Multicollinearity](#types-of-multicollinearity)
- [How to Detect Multicollinearity](#how-to-detect-multicollinearity)
- [How to Remove Multicollinearity](#how-to-remove-multicollinearity)

---

## What is Multicollinearity?

Multicollinearity is a statistical phenomenon that occurs when two or more independent variables in a multiple regression model are **highly correlated**. This strong linear relationship between predictors makes it difficult to isolate the individual effects of each variable on the dependent variable.

---

## Inference vs Prediction

Understanding whether your goal is **inference** or **prediction** matters when evaluating the impact of multicollinearity.

### Inference
- Focuses on understanding relationships between variables in a model
- Draws conclusions about the underlying population or process
- Involves hypothesis testing, confidence intervals, and significance testing
- Interpretability is a **key concern**
- Examples: Linear regression, logistic regression, ANOVA

### Prediction
- Focuses on making accurate forecasts for new, unseen data
- Aims to generalize patterns learned from training data
- Minimizes error metrics like MSE or cross-entropy loss
- Interpretability is **less critical**
- Examples: Decision trees, SVMs, neural networks, random forests, gradient boosting

> **Key Takeaway:** Multicollinearity is more problematic when your goal is **inference**, as it distorts the interpretation of individual coefficients. For pure prediction tasks, it may be less of a concern.

---

## When is Multicollinearity Bad?

Multicollinearity causes issues primarily in inferential models:

1. **Difficulty identifying the most important predictors** — High correlation between independent variables makes it hard to determine which variable has the most significant impact on the outcome.

2. **Inflated standard errors** — Larger standard errors for regression coefficients reduce statistical power, making it harder to find true relationships.

3. **Unstable and unreliable estimates** — Coefficients become sensitive to small changes in the data, making results difficult to interpret accurately.

---

## What Happens Mathematically?

In OLS regression, the coefficient vector is estimated as:

```
β = (XᵀX)⁻¹ Xᵀy
```

The variance of the estimates is:

```
Var(β) = SE(β)²
```

When multicollinearity is present:
- The matrix `XᵀX` becomes **near-singular** (ill-conditioned)
- This inflates `Var(β)`, leading to **high standard errors**
- Coefficients `β₀, β₁, β₂, ...` become **unstable** across different samples

### Example Model
Consider predicting `lpa` (salary) from `cgpa` and `iq`:

```
lpa = β₀ + β₁·cgpa + β₂·iq
```

Across 10 different samples of 100 observations each, the estimated values of `β₀, β₁, β₂` can vary widely — this instability is a direct consequence of multicollinearity inflating `SE(β)`.

### Practical Indicator — OLS Regression Output
Here's a real example using TV, Radio, and Newspaper ad spend to predict Sales:

```
OLS Regression Results
==============================================================
Dep. Variable:       Sales       R-squared:          0.897
Model:               OLS         Adj. R-squared:     0.896
Method:        Least Squares     F-statistic:        570.3
No. Observations:      200       Condition No.       454
==============================================================
             coef    std err       t      P>|t|
--------------------------------------------------------------
const       2.9389    0.312     9.422    0.000
TV          0.0458    0.001    32.809    0.000
Radio       0.1885    0.009    21.893    0.000
Newspaper  -0.0010    0.006    -0.177    0.860
==============================================================
```

Key observations:
- **Condition No. = 454** → far above the threshold of 30, a strong signal of multicollinearity
- **Newspaper p-value = 0.860** → statistically insignificant despite the model's high R² (0.897)
- **SE values are inflated** → making individual coefficient interpretation unreliable

---

## Perfect Multicollinearity

**Perfect multicollinearity** occurs when one independent variable is an *exact* linear combination of one or more other independent variables.

- Makes it **impossible** to uniquely estimate individual effects
- The matrix `XᵀX` becomes singular and **non-invertible**
- OLS estimation breaks down entirely

Example: Including both `temperature_celsius` and `temperature_fahrenheit` in the same model.

---

## Types of Multicollinearity

### 1. Structural Multicollinearity
Arises from how variables are **defined or constructed**:
- One variable is created as a linear combination of others
- Model includes interaction terms or polynomial terms without proper **scaling or centering**

### 2. Data-Driven Multicollinearity
Arises from **patterns in the observed data**:
- Independent variables happen to be highly correlated in the sample
- Not caused by model construction — purely a data characteristic

---

## How to Detect Multicollinearity

### Correlation Matrix
- Compute pairwise correlations between all predictor variables
- Off-diagonal values with **|r| > 0.8 or 0.9** suggest problematic correlation
- Limitation: Only captures pairwise relationships, not multivariate collinearity

### Variance Inflation Factor (VIF)
VIF measures how much the variance of a coefficient is inflated due to collinearity.

**Steps to calculate VIF for predictor Xᵢ:**
1. Regress Xᵢ on all other predictors
2. Calculate R² for that regression
3. Compute: `VIFᵢ = 1 / (1 - R²ᵢ)`

**Interpretation:**
| VIF Value | Interpretation |
|-----------|----------------|
| ~1 | No multicollinearity |
| 1–5 | Moderate (generally acceptable) |
| > 5 | High — investigate further |
| > 10 | Severe — action likely needed |

### Condition Number
- Calculated as the ratio of the **largest to smallest eigenvalue** of `XᵀX`
- A condition number **> 30** is a warning sign of multicollinearity
- High condition number → ill-conditioned matrix → unstable estimates

> **Note:** Use multiple diagnostics together. No single metric gives the full picture.

---

## How to Remove Multicollinearity

| Method | Description |
|--------|-------------|
| **Collect more data** | A larger sample can naturally reduce multicollinearity |
| **Remove a correlated variable** | Drop the variable with the highest VIF or least domain relevance |
| **Combine correlated variables** | Average, sum, or otherwise merge similar variables into one |
| **Use PLS Regression** | Partial Least Squares finds latent variables that maximize covariance with the response |
| **Apply Regularization** | Ridge regression penalizes large coefficients, reducing the impact of collinearity |
| **Dimensionality Reduction** | PCA transforms correlated features into uncorrelated principal components |

---

## References

- Session Notes: *Session on Multicollinearity* (May 2023)
- Statsmodels OLS Regression
- Standard ML literature on regularization and feature engineering
