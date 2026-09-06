# Regression Analysis — Complete Study Notes

> Full session notes — Statistics, OLS, R², F-test, T-test, Confidence Intervals & more

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | What is Regression Analysis? |
| 2 | Statistics Connection — Why ML is a Statistical Inference Problem |
| 3 | Inference vs Prediction |
| 4 | Statsmodel Linear Regression |
| 5 | TSS, RSS and ESS |
| 6 | Degree of Freedom |
| 7 | F-statistic & Prob(F-statistic) |
| 8 | R-squared |
| 9 | Adjusted R-squared |
| 10 | R² vs Adjusted R² — Which to use? |
| 11 | T-statistic |
| 12 | Confidence Intervals for Coefficients |
| 13 | Other OLS Output Metrics |

---

## 1. What is Regression Analysis?

Regression analysis is a statistical method used to examine the relationship between one **dependent variable** and one or more **independent variables**.

**Goal:** Understand how the dependent variable changes when independent variables are altered, and create a model that can **predict** the dependent variable.

### Steps in Regression Analysis

| Step | Action |
|------|--------|
| 1 | **Define the research question** — identify dependent & independent variables |
| 2 | **Collect and prepare data** — clean, handle missing values, outliers |
| 3 | **Visualize the data** — scatter plots to see trends and relationships |
| 4 | **Check assumptions** — linearity, independence, homoscedasticity, normality |
| 5 | **Fit the model** — estimate coefficients that minimize sum of squared residuals |
| 6 | **Interpret the model** — analyse coefficients, p-values, R² |
| 7 | **Validate the model** — train/test split, compute MSE/RMSE |
| 8 | **Report results** — summarize findings, limitations, assumptions |

---

## 2. Statistics Connection — Why ML is a Statistical Inference Problem

Machine Learning problems are fundamentally **Statistical Inference Problems**.

- We have features `X` and a target `y`
- We ask: **Is there a relationship between X and y?**
- If yes → is it **linear**? Is it **strong**?

```
X → y    →   Is there a relationship?
              ↓
         Is it linear?
              ↓
         Is it strong?
              ↓
           [Code]
```

For multiple features:

```
X = [x₁, x₂, x₃, ...]  →  y

Model: y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ...
```

Where `β₀` is the intercept and `β₁, β₂, ...` are the slopes (coefficients).

---

## 3. Inference vs Prediction

**Why do we need Regression Analysis?**

| | Inference | Prediction |
|-|-----------|------------|
| **Goal** | Understand relationship between X and y | Predict y for new X |
| **Focus** | Coefficients, p-values, significance | Accuracy, generalization |
| **Question** | Does TV spending affect Sales? By how much? | Given TV=100, what will Sales be? |
| **Tool** | Statsmodels (OLS) | Sklearn |

> Regression bridges both — it gives you coefficients to understand the relationship AND a formula to predict.

---

## 4. Statsmodel Linear Regression

### Model

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ
```

### Code

```python
import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_csv('advertising.csv')

# Fit OLS model
model = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()
print(model.summary())
```

### Reading the OLS Output

```
OLS Regression Results
===========================================================
Dep. Variable:       Sales     R-squared:          0.897
Model:               OLS       Adj. R-squared:     0.896
Method:    Least Squares       F-statistic:        570.3
No. Observations:    200       Prob (F-statistic): 1.58e-96
Df Residuals:        196       Log-Likelihood:    -386.18
Df Model:              3       AIC:                780.4
Covariance Type:  nonrobust    BIC:                793.6
===========================================================
             coef   std err       t    P>|t|  [0.025  0.975]
-----------------------------------------------------------
const       2.9389    0.312   9.422    0.000   2.324   3.554
TV          0.0458    0.001  32.809    0.000   0.043   0.049
Radio       0.1885    0.009  21.893    0.000   0.172   0.206
Newspaper  -0.0010    0.006  -0.177    0.860  -0.013   0.011
===========================================================
Omnibus:       60.414    Durbin-Watson:      2.084
Prob(Omnibus):  0.000    Jarque-Bera (JB):  151.241
Skew:          -1.327    Prob(JB):           1.44e-33
Kurtosis:       6.332    Cond. No.           454.
```

### What each part means

**Top section — Model info:**

| Field | Meaning |
|-------|---------|
| `Dep. Variable` | The target we're predicting (Sales) |
| `Model: OLS` | Ordinary Least Squares method |
| `No. Observations` | 200 data points |
| `Df Residuals` | 196 = n - (k+1) = 200 - 4 |
| `Df Model` | 3 = number of features (TV, Radio, Newspaper) |

**Coefficients table:**

| Field | Meaning |
|-------|---------|
| `coef` | The estimated β — how much y changes per unit of X |
| `std err` | Standard error of the coefficient |
| `t` | t-statistic = coef / std err |
| `P>\|t\|` | p-value — if < 0.05, the coefficient is significant |
| `[0.025, 0.975]` | 95% Confidence Interval for the coefficient |

**Interpretation of the example:**
- `TV coef = 0.0458` → for every $1 increase in TV budget, Sales increase by ~0.046 units
- `Newspaper p = 0.860` → NOT significant (p > 0.05), Newspaper has no meaningful effect
- `TV, Radio p = 0.000` → highly significant

---

## 5. TSS, RSS and ESS

These three quantities partition the total variance in the data:

```
TSS = ESS + RSS
```

| Metric | Full Name | Formula | Meaning |
|--------|-----------|---------|---------|
| **TSS** | Total Sum of Squares | `Σ(yᵢ - ȳ)²` | Total variance in y |
| **ESS** | Explained Sum of Squares | `Σ(ŷᵢ - ȳ)²` | Variance explained by the model |
| **RSS** | Residual Sum of Squares | `Σ(yᵢ - ŷᵢ)²` | Variance NOT explained (errors) |

**Visually:**

```
Total Variation (TSS)
├── Explained by model (ESS)   ← we want this to be large
└── Unexplained / Error (RSS)  ← we want this to be small
```

```python
import numpy as np

y_mean = np.mean(y)
TSS = np.sum((y - y_mean) ** 2)
ESS = np.sum((y_hat - y_mean) ** 2)
RSS = np.sum((y - y_hat) ** 2)

print(f"TSS = {TSS:.2f}")
print(f"ESS = {ESS:.2f}")
print(f"RSS = {RSS:.2f}")
print(f"Check: ESS + RSS = {ESS + RSS:.2f}")  # should equal TSS
```

---

## 6. Degree of Freedom

Total degrees of freedom are split between the model and residuals:

```
df_total   = n - 1
df_model   = k               (number of independent variables)
df_residuals = n - (k + 1)   (n minus number of parameters estimated)
```

**Check:**

```
df_total = df_model + df_residuals
(n - 1)  = k + (n - k - 1)
```

**Example from OLS output:**
```
n = 200, k = 3 (TV, Radio, Newspaper)

df_total     = 200 - 1 = 199
df_model     = 3
df_residuals = 200 - (3+1) = 196   ← matches "Df Residuals: 196"
```

---

## 7. F-statistic & Prob(F-statistic)

### What it tests

The F-test checks whether the **overall regression model is significant**:

```
H₀: β₁ = β₂ = ... = βₖ = 0  (no independent variable helps)
H₁: At least one βⱼ ≠ 0      (at least one variable matters)
```

### Steps

**1. Calculate Sum of Squares**
```
TSS = Σ(yᵢ - ȳ)²
ESS = Σ(ŷᵢ - ȳ)²
RSS = Σ(yᵢ - ŷᵢ)²
```

**2. Compute Mean Squares**
```
MSR (Mean Square Regression) = ESS / df_model
                              = ESS / k
                              → Average explained variance per feature

MSE (Mean Square Error)       = RSS / df_residuals
                              = RSS / (n - k - 1)
                              → Average unexplained variance per df
```

**3. F-statistic**
```
F = MSR / MSE
```

**4. Decision**
```
If Prob(F-statistic) < 0.05  → Reject H₀ → Model is significant ✅
If Prob(F-statistic) ≥ 0.05  → Fail to reject H₀ → Model is not useful
```

**From the example:**
```
F-statistic       = 570.3
Prob(F-statistic) = 1.58e-96   ← essentially 0, model is highly significant
```

```python
from scipy import stats

MSR = ESS / df_model
MSE = RSS / df_residuals
F   = MSR / MSE

p_value = 1 - stats.f.cdf(F, df_model, df_residuals)
print(f"F = {F:.3f}, p-value = {p_value:.4e}")
```

---

## 8. R-squared (R²)

R² measures the **goodness of fit** — what proportion of the variance in y is explained by the model.

```
R² = ESS / TSS = 1 - (RSS / TSS)
```

| R² value | Interpretation |
|----------|---------------|
| 0 | Model explains nothing |
| 1 | Model explains everything perfectly |
| 0.897 (example) | Model explains 89.7% of variance in Sales |

**Important:** R² always increases or stays the same when you add more variables — even useless ones. That's why we need Adjusted R².

```python
R_squared = ESS / TSS
print(f"R² = {R_squared:.4f}")
```

---

## 9. Adjusted R-squared

Adjusted R² penalizes the model for adding unnecessary variables:

```
Adjusted R² = 1 - [ (1 - R²) * (n - 1) / (n - k - 1) ]
```

Where:
- `n` = number of observations
- `k` = number of predictor variables

**Key behaviour:**
- Adding a **useful** variable → Adjusted R² increases
- Adding a **useless** variable → Adjusted R² decreases or stays same
- Always ≤ R²

**From the example:**
```
R²          = 0.897
Adjusted R² = 0.896   ← very close, meaning all 3 features are useful
```

```python
n = X.shape[0]
k = X.shape[1]
adj_r2 = 1 - (1 - R_squared) * (n - 1) / (n - k - 1)
print(f"Adjusted R² = {adj_r2:.4f}")
```

---

## 10. R² vs Adjusted R² — Which to use?

| Situation | Use |
|-----------|-----|
| Comparing models with **different numbers of features** | **Adjusted R²** |
| Understanding overall explanatory power of one model | R² |
| Guarding against **overfitting** | **Adjusted R²** |
| Feature selection | **Adjusted R²** |

> **Rule of thumb:** When in doubt, report both. If R² is high but Adjusted R² is much lower, you likely have irrelevant features dragging the model down.

---

## 11. T-statistic

The T-test checks whether **each individual coefficient** is significant:

```
For slope β₁:
  H₀: β₁ = 0  (no relationship between X₁ and y)
  H₁: β₁ ≠ 0  (a relationship exists)
```

### Steps

**1. Estimate coefficients** `b₀, b₁` from data

**2. Calculate Standard Error (SE)**

The standard error of a coefficient measures how much it varies across samples.
From the output: `SE(TV) = 0.001`

**3. Compute T-statistic**

```
t = coef / SE(coef)

Example: t(TV) = 0.0458 / 0.001 = 32.809
```

**4. Get p-value** from t-distribution with `df = n - 2`

**5. Decision**

```
If p-value < 0.05  → coefficient is significant ✅
If p-value ≥ 0.05  → coefficient is NOT significant ❌
```

**From the example:**

| Variable | coef | SE | t | p-value | Significant? |
|----------|------|----|---|---------|-------------|
| const | 2.9389 | 0.312 | 9.422 | 0.000 | ✅ Yes |
| TV | 0.0458 | 0.001 | 32.809 | 0.000 | ✅ Yes |
| Radio | 0.1885 | 0.009 | 21.893 | 0.000 | ✅ Yes |
| Newspaper | -0.0010 | 0.006 | -0.177 | 0.860 | ❌ No |

> Newspaper has p = 0.860 >> 0.05 → not significant → can be dropped from the model

---

## 12. Confidence Intervals for Coefficients

A confidence interval gives the **range** within which the true coefficient likely falls.

```
CI = coef ± (t_critical × SE)

For 95% CI: df = n - 2
```

**Example — TV coefficient:**

```
b₁ = 0.0458,  SE = 0.001

From OLS output [0.025, 0.975] = [0.043, 0.049]

So: 0.0458 ± (3.18 × 0.001)  →  [0.043, 0.049]
```

This means: we are 95% confident the true coefficient for TV is between **0.043 and 0.049**.

**Interpretation:**
- If CI **does not contain 0** → variable is significant (TV, Radio ✅)
- If CI **contains 0** → variable may not matter (Newspaper: [-0.013, 0.011] contains 0 ❌)

```python
from scipy import stats

alpha = 0.05
df = n - 2
t_critical = stats.t.ppf(1 - alpha/2, df)

lower = coef - t_critical * SE
upper = coef + t_critical * SE
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## 13. Other OLS Output Metrics

| Metric | Value (example) | Meaning |
|--------|----------------|---------|
| **Log-Likelihood** | -386.18 | Higher (less negative) = better fit. Used to compute AIC/BIC |
| **AIC** | 780.4 | Akaike Info Criterion — lower is better. Penalizes model complexity |
| **BIC** | 793.6 | Bayesian Info Criterion — lower is better. Penalizes more than AIC |
| **Omnibus / Prob(Omnibus)** | 60.414 / 0.000 | Tests normality of residuals. Prob < 0.05 → residuals not normal |
| **Durbin-Watson** | 2.084 | Tests autocorrelation. Value near 2 → no autocorrelation ✅ |
| **Jarque-Bera (JB)** | 151.241 | Another normality test on residuals |
| **Prob(JB)** | 1.44e-33 | Very small → residuals are NOT normally distributed |
| **Skew** | -1.327 | Skewness of residuals. 0 = perfectly symmetric |
| **Kurtosis** | 6.332 | Peakedness of residuals. Normal = 3 |
| **Cond. No.** | 454 | Condition Number — checks multicollinearity. >30 = potential issue |

### AIC vs BIC — Model Comparison

```
Use AIC/BIC to compare models:
- Lower AIC/BIC = better model
- BIC penalizes complexity more than AIC
- If AIC says "add this variable" but BIC says "don't" → trust BIC for parsimony
```

---

## Full Workflow — Regression Analysis in Python

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load data
df = pd.read_csv('advertising.csv')

# 2. Visualize
pd.plotting.scatter_matrix(df, figsize=(10, 8))
plt.show()

# 3. Fit OLS
model = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()
print(model.summary())

# 4. Extract metrics
print(f"R²           : {model.rsquared:.4f}")
print(f"Adjusted R²  : {model.rsquared_adj:.4f}")
print(f"F-statistic  : {model.fvalue:.4f}")
print(f"Prob(F)      : {model.f_pvalue:.4e}")

# 5. Coefficients and p-values
print(model.params)
print(model.pvalues)
print(model.conf_int())   # Confidence intervals

# 6. Residual diagnostics
residuals = model.resid
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(model.fittedvalues, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residuals vs Fitted")

plt.subplot(1, 2, 2)
stats.probplot(residuals, plot=plt)
plt.title("Q-Q Plot of Residuals")

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

- **F-test** → is the overall model significant? (tests all β together)
- **T-test** → is each individual coefficient significant?
- **R²** → proportion of variance explained (always increases with more features)
- **Adjusted R²** → penalizes for adding useless features — use this for comparison
- **CI not containing 0** → coefficient is statistically significant
- **Newspaper p = 0.860** → not significant, should be dropped
- **Durbin-Watson ≈ 2** → no autocorrelation in residuals ✅

---

## Resources

- [Statsmodels OLS Docs](https://www.statsmodels.org/stable/regression.html)
- [CS229 - Andrew Ng Regression Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest - Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [StatQuest - R² explained](https://www.youtube.com/watch?v=2AQKmw14mHM)

---

*Learning in public. Mistakes are part of the process.*
