# Assumptions of Linear Regression — Complete Study Notes

> Full session notes — All 5 assumptions, how to check them, what to do when they fail

---

## Recap — Where we are

From the last session (Regression Analysis), we covered:

```
Linear Regression
      ↓
Coefficients (OLS vs GD)
      ↓
Regression Analysis (R², F-test, T-test, CI)
      ↓
Assumptions of Linear Regression   ← THIS SESSION
      ↓
Multicollinearity
```

The OLS output gives us clues about assumptions:
- **Jarque-Bera / Omnibus** → test of normality of residuals
- **Durbin-Watson** → test for autocorrelation
- **Cond. No.** → hints at multicollinearity

---

## The 5 Assumptions of Linear Regression

| # | Assumption | Violated by | Key Test |
|---|-----------|-------------|----------|
| 1 | Linearity | Non-linear relationships | Residual plot, scatter plot |
| 2 | Normality of Residuals | Skewed/heavy-tailed errors | Q-Q plot, Omnibus, Jarque-Bera |
| 3 | Homoscedasticity | Unequal error variance | Residual plot, Breusch-Pagan |
| 4 | No Autocorrelation | Correlated error terms | Durbin-Watson test |
| 5 | No Multicollinearity | Correlated features | VIF, Cond. No. |

---

## 1. Linearity

### The Assumption

There is a **linear relationship** between the independent variables and the dependent variable. Changes in X lead to proportional changes in y.

```
y = β₀ + β₁x₁ + β₂x₂ + ε
```

### What happens when violated?

| Problem | Effect |
|---------|--------|
| Biased parameter estimates | Coefficients are wrong → incorrect inferences |
| Reduced predictive accuracy | Model underfits, misses patterns |
| Invalid hypothesis tests & CIs | t-tests and F-tests give misleading results |

### How to check

**1. Scatter plots**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot y vs each feature
for col in ['TV', 'Radio', 'Newspaper']:
    plt.figure()
    sns.scatterplot(x=df[col], y=df['Sales'])
    plt.title(f"Sales vs {col}")
    plt.show()
```

**2. Residual plots** — residuals should be randomly scattered around 0
```python
residuals = model.resid
fitted    = model.fittedvalues

plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted — check for linearity")
plt.show()
```

**3. Add polynomial terms and compare fit**
```python
import statsmodels.formula.api as smf

model_linear = smf.ols('Sales ~ TV', data=df).fit()
model_poly   = smf.ols('Sales ~ TV + I(TV**2)', data=df).fit()

print("Linear AIC:", model_linear.aic)
print("Poly   AIC:", model_poly.aic)
# If poly AIC is much lower → linearity assumption may be violated
```

### What to do when it fails?

| Fix | How |
|-----|-----|
| **Transformations** | `log(y)`, `sqrt(X)`, `1/X` to linearize |
| **Polynomial regression** | Add `X²`, `X³` terms to model |
| **Piecewise regression** | Fit separate linear models per segment |
| **GAMs / Splines** | Non-parametric methods that don't assume linearity |

---

## 2. Normality of Residuals

### The Assumption

The error terms (residuals) follow a **normal distribution** with mean 0 and constant variance:

```
ε ~ N(0, σ²)
```

### What happens when violated?

| Problem | Effect |
|---------|--------|
| Inaccurate hypothesis tests | t-tests and F-tests produce wrong results |
| Invalid confidence intervals | CIs are unreliable |
| Reduced model performance | Model may not be the right fit |

> **Note:** With large sample sizes, the Central Limit Theorem kicks in and this assumption becomes less critical.

### How to check

**1. Histogram of residuals**
```python
plt.hist(model.resid, bins=30, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.show()
# Should look like a bell curve centred at 0
```

**2. Q-Q plot** — points should fall on a straight diagonal line
```python
from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(model.resid, plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()
```

**3. Statistical tests**
```python
from scipy.stats import shapiro, jarque_bera, normaltest

# Shapiro-Wilk (best for small samples, n < 5000)
stat, p = shapiro(model.resid)
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}")

# Jarque-Bera (already in statsmodels OLS output)
stat, p = jarque_bera(model.resid)
print(f"Jarque-Bera:  stat={stat:.4f}, p={p:.4f}")

# Omnibus (already in OLS output)
stat, p = normaltest(model.resid)
print(f"Omnibus:      stat={stat:.4f}, p={p:.4f}")

# For ALL three: p > 0.05 → residuals are normal ✅
#               p < 0.05 → residuals are NOT normal ❌
```

**From the OLS output (Advertising example):**
```
Omnibus:       60.414    Prob(Omnibus): 0.000   ← p < 0.05 → NOT normal ❌
Jarque-Bera:  151.241    Prob(JB):    1.44e-33  ← p < 0.05 → NOT normal ❌
Skew:          -1.327                           ← negatively skewed
Kurtosis:       6.332                           ← heavy tails (normal = 3)
```

### Omnibus Test — Deep Dive

The Omnibus test checks normality using **skewness and kurtosis** of residuals:

```
H₀: Residuals are normally distributed
H₁: Residuals are NOT normally distributed
```

**Steps:**
1. Fit the model, get residuals
2. Compute **skewness** (asymmetry — should be ≈ 0)
3. Compute **kurtosis** (tailedness — should be ≈ 0 in excess terms)
4. Calculate test statistic K²:

```
K² = n/6 * (S² + (K-3)²/4)

Where:
S = skewness of residuals
K = kurtosis of residuals
n = number of observations
```

5. K² follows a **chi-squared distribution with 2 degrees of freedom**
6. If p-value < 0.05 → reject H₀ → residuals not normal

```python
from scipy.stats import normaltest

stat, p = normaltest(model.resid)
print(f"K² = {stat:.4f}, p-value = {p:.4f}")
if p < 0.05:
    print("Residuals are NOT normally distributed ❌")
else:
    print("Residuals appear normally distributed ✅")
```

### What to do when it fails?

| Fix | How |
|-----|-----|
| **Transformations** | `log(y)`, `sqrt(y)` to normalize the target |
| **Robust regression** | M-estimation, Least Median of Squares (LMS), Least Trimmed Squares (LTS) |
| **Bootstrapping** | Bootstrap CIs don't rely on normality |
| **GAMs / Splines** | Non-parametric — no normality assumption |
| **Model selection** | Use AIC/BIC cross-validation to find better model |

---

## 3. Homoscedasticity

### The Assumption

The **spread (variance) of residuals is constant** across all levels of X. When this fails, we have **heteroscedasticity** — the residual spread fans out or changes with X.

```
Homoscedastic:    Var(ε) = σ²       (constant)
Heteroscedastic:  Var(ε) = f(X)     (changes with X)
```

### What happens when violated?

| Problem | Effect |
|---------|--------|
| Inefficient estimates | Coefficients still unbiased but no longer BLUE (Best Linear Unbiased Estimator) |
| Inaccurate hypothesis tests | t-tests and F-tests misleading |
| Invalid confidence intervals | SEs are wrong → CIs are wrong |

### How to check

**1. Residual plot** — look for a funnel/cone shape
```python
plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted — check for homoscedasticity")
plt.show()

# Homoscedastic: random band around 0
# Heteroscedastic: funnel shape — spread increases with fitted values
```

**2. Breusch-Pagan Test** — formal statistical test

```
H₀: Variance of residuals is constant (homoscedastic)
H₁: Variance of residuals depends on X (heteroscedastic)
```

**Steps:**
1. Fit original OLS model, get residuals `ε`
2. Compute squared residuals `ε²`
3. Regress `ε²` on the same X variables → get R²
4. Calculate LM statistic:

```
LM = n × R²
```

5. LM follows a **chi-squared distribution with k degrees of freedom** (k = number of features)
6. If p-value < 0.05 → heteroscedasticity present

```python
from statsmodels.stats.diagnostic import het_breuschpagan

lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)

print(f"LM Statistic : {lm_stat:.4f}")
print(f"LM p-value   : {lm_pvalue:.4f}")

if lm_pvalue < 0.05:
    print("Heteroscedasticity detected ❌")
else:
    print("Homoscedasticity holds ✅")
```

### What to do when it fails?

| Fix | How |
|-----|-----|
| **Transformations** | `log(y)` often stabilizes variance |
| **Weighted Least Squares (WLS)** | Assign weights inversely proportional to variance |
| **Robust standard errors** | Use HC (heteroscedasticity-consistent) SEs |

```python
# Robust standard errors in statsmodels
model_robust = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit(cov_type='HC3')
print(model_robust.summary())
```

---

## 4. No Autocorrelation

### The Assumption

There should be **no correlation between the residuals**. Each error term should be independent of others.

```
Corr(εᵢ, εⱼ) = 0  for all i ≠ j
```

Most common in **time series data** where consecutive observations are naturally related.

### What happens when violated?

| Problem | Effect |
|---------|--------|
| Inefficient estimates | Still unbiased but not BLUE |
| Inaccurate hypothesis tests | t-tests and F-tests misleading |
| Invalid confidence intervals | CIs unreliable |

### How to check

**Durbin-Watson Test** (already in OLS output)

```
DW statistic ranges from 0 to 4:

DW ≈ 2   → No autocorrelation ✅
DW < 2   → Positive autocorrelation (adjacent residuals move together)
DW > 2   → Negative autocorrelation (adjacent residuals move oppositely)

Rule of thumb: 1.5 < DW < 2.5 is generally acceptable
```

**From the Advertising example:**
```
Durbin-Watson: 2.084   → No autocorrelation ✅
```

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(model.resid)
print(f"Durbin-Watson: {dw:.4f}")

if 1.5 < dw < 2.5:
    print("No significant autocorrelation ✅")
elif dw < 1.5:
    print("Positive autocorrelation detected ❌")
else:
    print("Negative autocorrelation detected ❌")
```

> **Note:** Durbin-Watson only detects **first-order** autocorrelation. For higher orders, use the Ljung-Box test.

### What to do when it fails?

| Fix | How |
|-----|-----|
| **Lagged variables** | Add `y(t-1)`, `X(t-1)` as predictors |
| **Differencing** | Use changes `Δy = y(t) - y(t-1)` instead of raw values |
| **Generalized Least Squares (GLS)** | Accounts for autocorrelation structure in errors |
| **Time series models** | AR, MA, ARIMA, STL for time-dependent data |
| **Newey-West standard errors** | HAC (heteroscedasticity and autocorrelation consistent) SEs |

```python
# Newey-West HAC standard errors
model_hac = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit(cov_type='HAC',
                                                                     cov_kwds={'maxlags': 1})
print(model_hac.summary())
```

---

## 5. No Multicollinearity

### The Assumption

The independent variables should **not be highly correlated with each other**. Multicollinearity occurs when features are strongly linearly related.

```
Example:
  X₁ = Height in cm
  X₂ = Height in inches   ← perfectly correlated with X₁ → perfect multicollinearity
```

### What happens when violated?

| Problem | Effect |
|---------|--------|
| Unstable coefficients | Small changes in data → large changes in β |
| Inflated standard errors | t-statistics drop → p-values rise → variables appear insignificant |
| Difficult interpretation | Can't isolate individual effect of each feature |
| Singular matrix | In extreme cases, OLS cannot be computed (X'X is not invertible) |

### How to check

**1. Correlation matrix**
```python
import seaborn as sns

corr = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# High correlation between features (|r| > 0.8) → multicollinearity concern
```

**2. Variance Inflation Factor (VIF)**

VIF measures how much the variance of a coefficient is inflated due to multicollinearity:

```
VIF = 1 / (1 - R²ⱼ)

Where R²ⱼ = R² from regressing feature j on all other features

VIF = 1       → No multicollinearity
VIF = 1–5     → Acceptable
VIF > 5–10    → Moderate multicollinearity ⚠️
VIF > 10      → Severe multicollinearity ❌
```

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = df[['TV', 'Radio', 'Newspaper']]
X['const'] = 1

vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif_data)
```

**3. Condition Number (from OLS output)**

```
Cond. No. < 30    → No multicollinearity
Cond. No. 30–100  → Moderate concern
Cond. No. > 100   → Severe multicollinearity

From example: Cond. No. = 454 → severe multicollinearity ❌
```

### What to do when it fails?

| Fix | How |
|-----|-----|
| **Remove correlated features** | Drop one of the highly correlated variables |
| **PCA** | Transform features into uncorrelated principal components |
| **Ridge Regression** | L2 regularization shrinks coefficients, handles multicollinearity |
| **Lasso Regression** | L1 regularization, can zero out redundant features |
| **Domain knowledge** | Use expertise to decide which feature to keep |

```python
# Ridge regression to handle multicollinearity
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge Coefficients:", ridge.coef_)
```

---

## Full Assumption Check — One Function

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_all_assumptions(model, X, df):
    residuals = model.resid
    fitted    = model.fittedvalues

    print("=" * 55)
    print("LINEAR REGRESSION ASSUMPTION CHECKS")
    print("=" * 55)

    # 1. Linearity
    print("\n1. LINEARITY")
    print("   → Check residual plot below for patterns")

    # 2. Normality
    stat, p = stats.normaltest(residuals)
    print(f"\n2. NORMALITY OF RESIDUALS")
    print(f"   Omnibus: stat={stat:.3f}, p={p:.4f}")
    print(f"   {'✅ Normal' if p > 0.05 else '❌ NOT Normal (p < 0.05)'}")

    # 3. Homoscedasticity
    lm, lm_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    print(f"\n3. HOMOSCEDASTICITY (Breusch-Pagan)")
    print(f"   LM={lm:.3f}, p={lm_p:.4f}")
    print(f"   {'✅ Homoscedastic' if lm_p > 0.05 else '❌ Heteroscedastic (p < 0.05)'}")

    # 4. Autocorrelation
    dw = durbin_watson(residuals)
    print(f"\n4. NO AUTOCORRELATION (Durbin-Watson)")
    print(f"   DW = {dw:.4f}")
    print(f"   {'✅ No autocorrelation' if 1.5 < dw < 2.5 else '❌ Autocorrelation detected'}")

    # 5. Multicollinearity
    print(f"\n5. NO MULTICOLLINEARITY (VIF)")
    X_vif = X.copy()
    X_vif['const'] = 1
    for i, col in enumerate(X_vif.columns):
        vif = variance_inflation_factor(X_vif.values, i)
        flag = '✅' if vif < 5 else ('⚠️' if vif < 10 else '❌')
        print(f"   {flag} VIF({col}) = {vif:.2f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(fitted, residuals, alpha=0.5)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title("Residuals vs Fitted\n(Linearity + Homoscedasticity)")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")

    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title("Histogram of Residuals\n(Normality)")

    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title("Q-Q Plot\n(Normality)")

    plt.tight_layout()
    plt.show()


# Usage
model = smf.ols('Sales ~ TV + Radio + Newspaper', data=df).fit()
X = df[['TV', 'Radio', 'Newspaper']]
check_all_assumptions(model, X, df)
```

---

## Summary Table

| Assumption | Test | Acceptable | Fix if fails |
|-----------|------|------------|-------------|
| Linearity | Residual plot, scatter | Random scatter around 0 | Log transform, polynomial, GAM |
| Normality of Residuals | Q-Q plot, Omnibus, JB | p > 0.05 | Log(y), robust regression, bootstrap |
| Homoscedasticity | Residual plot, Breusch-Pagan | p > 0.05 | Log(y), WLS, robust SEs |
| No Autocorrelation | Durbin-Watson | 1.5 < DW < 2.5 | Lag variables, GLS, ARIMA |
| No Multicollinearity | VIF, Cond. No. | VIF < 5, Cond. No. < 30 | Drop features, Ridge/Lasso, PCA |

---

## Resources

- [Statsmodels Diagnostic Tests](https://www.statsmodels.org/stable/diagnostic.html)
- [CS229 - Andrew Ng Regression Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest - Assumptions of Linear Models](https://www.youtube.com/watch?v=eTZ4VUZHzxw)

---

*Learning in public. Mistakes are part of the process.*
