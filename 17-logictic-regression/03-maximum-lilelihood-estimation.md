# Maximum Likelihood Estimation (MLE)

## Table of Contents

1. [Recap](#1-recap)
2. [Probability vs Likelihood](#2-probability-vs-likelihood)
3. [Maximum Likelihood Estimation](#3-maximum-likelihood-estimation)
4. [Some Examples](#4-some-examples)
   - [Example 1: Coin Toss](#example-1-coin-toss)
   - [Example 2: Drawing Balls from a Bag](#example-2-drawing-balls-from-a-bag)
   - [Example 3: Normal Distribution](#example-3-normal-distribution)
5. [MLE for Normal Distribution (full derivation)](#5-mle-for-normal-distribution-full-derivation)
6. [MLE in Machine Learning](#6-mle-in-machine-learning)
7. [MLE in Logistic Regression](#7-mle-in-logistic-regression)
8. [Properties & Limitations of MLE](#8-properties--limitations-of-mle)
9. [Some Important Questions](#9-some-important-questions)
10. [Tasks (with Solutions)](#10-tasks-with-solutions)
11. [Summary & Cheat Sheet](#11-summary--cheat-sheet)

> **Note:** several pages of the source PDF (the worked examples and the ML / logistic-regression pages) were handwritten, so their text could not be extracted. Those sections follow the standard textbook treatment of the same examples. The "Some Important Questions" section follows your typed notes.

---

## 1. Recap

From the Logistic Regression sessions:

- The model gives a probability `ŷ = σ(β0 + β1x1 + ...)` for each point.
- **Likelihood** = the *product of the predicted probabilities for the actual class of each observation*. The best line is the one that **maximizes** it.
- Taking `−log` and averaging gives the **log loss (binary cross-entropy)**, so maximizing likelihood ⇔ minimizing log loss.

In this session we study the idea behind that, **Maximum Likelihood Estimation (MLE)**, in general: where it comes from, how to apply it to simple distributions, and how it explains the loss functions used in machine learning.

---

## 2. Probability vs Likelihood

### 2.1 Definitions

> **Probability:** a measure of the chance that a certain event will occur out of all possible events. It is usually presented as a ratio or fraction, and it ranges from **0** (the event will not happen) to **1** (the event is certain to happen).

> **Likelihood:** in a statistical context, likelihood is a function that measures the **plausibility of a particular parameter value given some observed data**. It quantifies how well a specific outcome supports specific parameter values.

### 2.2 In plain words

- A **probability** quantifies how often you observe a certain outcome of a test, **given a certain understanding of the underlying data** (a known model).
- A **likelihood** quantifies **how good one's model is, given a set of data that's been observed**.

> **Probabilities describe test outcomes, while likelihoods describe models.**

### 2.3 Side-by-side

| | Probability | Likelihood |
|---|---|---|
| Question | "Given the model/parameters, how likely is this data?" | "Given the data, how plausible are these parameters?" |
| Known (fixed) | Parameters θ | Data x |
| Varies | Data x | Parameters θ |
| Notation | `P(x \| θ)` | `L(θ \| x)` |
| Sums/integrates to 1? | **Yes** (over all possible data) | **No** (over θ it need not) |
| Example | "If the coin is fair, what's the chance of 7 heads in 10?" | "I saw 7 heads in 10. How plausible is p = 0.5 vs p = 0.7?" |

The **same formula** is used for both. What changes is what you treat as fixed and what you vary.

### 2.4 Numeric illustration

A coin is tossed 10 times; 7 heads. The binomial formula is `C(10,7) · p⁷ · (1−p)³`, with `C(10,7) = 120`.

| Value of p | Value of the formula |
|---|---|
| 0.3 | 0.0090 |
| 0.5 | 0.1172 |
| **0.7** | **0.2668** (highest) |
| 0.9 | 0.0574 |

- **As a probability** (p fixed at 0.5): "7 heads out of 10 has probability 0.117".
- **As a likelihood** (data fixed at 7 heads): "p = 0.7 is more plausible than p = 0.5; likelihood 0.267 vs 0.117". Note these values don't add up to 1 across p.

---

## 3. Maximum Likelihood Estimation

> **Maximum Likelihood Estimation (MLE)** is a method of estimating the **parameters of a statistical model** given some observed data.

**Idea:** pick the parameter values that make the observed data **most probable** under the model.

### 3.1 The recipe (4 steps)

1. **Assume a model / distribution** with unknown parameters θ (e.g., Bernoulli(p), Normal(μ, σ²)).
2. **Write the likelihood** of the whole dataset. For independent observations (i.i.d.) it is a **product**:

   ```
   L(θ) = ∏ᵢ f(xᵢ | θ)
   ```

3. **Take the logarithm** to get the **log-likelihood** (product → sum, easier and numerically stable):

   ```
   ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ | θ)
   ```

4. **Maximize**: set the derivative to zero, `dℓ/dθ = 0`, solve for θ, and check it is a maximum (second derivative < 0). If no closed form exists, use numerical optimization (gradient descent).

```
θ̂_MLE = argmax_θ  L(θ) = argmax_θ  log L(θ) = argmin_θ  [ − log L(θ) ]
```

### 3.2 Why log?

- `log` is increasing, so the maximizer is **unchanged**.
- Products of many small numbers underflow; sums don't.
- Derivatives of sums are simple.
- The negative log-likelihood is exactly the **loss function** we minimize in ML (section 6).

---

## 4. Some Examples

### Example 1: Coin Toss

**Problem.** A coin with unknown probability of heads `p`. We toss it `n = 10` times and get `k = 7` heads (H H T H H H T H T H). Estimate `p`.

**Step 1: model.** Each toss ~ Bernoulli(p): `P(H) = p`, `P(T) = 1 − p`.

**Step 2: likelihood** of the observed sequence:

```
L(p) = p^k · (1 − p)^(n−k) = p⁷ · (1 − p)³
```

**Step 3: log-likelihood:**

```
ℓ(p) = k·ln p + (n − k)·ln(1 − p)
```

**Step 4: maximize:**

```
dℓ/dp = k/p − (n − k)/(1 − p) = 0
⇒ k(1 − p) = (n − k)p
⇒ k = n·p
⇒ p̂ = k / n = 7/10 = 0.7
```

Second derivative `−k/p² − (n−k)/(1−p)² < 0` ⇒ a maximum. ✓

**Result:** `p̂ = fraction of heads`, the intuitive answer. MLE justifies it.

**Check numerically** (likelihood of the sequence):

| p | L(p) = p⁷(1−p)³ |
|---|---|
| 0.5 | 0.000977 |
| 0.6 | 0.001792 |
| **0.7** | **0.002224** (max) |
| 0.8 | 0.001678 |

```python
import numpy as np
p = np.linspace(0.01, 0.99, 99)
L = p**7 * (1 - p)**3
print(p[L.argmax()])     # 0.7
```

### Example 2: Drawing Balls from a Bag

**Problem A (unknown proportion).** A bag contains red and blue balls in unknown proportion. We draw a ball, note its colour, **replace** it, and repeat. After `n = 20` draws we saw `r = 6` red. Estimate `p` = proportion of red balls.

This is the same Bernoulli/binomial model as the coin toss:

```
L(p) = p^r (1 − p)^(n − r)   ⇒   p̂ = r / n = 6/20 = 0.3
```

**Problem B (discrete parameter).** The bag has exactly **3 balls**, some red and some blue; `θ` = number of red balls ∈ {0, 1, 2, 3}. We draw 5 times with replacement and see **4 red and 1 blue** (in a given order). Which θ is the most likely?

Here `p = θ/3`, and `L(θ) = p⁴(1 − p)`:

| θ (red balls) | p = θ/3 | L = p⁴(1−p) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1/3 | (1/81)(2/3) = 0.0082 |
| **2** | **2/3** | **(16/81)(1/3) = 0.0658** (max) |
| 3 | 1 | 0 (we saw a blue ball, impossible) |

**MLE: θ̂ = 2 red balls.** Here we simply compare likelihoods for each candidate, since θ is discrete (no calculus needed).

**Without replacement:** use the hypergeometric distribution. MLE for the number of red balls works the same way (pick the count that maximizes the likelihood of the observed draw).

### Example 3: Normal Distribution

**Problem.** Heights (cm) of 5 people: `160, 165, 170, 175, 180`. Assume they come from `N(μ, σ²)`. Estimate `μ` and `σ²`.

**MLE results** (derived in the next section):

```
μ̂ = (1/n) Σ xᵢ                  = sample mean
σ̂² = (1/n) Σ (xᵢ − μ̂)²          = sample variance (dividing by n)
```

For this data: `μ̂ = 850/5 = 170`, `σ̂² = (100 + 25 + 0 + 25 + 100)/5 = 50`, `σ̂ ≈ 7.07`.

So the best-fitting normal curve is centred at the mean and its width equals the data's spread. ✓

---

## 5. MLE for Normal Distribution (full derivation)

**Model:** `x₁, …, xₙ` i.i.d. from `N(μ, σ²)` with density

```
f(x | μ, σ²) = (1 / √(2πσ²)) · exp( −(x − μ)² / (2σ²) )
```

**Likelihood:**

```
L(μ, σ²) = ∏ᵢ (2πσ²)^(−1/2) · exp( −(xᵢ − μ)² / (2σ²) )
```

**Log-likelihood:**

```
ℓ(μ, σ²) = −(n/2)·ln(2π) − (n/2)·ln(σ²) − (1/(2σ²)) · Σᵢ (xᵢ − μ)²
```

### Step 1: differentiate w.r.t. μ

```
∂ℓ/∂μ = (1/σ²) · Σᵢ (xᵢ − μ) = 0
⇒ Σ xᵢ − nμ = 0
⇒ μ̂ = (1/n) Σ xᵢ = x̄
```

### Step 2: differentiate w.r.t. σ²

```
∂ℓ/∂σ² = −n/(2σ²) + (1/(2σ⁴)) · Σ (xᵢ − μ)² = 0
⇒ σ̂² = (1/n) Σ (xᵢ − μ̂)²
```

### Conclusions

- MLE of the mean = **sample mean**.
- MLE of the variance = **average squared deviation** (divide by `n`).
- ⚠️ `σ̂²` is **slightly biased** (it underestimates the true variance). The unbiased estimator divides by `n − 1`. For large `n` the difference vanishes. (MLE is *consistent* but not always unbiased.)

```python
import numpy as np
x = np.array([160, 165, 170, 175, 180])
mu = x.mean()                      # 170.0
sigma2 = ((x - mu) ** 2).mean()    # 50.0   (MLE: divide by n)
print(mu, sigma2, x.var(ddof=0), x.var(ddof=1))   # 170.0 50.0 50.0 62.5
```

---

## 6. MLE in Machine Learning

**Key insight:** training a model by minimizing a standard loss is usually **the same as** doing MLE under a specific assumption about the noise or target distribution.

```
Negative log-likelihood (NLL)  =  Loss function
Minimize loss  ⇔  Maximize likelihood
```

### 6.1 Linear regression ⇒ Mean Squared Error

Assume `y = wᵀx + ε` with Gaussian noise `ε ~ N(0, σ²)`. Then `y | x ~ N(wᵀx, σ²)`:

```
log L(w) = −(n/2) ln(2πσ²) − (1/(2σ²)) Σ (yᵢ − wᵀxᵢ)²
```

The first term doesn't depend on `w`, so maximizing log L ⇔ **minimizing Σ(yᵢ − ŷᵢ)²** = MSE.

### 6.2 Logistic regression ⇒ Binary cross-entropy

Assume `y | x ~ Bernoulli(ŷ)` with `ŷ = σ(wᵀx)` (see section 7).

### 6.3 Multiclass ⇒ Categorical cross-entropy

Assume `y | x ~ Categorical(ŷ₁ … ŷ_K)` with softmax probabilities ⇒ NLL = `−Σ yᵏ log ŷᵏ`.

### 6.4 Summary table

| Model | Assumed distribution of y | Negative log-likelihood = Loss |
|---|---|---|
| Linear regression | Gaussian, fixed variance | **Mean Squared Error** |
| Linear regression | Laplace | Mean Absolute Error |
| Logistic regression | Bernoulli | **Binary cross-entropy (log loss)** |
| Softmax regression | Categorical | **Categorical cross-entropy** |
| Poisson regression | Poisson | Poisson deviance |
| Neural network (classification) | Bernoulli / Categorical | Cross-entropy |
| Neural network (regression) | Gaussian | MSE |

### 6.5 Adding a prior: regularization = MAP

If we also assume a prior belief about the parameters, we get **Maximum A Posteriori (MAP)** estimation:

| Prior on weights | Resulting penalty |
|---|---|
| Gaussian `N(0, τ²)` | L2 (Ridge) |
| Laplace | L1 (Lasso) |

`MLE = loss only`  →  `MAP = loss + regularization`.

---

## 7. MLE in Logistic Regression

**Model:** `P(y = 1 | x) = ŷ = σ(z)`, `z = β0 + β1x1 + ... + βdxd`.

For one observation (label `y ∈ {0,1}`), the Bernoulli probability is

```
P(y | x) = ŷ^y · (1 − ŷ)^(1−y)
```

- If `y = 1` it equals `ŷ`; if `y = 0` it equals `1 − ŷ`. This is "the predicted probability of the actual class".

**Likelihood** for `n` independent observations:

```
L(β) = ∏ᵢ ŷᵢ^yᵢ · (1 − ŷᵢ)^(1 − yᵢ)
```

**Log-likelihood:**

```
ℓ(β) = Σᵢ [ yᵢ log ŷᵢ + (1 − yᵢ) log(1 − ŷᵢ) ]
```

**Negative average log-likelihood = Log Loss:**

```
J(β) = −(1/n) Σᵢ [ yᵢ log ŷᵢ + (1 − yᵢ) log(1 − ŷᵢ) ]
```

So **minimizing log loss is exactly maximizing the likelihood**: `β̂ = argmax L(β) = argmin J(β)`.

**No closed form** (unlike the normal distribution), so we use **gradient descent**:

```
∂J/∂βj = (1/n) Σᵢ (ŷᵢ − yᵢ) xᵢⱼ
```

**Numeric check** (from the Logistic Regression notes): probabilities assigned to the true class were `0.8, 0.6, 0.7, 0.6`:

```
Likelihood = 0.8 × 0.6 × 0.7 × 0.6 = 0.2016
Log loss   = −(1/4)[ln 0.8 + ln 0.6 + ln 0.7 + ln 0.6] = 0.400
```

A different line with `0.7, 0.4, 0.3, 0.6` gives likelihood 0.0504 and log loss 0.747. Higher likelihood ⇔ lower loss ⇔ better model.

---

## 8. Properties & Limitations of MLE

### Good properties (for large samples, under regularity conditions)

| Property | Meaning |
|---|---|
| **Consistency** | As n → ∞, θ̂ converges to the true θ |
| **Asymptotic normality** | θ̂ is approximately normally distributed around θ, which gives confidence intervals |
| **Efficiency** | Reaches the lowest possible variance (Cramér-Rao bound) asymptotically |
| **Invariance** | If θ̂ is the MLE of θ, then g(θ̂) is the MLE of g(θ) |

### Limitations

- Needs an **assumed distribution**. If the assumption is wrong, the estimates may be poor.
- Can **overfit with small samples** (e.g., 3 tosses all heads ⇒ p̂ = 1). Priors/regularization fix this.
- May be **biased** (e.g., σ̂² divides by n).
- Sometimes **no closed form** ⇒ numerical optimization, possible local optima (non-convex models).
- Sensitive to **outliers** (e.g., Gaussian assumption).

---

## 9. Some Important Questions

### Q1. Is MLE a general concept applicable to all machine learning algorithms?

MLE is a general **statistical** concept that can be applied to many machine learning algorithms, particularly **parametric** ones (defined by a set of parameters), but it is **not applicable to all** machine learning algorithms.

**Where it is used:** linear regression, logistic regression, neural networks and others. These use MLE to find the parameter values that best fit the training data.

**Where it is not used:**

1. **Non-parametric methods**: k-Nearest Neighbors and Decision Trees make no strong assumptions about the data distribution and have no fixed set of parameters to optimize with MLE.
2. **Unsupervised learning algorithms**: e.g., K-means uses a different objective function not tied to a probability distribution.
3. **Reinforcement learning**: focuses on learning from rewards and punishments over a sequence of actions rather than fitting a data distribution.

*(Nuance: some of these have probabilistic versions. Gaussian Mixture Models, for example, are fit by MLE using the EM algorithm.)*

### Q2. How is MLE related to the concept of loss functions?

A loss function measures how well a model's predictions align with actual values, and training aims to find parameters that **minimize** it. MLE estimates parameters that **maximize the likelihood function**, which is conceptually similar to minimizing a loss. For many common models, **minimizing the loss function is equivalent to maximizing the likelihood function**. Many common loss functions can be derived from MLE under certain assumptions about the data or model, so by minimizing them we are effectively performing MLE.

### Q3. Then why does a loss function exist? Why don't we just maximize likelihood?

The two are **two perspectives on the same problem**: the optimization view (minimize the difference between predicted and actual values) and the statistical view (maximize the likelihood of the observed data given the parameters). For many models they are equivalent. We often prefer to talk about loss because:

1. **Computational reasons:** it is often easier and more efficient to minimize a loss than to maximize a likelihood, particularly for complex models like neural networks. (Also, optimizers are conventionally written as minimizers, and the log of products becomes sums.)
2. **Generalization:** the loss concept is more general. Not every problem can be framed as maximizing a likelihood. For example, many non-parametric and unsupervised methods don't involve likelihoods.
3. **Flexibility:** losses are easy to customize, e.g., weighting some errors more, or using a loss that is robust to outliers (Huber loss, hinge loss).

In summary: loss minimization and MLE are closely related and often equivalent, but the loss concept is more flexible and computationally convenient, which is why it is more common in the ML community.

### Q4. Then why study maximum likelihood at all?

1. **Statistical foundation:** MLE gives a principled way to **derive** the loss functions used in common algorithms and explains why they work and under which assumptions.
2. **Interpretability:** the MLEs are the parameters that make the observed data most likely under the model, which helps explain what the model learned.
3. **Model comparison:** likelihood-based tools such as **AIC** and **BIC** help choose the best model:

   ```
   AIC = 2k − 2 ln(L̂)
   BIC = k·ln(n) − 2 ln(L̂)       (k = number of parameters, n = sample size, L̂ = max likelihood)
   ```

   Lower is better. Both reward a good fit and penalize extra parameters.
4. **Generalization to other methods:** MLE is a stepping stone to more general methods such as **Expectation-Maximization** and **Bayesian inference**.
5. **Deeper understanding:** better intuition, better model selection and ultimately better performance.

While you can often get by with a practical understanding of loss functions and optimizers, understanding MLE is extremely valuable for knowing **how and why** models work.

---

## 10. Tasks (with Solutions)

The PDF lists the *Tasks* heading followed by the four questions above. Here are additional practice problems with solutions.

### Task A: MLE for a Bernoulli parameter (with a prior-free check)

Data: `1, 0, 1, 1, 0, 1, 1, 1` (1 = success). Find `p̂`.

`k = 6` successes in `n = 8` ⇒ **p̂ = 6/8 = 0.75**.

### Task B: MLE for the exponential distribution

`f(x | λ) = λ e^(−λx)`, data `x₁…xₙ`.

```
ℓ(λ) = n ln λ − λ Σ xᵢ
dℓ/dλ = n/λ − Σ xᵢ = 0   ⇒   λ̂ = n / Σ xᵢ = 1 / x̄
```

### Task C: Show that MLE for linear regression gives least squares

See section 6.1: Gaussian noise ⇒ maximizing the likelihood = minimizing Σ(y − ŷ)².

### Task D: Show that MLE for logistic regression gives log loss

See section 7: Bernoulli likelihood ⇒ `J(β) = −(1/n)Σ[y log ŷ + (1−y) log(1−ŷ)]`.

### Task E: Fit a distribution with code

```python
import numpy as np
from scipy import stats

data = np.random.default_rng(0).normal(loc=5, scale=2, size=1000)

mu_hat, sigma_hat = stats.norm.fit(data)       # MLE fit
print(mu_hat, sigma_hat)                       # ≈ 5, ≈ 2

# Same thing manually
print(data.mean(), data.std(ddof=0))
```

### Task F: Numerical MLE for logistic regression

```python
import numpy as np
from scipy.optimize import minimize

def neg_log_likelihood(beta, X, y):
    z = X @ beta
    p = 1 / (1 + np.exp(-z))
    eps = 1e-12
    return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

rng = np.random.default_rng(0)
X = np.c_[np.ones(200), rng.normal(size=(200, 2))]       # column of 1s = intercept
true_beta = np.array([-0.5, 1.5, -2.0])
y = (rng.random(200) < 1 / (1 + np.exp(-X @ true_beta))).astype(int)

res = minimize(neg_log_likelihood, x0=np.zeros(3), args=(X, y), method="BFGS")
print("MLE beta:", res.x)       # close to the true [-0.5, 1.5, -2.0]
```

---

## 11. Summary & Cheat Sheet

1. **Probability** = chance of data given a model (parameters fixed). **Likelihood** = plausibility of parameters given data (data fixed). *Probabilities describe outcomes; likelihoods describe models.*
2. **MLE** = estimate parameters so that the observed data is **most probable**: `θ̂ = argmax L(θ)`.
3. **Recipe:** assume a distribution → write the likelihood (product for i.i.d. data) → take the log → differentiate, set to 0 → solve (or use numerical optimization).
4. **Standard results:** coin toss `p̂ = k/n`; normal `μ̂ = x̄`, `σ̂² = (1/n)Σ(x − x̄)²`; exponential `λ̂ = 1/x̄`.
5. **In ML:** `loss = negative log-likelihood`. Gaussian noise ⇒ MSE; Bernoulli ⇒ log loss; Categorical ⇒ cross-entropy. Add a prior ⇒ MAP ⇒ L1/L2 regularization.
6. **Logistic regression:** likelihood `∏ ŷ^y (1−ŷ)^(1−y)` ⇒ log loss; no closed form ⇒ gradient descent.
7. MLE isn't universal (k-NN, trees, k-means, RL don't use it), but understanding it explains **why** losses look the way they do, enables model comparison (AIC/BIC), and leads to EM and Bayesian methods.

```
Assumed distribution ─► Likelihood L(θ) ─► log L(θ) ─► maximize ─► θ̂
                                              │
                                   −log L(θ) = LOSS FUNCTION
```

| Distribution | Parameter | MLE |
|---|---|---|
| Bernoulli(p) | p | k/n |
| Normal(μ, σ²) | μ, σ² | x̄, (1/n)Σ(x−x̄)² |
| Exponential(λ) | λ | 1/x̄ |
| Poisson(λ) | λ | x̄ |
