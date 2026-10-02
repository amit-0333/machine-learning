# Bias–Variance Tradeoff — Mathematics & Code

> Companion to **README_1_Intuition_Notes.md**. This one answers the three lecture questions with formulas and a runnable simulation.

## Contents
1. [Prerequisites: expected value & variance](#1-prerequisites-expected-value--variance)
2. [Notation](#2-notation)
3. [Bias and variance defined](#3-bias-and-variance-defined)
4. [Bias–variance decomposition](#4-biasvariance-decomposition)
5. [Link to underfitting / overfitting](#5-link-to-underfitting--overfitting)
6. [Why there is a tradeoff](#6-why-there-is-a-tradeoff)
7. [Code example](#7-code-example)
8. [Analogy](#8-analogy)
9. [Summary](#9-summary)

---

## 1. Prerequisites: expected value & variance
**Expected value** E[X] is the long-run average of a random variable over many repetitions. Example: a fair six-sided die has E[X] = 3.5.

**Variance** measures spread around that average:

```
Var(X) = E[(X − E[X])²]          (sample version: Σ(xᵢ − x̄)² / n)
```

## 2. Notation

| Symbol | Meaning |
|---|---|
| `f(x)` | true (unknown) function, e.g. x² |
| `y = f(x) + ε` | observed target; ε is noise with mean 0 and variance σ² |
| `f̂(x)` | model's prediction (written f′(x) in the lecture) |
| `E[f̂(x)]` | average prediction over many different training sets |

Key idea: **f̂ depends on the training set**, so f̂(x) is itself a random variable. The "3 students" in the lecture are 3 draws of that randomness; in theory we imagine many.

## 3. Bias and variance defined

**Bias**: systematic error; how far the *average* prediction is from the truth.

```
Bias(f̂(x)) = E[f̂(x)] − f(x)
```

- Bias = 0 → **unbiased predictor** (on average it hits the truth).
- Sign can be positive or negative (over- vs under-prediction).
- Lecture example: true f(x) = x² + 3 but the model learns f̂(x) = x + 5 → consistently wrong → high bias → underfitting.

**Variance**: how much predictions move when the training set changes.

```
Var(f̂(x)) = E[ (f̂(x) − E[f̂(x)])² ]
```

High variance → predictions swing with the data → overfitting.

## 4. Bias–variance decomposition
Expected squared error at a point x, for `y = f(x) + ε`:

```
E[(y − f̂(x))²]  =  Bias(f̂(x))²  +  Var(f̂(x))  +  σ²
                    ───────────     ──────────     ──
                    erroneous        sensitivity    irreducible
                    assumptions      to the sample  noise
```

1. **Bias²**: error from wrong assumptions in the algorithm (missing real structure → underfitting).
2. **Variance**: error from sensitivity to small fluctuations in the training set (fitting noise → overfitting).
3. **Irreducible error σ²**: noise inherent to the problem; no model can remove it.

**Short derivation.** Let f̄ = E[f̂(x)]. Because ε is independent of f̂ and has mean 0:

```
E[(y − f̂)²] = E[(f + ε − f̂)²]
            = E[(f − f̂)²] + 2E[ε(f − f̂)] + E[ε²]
            = E[(f − f̂)²] + 0 + σ²
```

Add and subtract f̄ inside the first term:

```
E[(f − f̂)²] = E[(f − f̄ + f̄ − f̂)²]
            = (f − f̄)² + E[(f̄ − f̂)²] + 2(f − f̄)·E[f̄ − f̂]
            = Bias² + Var + 0          (since E[f̂] = f̄)
```

## 5. Link to underfitting / overfitting

| | Bias | Variance | Train error | Test error |
|---|---|---|---|---|
| **Underfitting** (e.g. linear on x²) | High | Low | High | High |
| **Good fit** (degree 2–3) | Low | Low | Low | Low |
| **Overfitting** (very high degree) | Low | High | Very low | High |

Rule of thumb from the lecture: a big train/test gap (e.g. 90% vs 80%) signals high variance.

## 6. Why there is a tradeoff
Total expected error = Bias² + Var + σ², and σ² is fixed. Increasing model complexity (e.g. polynomial degree) makes the model flexible enough to track f(x), so **Bias² falls**. The same flexibility lets it chase noise in whichever sample it sees, so **Var rises**. Their sum is U-shaped in complexity, so the best model sits where the marginal drop in bias equals the marginal rise in variance.

```
error
 │ \                         ___/  variance
 │  \  bias²            ___/
 │   \___          ___/
 │       \___  ___/   ← total error minimum
 │           \/
 └──────────────────────────── model complexity
   underfit   |   overfit
```

## 7. Code example
Simulate many "students", each with their own training sample from `y = x² + noise` on [−15, 10]. Fit polynomials of different degrees and **measure** bias² and variance at test points.

```python
import numpy as np

rng = np.random.default_rng(42)
f = lambda x: x ** 2
x_lo, x_hi = -15, 10
sigma = 15            # noise std-dev -> irreducible error = sigma**2
n_train = 40          # points per student
n_students = 500      # number of different training sets
x_test = np.linspace(x_lo, x_hi, 100)

def scale(x):         # map to [-1, 1] for numerical stability at high degree
    return 2 * (x - x_lo) / (x_hi - x_lo) - 1

def run(degree):
    preds = np.empty((n_students, len(x_test)))
    for s in range(n_students):
        x = np.linspace(x_lo, x_hi, n_train)   # same x's, fresh noise per student
        y = f(x) + rng.normal(0, sigma, n_train)
        model = np.polynomial.Chebyshev.fit(scale(x), y, degree)  # stable fit
        preds[s] = model(scale(x_test))
    mean_pred = preds.mean(axis=0)                      # E[f_hat(x)]
    bias2 = np.mean((mean_pred - f(x_test)) ** 2)       # Bias^2
    var = np.mean(preds.var(axis=0))                    # Variance
    return bias2, var

print(f"{'degree':>6} {'bias^2':>10} {'variance':>10} {'bias^2+var+sigma^2':>20}")
for d in [1, 2, 3, 8, 15]:
    b2, v = run(d)
    print(f"{d:>6} {b2:>10.2f} {v:>10.2f} {b2 + v + sigma**2:>20.2f}")
```

**What to expect:** degree 1 has huge bias² (~2260) and small variance (underfit); degrees 2–3 have both small (best total); degrees 8–15 keep bias² near zero but variance keeps climbing (~47 → ~91), so total error rises again (overfit).

### Optional: plot the U-curve
```python
import matplotlib.pyplot as plt

degrees = range(1, 16)
res = [run(d) for d in degrees]
b2 = [r[0] for r in res]; v = [r[1] for r in res]
plt.plot(degrees, b2, label="Bias²")
plt.plot(degrees, v, label="Variance")
plt.plot(degrees, np.array(b2) + np.array(v) + sigma**2, label="Total error")
plt.yscale("log"); plt.xlabel("Polynomial degree (model complexity)")
plt.ylabel("Error"); plt.legend(); plt.show()
```

## 8. Analogy
Think of **archery** (the bullseye figure in the notes):
- **High bias**: your sight is misaligned, so every arrow lands consistently off-centre.
- **High variance**: your sight is fine but your hands shake, so arrows scatter all over.
- **Irreducible error**: wind you can't control.
- Goal: aligned sight *and* steady hands.

## 9. Summary
- **Bias** = `E[f̂(x)] − f(x)` → high ⇒ underfitting.
- **Variance** = `E[(f̂(x) − E[f̂(x)])²]` → high ⇒ overfitting.
- **Expected error = Bias² + Variance + Irreducible error.**
- Raising complexity trades bias for variance; pick the complexity that minimises the sum.
