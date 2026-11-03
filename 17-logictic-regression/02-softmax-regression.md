# Logistic Regression: Session 2 (Multiclass Classification)

Extending logistic regression beyond two classes with **One-vs-Rest (OVR)** and **Softmax (Multinomial) Regression**.

## Table of Contents

1. [Recap](#1-recap)
2. [What is Multiclass Classification?](#2-what-is-multiclass-classification)
3. [How Logistic Regression Handles Multiclass Problems](#3-how-logistic-regression-handles-multiclass-problems)
4. [OVR Approach](#4-ovr-approach)
5. [Code: OVR](#5-code-ovr)
6. [SoftMax Function](#6-softmax-function)
7. [SoftMax Logistic Regression](#7-softmax-logistic-regression)
8. [Code: Softmax Regression](#8-code-softmax-regression)
9. [When to Use What?](#9-when-to-use-what)
10. [Tasks (with Solutions)](#10-tasks-with-solutions)
11. [Summary](#11-summary)

---

## 1. Recap

From Session 1 (binary logistic regression):

| Concept | Formula / Idea |
|---|---|
| Linear score | `z = β0 + β1·x1 + β2·x2 + ...` |
| Sigmoid | `σ(z) = 1 / (1 + e^−z)` → probability in (0, 1) |
| Prediction | `ŷ = P(y = 1 \| x) = σ(z)`; class 1 if `ŷ ≥ 0.5` |
| Best line | The one that **maximizes likelihood** |
| Loss | **Log loss (binary cross-entropy)** `L = −(1/n) Σ [ y·log ŷ + (1−y)·log(1−ŷ) ]` |
| Optimization | **Gradient descent**: `β ← β − η·∇L` with `∇L = (1/n)·Xᵀ(ŷ − y)` |
| Boundary | `z = 0` is a straight line → logistic regression is a **linear model** |

**Limitation:** the sigmoid produces a single probability, so it only separates **two** classes. This session fixes that.

---

## 2. What is Multiclass Classification?

A classification problem where the target variable can take **more than two** discrete values, and each sample belongs to **exactly one** class.

**Example from the notes:** a placement dataset with features `cgpa` and `iq` and a 3-valued target:

| cgpa | iq | placement |
|---|---|---|
| 8.1 | 110 | **Yes** (placed) |
| 5.2 | 85 | **No** (not placed) |
| 7.0 | 98 | **Opt** (opted out) |

Other examples: digit recognition (10 classes), iris flower species (3), sentiment (negative / neutral / positive), news topic categorization.

### Types of classification (don't confuse them)

| Type | Number of classes | Labels per sample | Example |
|---|---|---|---|
| **Binary** | 2 | 1 | spam / not spam |
| **Multiclass** | > 2 | exactly **1** | digit 0 to 9 |
| **Multilabel** | > 2 | **one or more** | a movie tagged "action" and "comedy" |

This session focuses on **multiclass** (one label per sample).

---

## 3. How Logistic Regression Handles Multiclass Problems

Logistic regression is natively binary, so there are **two strategies** to extend it:

```
                 Multiclass Logistic Regression
                          │
         ┌────────────────┴─────────────────┐
         ▼                                  ▼
 One-vs-Rest (OVR)              Softmax / Multinomial LR
 K separate binary models       One model, K sets of weights
 (sigmoid each)                 (softmax over all K scores)
```

---

## 4. OVR Approach

**OVR** (One-vs-Rest, also called One-vs-All) converts a K-class problem into **K binary problems**.

### 4.1 Training

For each class `k`:

1. Relabel the data: samples of class `k` → **1**, all other samples → **0**.
2. Train an ordinary binary logistic regression on this relabelled data.

**Example with 3 classes (Yes, No, Opt):**

| Model | Positive class (1) | Negative class (0) |
|---|---|---|
| Model 1 | Yes | No + Opt |
| Model 2 | No | Yes + Opt |
| Model 3 | Opt | Yes + No |

### 4.2 Prediction

For a new point, every model outputs `P(class k | x)`. Predict the class with the **highest probability**:

```
ŷ = argmax over k of  σ(zk)
```

### 4.3 Number of parameters

Each binary model has `1 + (number of features)` parameters. With 2 features (`cgpa`, `iq`) → `β0, β1, β2` = **3 per model**.

For K = 3 classes: 3 models × 3 parameters = **9 parameters**. In general: `K × (d + 1)`.

(Geometrically: **K lines** are drawn, each separating one class from the rest.)

### 4.4 Drawbacks

- The K models are trained **independently**, so the K probabilities **do not naturally sum to 1** (scikit-learn normalizes them afterwards).
- Each "class vs rest" problem is **imbalanced** (1 class against K−1 combined).
- Regions where several or none of the classifiers fire are ambiguous (resolved by taking the largest score).
- K models mean K training runs.

---

## 5. Code: OVR

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# 3 classes, 2 features (like cgpa & iq)
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

ovr = OneVsRestClassifier(LogisticRegression())
ovr.fit(X_train, y_train)

print("Number of binary models:", len(ovr.estimators_))        # 3
print("coef_ of model 1:", ovr.estimators_[0].coef_)           # shape (1, 2)
print("Accuracy:", accuracy_score(y_test, ovr.predict(X_test)))
print(ovr.predict_proba(X_test[:2]))                           # rows sum to 1
```

> **Version note:** older tutorials use `LogisticRegression(multi_class="ovr")`. That argument is deprecated in recent scikit-learn versions. Wrapping with `OneVsRestClassifier` is the future-proof way and works in all versions.

---

## 6. SoftMax Function

### 6.1 From sigmoid to softmax

- **Sigmoid** (binary): turns one score `z` into a probability between 0 and 1.

  ```
  σ(z) = 1 / (1 + e^−z)
  ```

- **Softmax** (multiclass): turns **K scores** `z1, z2, ..., zK` into **K probabilities that sum to 1**.

  ```
  σ(z)ᵢ = e^(zᵢ) / Σⱼ₌₁..K e^(zⱼ)
  ```

### 6.2 Example with 3 scores (from the notes)

```
σ(z1) = e^z1 / (e^z1 + e^z2 + e^z3)
σ(z2) = e^z2 / (e^z1 + e^z2 + e^z3)
σ(z3) = e^z3 / (e^z1 + e^z2 + e^z3)

σ(z1) + σ(z2) + σ(z3) = 1
```

That sum-to-1 property is exactly what makes it suitable for **multiclass** problems.

**Numeric example:** `z = [2.0, 1.0, 0.1]`

| | z | e^z | softmax |
|---|---|---|---|
| Class 1 | 2.0 | 7.389 | 0.659 |
| Class 2 | 1.0 | 2.718 | 0.242 |
| Class 3 | 0.1 | 1.105 | 0.099 |
| **Sum** | | 11.212 | **1.000** |

### 6.3 Properties

- Every output is in (0, 1) and all outputs **sum to 1** → a valid probability distribution.
- The exponential makes **larger scores dominate** (a "soft" version of argmax: the highest score gets the largest probability but not all of it).
- **Shift invariance:** adding the same constant to all `z` doesn't change the result. We use this for numerical stability.
- It generalizes sigmoid: with K = 2 it reduces to a sigmoid (proved in Task 1).

### 6.4 Code (numerically stable)

```python
import numpy as np

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # subtract max to avoid overflow
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

print(softmax(np.array([2.0, 1.0, 0.1])))   # [0.659 0.242 0.099]
```

---

## 7. SoftMax Logistic Regression

Also called **Multinomial Logistic Regression** or **Softmax Regression**. It is **one model** that handles all K classes **at once**.

### 7.1 The model

For K classes and d features, learn one weight vector per class:

```
zk = β0k + β1k·x1 + β2k·x2 + ... + βdk·xd        for k = 1..K

ŷk = P(y = k | x) = e^zk / Σⱼ e^zj
```

Predict the class with the highest probability.

### 7.2 Geometric view and parameter count

- Binary LR → **1 line**.
- Softmax LR with K classes → **K lines** (one score per class) which together carve the plane into K regions; the decision boundary lies where two top scores are equal.

| Setup | Parameters |
|---|---|
| 3 classes, 2 features (cgpa, iq) | 3 lines × 3 parameters (β0, β1, β2) = **9** |
| 4 classes, 2 features | 4 × 3 = **12** |
| K classes, d features | **K × (d + 1)** |

In matrix form: `Z = X·W + b`, where `W` has shape `(d, K)` and `b` has shape `(K,)`.

### 7.3 Training: Step 1 is one-hot encoding the target

Convert the K-valued target into K binary columns (**one-hot encoding**):

| cgpa | iq | placement | → | y_Yes | y_No | y_Opt |
|---|---|---|---|---|---|---|
| … | … | Yes | | **1** | 0 | 0 |
| … | … | No | | 0 | **1** | 0 |
| … | … | Opt | | 0 | 0 | **1** |

### 7.4 The loss function: Categorical Cross-Entropy

Binary case (K = 2):

```
L = −(1/n) Σᵢ [ yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) ]
```

General K-class case:

```
L = −(1/n) Σᵢ₌₁..n  Σₖ₌₁..K  yᵢᵏ · log(ŷᵢᵏ)
```

- `i` runs over samples, `k` over classes.
- `yᵢᵏ` is 1 only for the true class (one-hot), so for each sample only **one term survives**: `−log(predicted probability of the true class)`.
- For `K = 2` this collapses into the binary log loss above (Task 2).

**Walk-through for one row (i = 1, from the notes):** suppose the true placement is **Yes**, so `y = (1, 0, 0)`:

```
Σₖ yᵏ·log(ŷᵏ) = 1·log(ŷ_Yes) + 0·log(ŷ_No) + 0·log(ŷ_Opt) = log(ŷ_Yes)
```

For a row with true class "No" the surviving term is `log(ŷ_No)`, and for "Opt" it is `log(ŷ_Opt)`.

So the loss only looks at **the probability assigned to the correct class**, same idea as likelihood: *maximize the probability of the right answer for every sample*.

### 7.5 Training: gradient descent

Minimize `L` with gradient descent. The gradient has the same elegant form as in the binary case (derived in Task 4):

```
∂L/∂zᵢₖ = ŷᵢₖ − yᵢₖ                   (per sample, per class)

∂L/∂W = (1/n) · Xᵀ (Ŷ − Y)             (shape d × K)
∂L/∂b = (1/n) · Σᵢ (Ŷᵢ − Yᵢ)           (shape K)

W ← W − η · ∂L/∂W
b ← b − η · ∂L/∂b
```

where `Ŷ` is the `n × K` matrix of softmax outputs and `Y` is the `n × K` one-hot matrix.

### 7.6 Training → Prediction summary

```
Training:    one-hot encode y → compute Z = XW + b → softmax → cross-entropy loss
             → gradients → update W, b → repeat
Prediction:  Z = XW + b → softmax → pick argmax class
```

---

## 8. Code: Softmax Regression

### 8.1 With scikit-learn

By default, `LogisticRegression` uses the **multinomial (softmax)** formulation for multiclass problems with solvers like `lbfgs`, `newton-cg`, `sag`, `saga`.

```python
sm = LogisticRegression()          # multinomial by default for multiclass
sm.fit(X_train, y_train)

print("coef_ shape:", sm.coef_.shape)            # (3, 2)  -> K × d
print("intercept_ shape:", sm.intercept_.shape)  # (3,)    -> K
# total parameters = 3×2 + 3 = 9

print("Accuracy:", sm.score(X_test, y_test))
proba = sm.predict_proba(X_test[:3])
print(proba, proba.sum(axis=1))                  # each row sums to 1
```

### 8.2 From scratch (NumPy)

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def fit_softmax(X, y, K, lr=0.5, epochs=2000):
    n, d = X.shape
    W = np.zeros((d, K))
    b = np.zeros(K)
    Y = np.eye(K)[y]                    # one-hot encoding

    for _ in range(epochs):
        P = softmax(X @ W + b)          # n × K probabilities
        G = (P - Y) / n                 # gradient wrt scores Z
        W -= lr * (X.T @ G)
        b -= lr * G.sum(axis=0)
    return W, b

W, b = fit_softmax(X_train, y_train, K=3)
pred = (X_test @ W + b).argmax(axis=1)
print("Accuracy:", (pred == y_test).mean())
```

### 8.3 Binary case

For a 2-class problem, scikit-learn uses a single sigmoid model (`coef_` shape `(1, d)`), i.e. **3 parameters** for 2 features. A softmax with K = 2 would use 6 parameters, but they are redundant: only the *difference* between the two weight sets matters (see Task 1).

---

## 9. When to Use What?

**Use One-vs-Rest (OVR) when:**

1. **Classes are non-mutually exclusive:** OVR is appropriate if an instance can belong to more than one class, as each classifier provides an independent probability for each class.
2. **Dealing with imbalanced data:** OVR might perform better when class distribution is highly imbalanced, since each class gets a dedicated model.

**Use Multinomial Logistic Regression (SoftMax Regression) when:**

1. **Computational efficiency is required:** Softmax regression is generally more efficient for large datasets and a high number of classes.
2. **Classes are mutually exclusive:** SoftMax regression is a good choice when each instance can only belong to one class. The SoftMax function provides a set of probabilities that sum to 1, fitting well with mutually exclusive classes.
3. **Interpretability is important:** The probabilities output by SoftMax regression are more interpretable than those from OVR, as they always sum to 1. This can make model predictions easier to explain.

### Quick comparison table

| Aspect | OVR | Softmax |
|---|---|---|
| Number of models | K binary models | 1 joint model |
| Output function | Sigmoid per class | Softmax across classes |
| Probabilities sum to 1? | No (needs normalizing) | **Yes**, by design |
| Training | Each class independently | All classes together |
| Loss | K binary log losses | Categorical cross-entropy |
| Parameters | K × (d + 1) | K × (d + 1) |
| Class balance issue | Each model is "1 vs rest" (imbalanced) | Not by construction |
| Multilabel (several classes at once) | Natural fit (independent per-label models) | Not suitable |

### Extra notes / caveats

- For a **multilabel** problem (a sample may have several labels at once), the independent-sigmoid idea behind OVR is the right one; plain multiclass softmax forces exactly one winner.
- OVR with imbalanced classes: each dedicated model sees a very skewed "class vs rest" problem, so the claim above holds only in some cases. In practice, compare both with cross-validation and use `class_weight="balanced"` when needed.
- In practice, **softmax is the default and usually the better choice** for ordinary single-label multiclass problems.

---

## 10. Tasks (with Solutions)

The session ended with four derivations to practise. Worked solutions below.

### Task 1: Derive sigmoid from softmax

Take **K = 2** classes with scores `z1` and `z2`:

```
softmax(z)₁ = e^z1 / (e^z1 + e^z2)
```

Divide numerator and denominator by `e^z1`:

```
        1
= ───────────────  =  1 / (1 + e^−(z1 − z2))  =  σ(z1 − z2)
  1 + e^(z2 − z1)
```

Define a single score `z = z1 − z2` (or set `z2 = 0` and `z1 = z`):

```
P(class 1) = σ(z)            P(class 2) = 1 − σ(z) = σ(−z)
```

So **sigmoid is softmax for two classes**. This also shows why only the *difference* of scores matters, and why binary LR needs only one weight set (3 parameters instead of 6).

### Task 2: Derive binary cross-entropy from categorical cross-entropy

Categorical cross-entropy for one sample:

```
L = − Σₖ₌₁..K yᵏ · log(ŷᵏ)
```

For **K = 2**: `L = −[ y¹·log(ŷ¹) + y²·log(ŷ²) ]`

Because the two classes are complementary:

```
y¹ = y,     y² = 1 − y
ŷ¹ = ŷ,     ŷ² = 1 − ŷ
```

Substitute:

```
L = −[ y·log(ŷ) + (1 − y)·log(1 − ŷ) ]
```

which is exactly **binary cross-entropy (log loss)**. Averaging over n samples gives the formula from Session 1.

### Task 3: Derivative of the softmax function

Let `pᵢ = e^zᵢ / S` where `S = Σₖ e^zₖ`. We want `∂pᵢ/∂zⱼ`.

Use the quotient rule. Note `∂S/∂zⱼ = e^zⱼ`.

**Case 1: i = j**

```
∂pᵢ/∂zᵢ = ( e^zᵢ · S − e^zᵢ · e^zᵢ ) / S²
        = (e^zᵢ / S) · (1 − e^zᵢ / S)
        = pᵢ (1 − pᵢ)
```

**Case 2: i ≠ j**

```
∂pᵢ/∂zⱼ = ( 0 · S − e^zᵢ · e^zⱼ ) / S²
        = − (e^zᵢ / S)(e^zⱼ / S)
        = − pᵢ pⱼ
```

**Compact form** using the Kronecker delta (`δᵢⱼ = 1` if i = j, else 0):

```
∂pᵢ/∂zⱼ = pᵢ (δᵢⱼ − pⱼ)
```

As a matrix (the Jacobian): `J = diag(p) − p·pᵀ`.

**Sanity check:** for K = 2, `∂p₁/∂z₁ = p₁(1 − p₁)`, the same as the sigmoid derivative `σ(1 − σ)`.

### Task 4: Gradients of the cross-entropy error

Loss for one sample (one-hot `y`): `L = − Σₖ yₖ · log(pₖ)`, with `p = softmax(z)`.

**Step 1: gradient with respect to the scores z**

```
∂L/∂zⱼ = − Σₖ yₖ · (1/pₖ) · ∂pₖ/∂zⱼ
       = − Σₖ yₖ · (1/pₖ) · pₖ (δₖⱼ − pⱼ)        (from Task 3)
       = − Σₖ yₖ (δₖⱼ − pⱼ)
       = − yⱼ + pⱼ · Σₖ yₖ
       = pⱼ − yⱼ                                   (because Σₖ yₖ = 1 for one-hot y)
```

> **∂L/∂zⱼ = pⱼ − yⱼ**: *predicted probability minus actual label.*

**Step 2: gradient with respect to the weights**

Since `zⱼ = wⱼ·x + bⱼ`, by the chain rule:

```
∂L/∂wⱼ = (pⱼ − yⱼ) · x
∂L/∂bⱼ = (pⱼ − yⱼ)
```

**Step 3: over the whole dataset (matrix form)**

```
∂L/∂W = (1/n) · Xᵀ (P − Y)
∂L/∂b = (1/n) · 1ᵀ (P − Y)
```

Same shape of result as binary logistic regression (`Xᵀ(ŷ − y)`), which is why the training loop is nearly identical.

---

## 11. Summary

1. **Multiclass classification:** target with more than two classes, one label per sample.
2. Logistic regression extends to multiclass in two ways:
   - **OVR:** train K binary models (class k vs rest) and pick the highest probability. Probabilities are independent and not guaranteed to sum to 1.
   - **Softmax (multinomial):** a single model producing K scores `zk`, turned into probabilities by `e^zk / Σ e^zj`, which always sum to 1.
3. **Softmax is the generalization of sigmoid:** K = 2 reduces to the sigmoid.
4. **Softmax regression training:** one-hot encode the target, minimize **categorical cross-entropy** `−(1/n) ΣΣ yᵏ log ŷᵏ`, which for K = 2 is binary cross-entropy.
5. **Parameters:** K × (d + 1); for 3 classes and 2 features that is **9**.
6. **Gradient** of cross-entropy w.r.t. scores is simply `ŷ − y`, so the update loop looks almost the same as for binary LR.
7. **Choosing:** softmax for mutually exclusive classes, efficiency and interpretable probabilities; OVR when classes can overlap or you want a separate model per class.

```
Binary:      z ──► sigmoid ──► P(class 1)               loss: binary cross-entropy
Multiclass:  z1..zK ──► softmax ──► P(class 1..K)       loss: categorical cross-entropy
```
