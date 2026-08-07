# Session on Optimization — Notes

*(21 April 2023)*

## 1. Where Optimization Fits (Big Picture)

```
Stats → Prob → Linear Alg  ─┐
                             ├─→ Calculus → Optimization
                Approx ──────┘        ↑
                                      ML

ML / DL
```

Optimization sits at the intersection of calculus and the math foundations (stats, probability, linear algebra) — it's the machinery that lets ML and DL models actually **learn** from data.

---

## 2. Functions

> A function is a mathematical rule that takes an input value, processes it according to a specific formula or set of instructions, and produces a unique output value — each input maps to exactly one output.

```
input → [rule] → output

y = f(x)   →   "y is a function of x"
```

**Example:**
```
y = f(x) = x² + 2x
f(2) = (2)² + 2(2) = 8
```
For x = 2, 1, 3, 4, 5, ... this produces a sequence of outputs (e.g. f(1) = 3, f(3) = 15, ...). This is a **single-variable function**, f(x).

### Multivariable Functions

```
z = f(x, y) = x² + y²          (2 inputs → 1 output, "2→1")
```
- `z` → output, `x, y` → inputs
- Plotting this gives a 3D bowl-shaped surface.
- Example: x=1, y=1 → z=2. x=1, y=2 → z≠3 (i.e. output isn't simply additive).
- General form for n inputs: `f(x₁, x₂, x₃, ..., xₙ)` — this lives in **(n+1)-dimensional** space (n input dimensions + 1 output dimension).

---

## 3. Parameters in a Function

> Parameters are the variables that define a function's behaviour — the constants/coefficients in its formula that determine how inputs are processed. In `f(x) = ax² + bx + c`, `a`, `b`, `c` are the parameters; changing them reshapes/repositions the parabola.

**Distinguishing parameter vs. input**, using `y = f(x) = 5x²`:
- `x` → input
- `5` → parameter

Changing the parameter (e.g. `5x²` vs `6x²` vs `9x²`) changes the shape of the curve, even though it's still "the same function template" `ax²`.

---

## 4. ML Models as Mathematical Functions

Any ML model can be thought of as a function mapping inputs to an output.

**Example — Placement prediction:**

| CGPA | IQ | Placement |
|---|---|---|
| 9 | 90 | 9 |
| 10 | 100 | 10 |
| 8 | 80 | 8 |

```
y → placement (output)
x → cgpa, iq (inputs)

placement = f(cgpa, iq)
```

---

## 5. Parametric vs. Non-Parametric ML Models

| | Parametric | Non-Parametric |
|---|---|---|
| Assumption | Makes an assumption about the data distribution | No such assumption |
| # of parameters | **Fixed**, irrespective of the number of rows/data points | Grows with the data |
| Examples (from notes) | Linear regression, ANN (artificial neural nets) | KNN, k-RNN, decision trees |

**Linear Regression example:**
```
y = mx + b            (m, b → parameters, fixed count regardless of data size)
```
Given `cgpa | iq → placement`, the model is:
```
placement = f(cgpa, iq)
y = β₀ + β₁x₁ + β₂x₂
```
Here `β₀, β₁, β₂` are the model's parameters — fixed at 3 values no matter how many rows (students) are in the dataset.

**Worked numeric example:** if `β₀=1, β₁=2, β₂=3` and a student has cgpa=8, iq=80:
```
y = 1 + 2(8) + 3(80) = 1 + 16 + 240 = 257
```

**Non-parametric models** (e.g. a decision tree or KNN) don't have this fixed-parameter-count property — their effective complexity grows with the training data.

---

## 6. Linear Regression as a Parametric ML Model

- The model starts with **some random initial values** for `β₀, β₁, β₂`.
- Using these, it produces a prediction `ŷ` for each row, compared against the true `y`.

Example:
| y (actual) | ŷ (predicted) |
|---|---|
| 8 | 11 |
| 6 | 3 |
| 7 | 8 |

- The gap between `β` (current parameter values) and the "correct" parameter values `A` is closed using a **loss function** — an iterative process of adjusting β to reduce that gap.

---

## 7. Loss Function

> A loss function (a.k.a. cost function / objective function) measures the difference between predicted output and actual target values. Training an ML model means **minimizing** this loss function — which corresponds to improving the model's predictions.

**Why not just sum the raw differences `(y − ŷ)`?**

| y | ŷ | y − ŷ |
|---|---|---|
| 8 | 11 | −3 |
| 9 | 6  | 3 |
| 6 | 3  | 3 |

If you just sum `(y - ŷ)` directly, positive and negative errors can **cancel out** (e.g. −3 + 3 + 3 ≠ a fair measure of total error) — so we need something that avoids cancellation.

**Fix:** use absolute value or square:
```
Σ|y − ŷ|          (absolute error)
Σ(y − ŷ)²         (squared error)  ← more common, since it's smooth/differentiable everywhere
```

This connects directly to **optimization**: the loss surface (e.g. plotting `(y − ŷ)²`) has a minimum, found where its derivative ("diff") is zero — same bowl/parabola idea as before.

### How to Select a Good Loss Function

1. **Problem type** — regression → MSE/MAE; binary classification → cross-entropy/hinge loss; multi-class → categorical cross-entropy/multi-class hinge loss. Pick what matches the task.
2. **Robustness to outliers** — MSE is sensitive to outliers (squares amplify large errors); MAE or Huber loss are more robust if the data is noisy.
3. **Interpretability & ease of use** — simpler loss functions (MSE, cross-entropy) are easier to understand, compute, and differentiate.
4. **Differentiability** — most optimizers (e.g. gradient descent) need continuous first-order derivatives to compute gradients.
5. **Compatibility with the model** — e.g. linear regression assumes Gaussian noise, which is exactly why MSE is the natural fit for it.

---

## 8. Calculating Parameters From a Loss Function — "The Easy Way" (and its problem)

**Setup:**

| cgpa | iq | placement (y) | ŷ |
|---|---|---|---|
| 8 | 80 | 8 | 11 |
| 7 | 70 | 7 | 6 |
| 6 | 60 | 6 | 3 |

Loss function (mean squared error form):
```
L(β₀, β₁, β₂) = argmin (1/n) Σᵢ (yᵢ − ŷᵢ)²
                 β₀,β₁,β₂
```

**"Easy way" — solve for one parameter at a time.**
Fix `β₁ = 1, β₂ = 2` (arbitrary starting values), and treat `L` purely as a function of `β₀`:
```
ŷᵢ = β₀ + β₁x₁ + β₂x₂
yᵢ = β₀ + x₁ + 2x₂        (with β₁=1, β₂=2 plugged in)

L = Σᵢ (yᵢ − β₀ − x₁ − 2x₂)²
```
Plotting `L` vs. `β₀` alone gives a **parabola** (convex, one minimum) — because with β₁, β₂ held fixed, L becomes a simple quadratic in β₀ alone. At the minimum, **slope = 0**.

Setting the derivative to zero and solving:
```
dL/dβ₀ = Σᵢ −2(yᵢ − β₀ − x₁ − 2x₂) = 0
⇒ Σ(yᵢ − x₁ − 2x₂) = β₀ · n     (roughly; solving gives a direct formula for β₀)
```

This gives a **direct/global/closed-form solution** for β₀ — but only works cleanly when solving for one parameter at a time with the others held fixed ("limited scenarios").

### Problem with the Easy Way

1. **Non-convexity**: the loss function isn't always convex — it can have multiple local minima/maxima. Setting the gradient to zero may land you on a local (not global) minimum.
2. **Complexity**: for some models the analytical (closed-form) solution is expensive or impossible to compute directly — this is exactly the issue behind the OLS formula `β = (XᵀX)⁻¹XᵀY` for real-time / large-scale settings.
3. **Scalability**: with massive data or high-dimensional features, computing the closed-form solution is computationally prohibitive.
4. **Online learning / streaming data**: when data arrives continuously rather than all at once, an analytical solution isn't practical — models need to update incrementally. Gradient descent (and variants like stochastic gradient descent) are well suited for this.

---

## 9. Convex vs. Non-Convex Loss Functions

- **Convex function**: a line segment connecting any two points on the curve stays **above** the curve. Has only **one global minimum**.
- **Non-convex function**: this property fails — the curve can have multiple local minima/maxima, and a line between two points may dip below the curve at some spots.

This matters directly for optimization: convex loss functions guarantee that "gradient = 0" finds *the* global minimum; non-convex ones don't.

---

## 10. Gradient Descent

### Where it fits among optimization techniques
```
Optimization techniques:
 - Linear regression   ─┐
 - Logistic regression   ├─ dL/dβ₀  (first-order derivative, f'(x))
 - Perceptron           ─┘
 - Deep learning (ANN, KNN, RNN)

 SVM, PCA → quadratic programming (different family of optimization)
```
Second-order derivative also referenced: `f''(x) = d²L/dβ²`.

### The Update Rule

For a single parameter β₀, starting at some point on the loss curve:

```
┌───────────────────────────────┐
│  β₀(new) = β₀(old) − η · dL/dβ₀ │
└───────────────────────────────┘
```
- `η` (eta) = **learning rate**
- `dL/dβ₀` = the gradient (slope) of the loss at the current β₀

**Worked numeric example from the notes:**
- Start at `β₀ = 2`, where `dL/dβ₀ = −10`
- Update: `β₀_new = 2 − (0.01)(−10) = 2 + 0.1 = 2.1`
- Next iteration: `β₀ = 2.1 + 0.01 × 9 = 2.19` (gradient shrinking as it approaches the minimum, which is around β₀ = 5 in the sketched curve)

Intuition: if the slope is negative, moving in the negative-gradient direction increases β₀ (moves right, toward the minimum); if the slope is positive, β₀ decreases (moves left). This is why the formula subtracts `η × gradient`.

### Local Minima, Momentum, and Advanced Optimizers

On a non-convex (wavy) loss surface, plain gradient descent can get stuck in a **local minimum** rather than reaching the global one. This motivates more advanced optimizers used especially in **deep learning / neural nets**:
- **Momentum**
- **NAG** (Nesterov Accelerated Gradient)
- **RMSProp**
- **Adam**

These help gradient-based methods escape shallow local minima / navigate tricky loss landscapes more effectively than vanilla gradient descent.

### Epochs

- An **epoch** = one full pass of the gradient-descent update process. Repeated epochs move the parameter progressively closer to the minimum along the convex curve.

**Worked mini-example (single-variable):**
```
f(β) = β² + 2
Start: β = 5 → f(5) = 27
After one update step: β = 4.9 → f(4.9) = (4.9)² + 2 = 26
```
This is the general parameter-update pattern for any parameter:
```
βₙ = β₀ − η (∂L/∂β)
```

---

## 11. Gradient Descent with Multiple Parameters

For a model with parameters `β₀, β₁, ..., βₙ` (an `(n+1)`-length vector, since OLS-style regression has `n` columns/features plus the intercept), the loss surface `L(β₀, β₁)` in 2 parameters looks like a **bowl** in 3D (higher-dim analogue of the 2D parabola).

**Update rule — one equation per parameter, applied simultaneously:**
```
β₀ = β₀ − η (∂L/∂β₀)
β₁ = β₁ − η (∂L/∂β₀)   [general form: ∂L/∂β₁]
β₂ = β₂ − η (∂L/∂β₂)
   ⋮
```

**The Gradient Vector:**
```
∇L(β₀, ..., βₙ) = ( ∂L/∂β₀ , ∂L/∂β₁ , ... , ∂L/∂βₙ )
```
This vector of partial derivatives points in the direction of steepest ascent; gradient descent moves in the **opposite** direction (steepest descent) to minimize the loss, exactly like the single-parameter case but now across every parameter simultaneously.

---

## 12. Problems Faced in Optimization (Beyond the Basics — mainly Deep Learning context)

1. **Non-convexity**: loss functions for models like ANNs have complex landscapes with multiple local minima, maxima, and **saddle points**, making it hard to find the true global minimum.
2. **Ill-conditioning**: gradients in different dimensions can have very different magnitudes, causing gradient descent to oscillate/zig-zag and converge slowly (sketched as a narrow, elongated bowl where updates bounce between the steep walls instead of heading straight to the minimum).
3. **Vanishing / exploding gradients**: in deep networks, gradients can shrink toward zero or blow up as they propagate through layers — leading to slow or unstable training.
4. **Overfitting**: optimizing the loss too well on training data can cause the model to memorize noise rather than learn the underlying pattern, hurting performance on unseen data.
5. **Scalability**: with large numbers of features/instances/parameters, optimization gets computationally expensive. The notes give a concrete scale example: a neural network with **(m, b)-style parameters** across, say, **10 layers**, can have on the order of **60 billion** parameters total — illustrating just how large-scale this problem gets for deep learning.

---

## 13. Other Optimization Techniques (Constrained Optimization)

Beyond plain gradient descent (`gradient → dₛ`, i.e. derivative-based descent), some models involve **constrained optimization** — minimizing a loss subject to a constraint:

```
argmin  (y − ŷ)²      subject to a constraint, e.g. ŷ = 2
  β₀
```

- **SVM** (Support Vector Machines) and **PCA** (Principal Component Analysis) are given as examples of ML techniques that rely on **constrained optimization** (e.g. quadratic programming) rather than plain unconstrained gradient descent.

---

## Summary

| Section | Core Idea |
|---|---|
| Functions & parameters | ML models are just functions; parameters are the tunable constants inside them |
| Parametric vs non-parametric | Parametric = fixed # params (Linear Reg, ANN); Non-parametric = grows with data (KNN, trees) |
| Loss function | Measures prediction error; must avoid cancellation (use squared/absolute error) |
| "Easy way" & its problems | Solving one parameter at a time via closed-form works only in limited cases — breaks down with non-convexity, scale, and streaming data |
| Convex vs non-convex | Convex → one global minimum; non-convex → possible local minima/maxima traps |
| Gradient Descent | Iteratively updates parameters: `β_new = β_old − η·(∂L/∂β)`, generalizes to a vector of parameters via the gradient ∇L |
| Optimization challenges | Non-convexity, ill-conditioning, vanishing/exploding gradients, overfitting, scalability |
| Constrained optimization | SVM, PCA use constraint-based optimization (e.g. quadratic programming) rather than plain gradient descent |
