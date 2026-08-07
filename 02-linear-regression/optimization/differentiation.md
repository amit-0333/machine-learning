# Session on Differentiation — Notes

## 1. Where This Fits

```
Optimization → big picture of Calculus in ML
                        ↑
                Differentiation
                        ↑
                     Stats
```

Differentiation is the calculus tool that underpins optimization in ML.

**Roadmap of topics covered in this session:**
1. What is differentiation?
2. Why instantaneous?
3. Relation with slope
4. Maxima and Minima
5. How to calculate derivative
6. Intuition
7. Derivative in ML

---

## 2. What Is Differentiation?

> Differentiation is the process of finding the **derivative** of a function. The derivative represents the **instantaneous rate of change** of the function with respect to its variable (typically `x`).

```
diff → slope
dy/dx = 0 → maxima / minima  (a first preview of why derivatives locate turning points)
```

### Rate of change (average, over an interval)

```
y = f(x) = x²

rate of change = [f(x + Δx) − f(x)] / Δx
```

### Instantaneous rate of change (at a point)

As the interval `Δx` shrinks toward 0, the average rate of change becomes the **instantaneous** rate of change — the derivative:

```
lim   [f(x + Δx) − f(x)] / Δx
Δx→0

f(x + dx) − f(x)       df(x)
─────────────────  =   ─────      (instantaneous rate of change / derivative)
       dx                dx
```

As `Δx` (or `dx`) shrinks — e.g. from `0.1` down to `0.00000001` — the ratio converges to the exact instantaneous rate of change at that point.

### Real-world example — Population growth

```
y = f(x) = x³      (population over time, illustrative)
```
- Given population figures at two years, e.g. 1980 → 80 crore and 1990 → 100 crore:
```
growth rate ≈ (100 − 80) / (1990 − 1980) = 20 / 10 = 2 (crore/year, average rate)
```
- This is an **average** rate of change over a 10-year window — differentiation instead gives the **exact** rate of change at a single instant (e.g. exactly in 1990), not just an average.

### Real-world example — Distance, velocity, and speed

```
y = f(t) = t²        (distance as a function of time)
```
- From t=0 to t=4, distance goes from 0 to 16:
```
speed = distance / time = (16 − 0) / (4 − 0) = 4 units/time   (average speed)
```
- Velocity/speed **at a specific instant** `t = 4` is again found via the derivative — the instantaneous version of this same distance/time ratio, i.e. `dx/dt` at that point, not the average over the whole interval.

### Relation with slope

```
        f(x+dx)
         ●
        /│
       / │
      /  │ ← this rise/run, as dx→0, becomes the slope of the tangent line
     ●───┘
   f(x)
   x    x+dx
```
- `df/dx` at a point = the **slope of the curve at that point** (slope of the tangent line).
- As `dx → 0`, the secant line (through two nearby points) becomes the tangent line, and its slope is exactly the derivative at that point.

---

## 3. Maxima, Minima, and Optimization Link

```
diff = 0  →  derivative = 0 at a maxima or minima (a flat point / turning point on the curve)
```

**Worked example:**
```
y = x² + 2x

dy/dx = 2x + 2 = 0
⇒ x = −2/2 = −1
```
So the function has a turning point (minima, since the parabola opens upward) at `x = −1`.

This is exactly the same idea used in **optimization**: setting the derivative of a loss function to zero locates a maxima/minima — a critical point of the loss surface.

---

## 4. How to Calculate a Derivative — First Principles

Deriving `d/dx(x²) = 2x` step by step from the definition:

```
y = f(x) = x²

df(x)/dx = [f(x+dx) − f(x)] / dx
         = [(x+dx)² − x²] / dx
         = [x² + (dx)² + 2x·dx − x²] / dx
         = [(dx)² + 2x·dx] / dx
```
As `dx → 0`, the `(dx)²` term vanishes (it goes to 0 faster than the linear term), leaving:
```
df/dx = 2x
```

### Intuition
Think of the derivative `dy/dx` as an operator/"machine" that takes a function like `x²` and transforms it into its rate-of-change function `2x` — a new function describing how steeply the original one is changing at every point.

---

## 5. Derivative of a Constant

```
d/dx (c) = 0
```

**Why:** for `y = 5` (a horizontal line), `f(x) = 5` and `f(x+dx) = 5` as well — the function's value never changes as `x` changes, so:
```
dy/dx = [f(x+dx) − f(x)] / dx = (5 − 5)/dx = 0
```
A constant has **zero rate of change** — geometrically, its graph is a flat horizontal line, so the slope is 0 everywhere.

---

## 6. Cheatsheet — Common Derivatives

```
d/dx (x)          = 1
d/dx (sin x)       = cos x
d/dx (cos x)       = −sin x
d/dx (tan x)       = sec²x
d/dx (sec x)       = sec x · tan x
d/dx (csc x)       = −csc x · cot x
d/dx (cot x)       = −csc²x
d/dx (sin⁻¹x)      = 1 / √(1 − x²)
d/dx (cos⁻¹x)      = −1 / √(1 − x²)
d/dx (tan⁻¹x)      = 1 / (1 + x²)
d/dx (aˣ)          = aˣ · ln(a)
d/dx (eˣ)          = eˣ
d/dx (ln(x))       = 1/x ,  x > 0
d/dx (ln|x|)       = 1/x
d/dx (log_a(x))    = 1 / (x · ln(a))
```

---

## 7. Power Rule

```
d/dx (xⁿ) = n·xⁿ⁻¹
```

**Derivation for x²** (using the geometric "growing square" intuition — area of a square of side `x` growing to side `x+dx`):
```
f(x+dx) = x² + x·dx + x·dx + (dx)²
        = x² + 2x·dx + (dx)²

[f(x+dx) − f(x)] / dx = [2x·dx + (dx)²] / dx = 2x     (as dx→0, the (dx)² term drops out)
```

**Derivation for x³** (analogous "growing cube" argument):
```
f(x+dx) = x³ + x²dx + x²dx + x²dx + (higher-order dx terms)
dy/dx = 3x²
```

**Pattern (verified for several powers):**
```
x⁴ → 4x³
x⁵ → 5x⁴
x⁶ → 6x⁵
xⁿ → nxⁿ⁻¹
```

---

## 8. Sum Rule

```
(f(x) ± g(x))' = f'(x) ± g'(x)
```

**Example:**
```
d/dx (x² + log x) = dx²/dx + d(log x)/dx
```

**Visual/graphical intuition:** if `green = x² → f(x)` and `blue = x → g(x)`, then plotting `h(x) = f(x) + g(x)` gives the `red` curve, and at any point, red's height (and slope) is literally the sum of green's and blue's — i.e. `y = x + x²` is built by adding the two component functions point by point.

---

## 9. Product Rule

```
(f(x)·g(x))' = f'(x)·g(x) + f(x)·g'(x)
(c·f(x))'    = c·f'(x)           (constant multiple rule)
```

**Geometric derivation (area of a growing rectangle)**, using `y = x²·sin x`:
```
dy/dx = d(x²)/dx · sin x + x² · d(sin x)/dx
```
Worked out from first principles using an "area" argument (rectangle of sides `x²` and `sin x`, expanding to `(x+dx)` sides) and simplifying — dropping the `(dx)²` higher-order term as `dx → 0` — confirms the product rule form above.

**Simple check — `y = 5x²`:**
```
y = 5x² = 5 · d(x²)/dx = 5 · 2x = 10x
```
(constant multiple rule, consistent with product rule when one factor is a constant)

Also verified via first principles:
```
dy/dx = [ (x+dx)²·5 − 5x² ] / dx = 5(2x) = 10x
```

---

## 10. Quotient Rule

```
d/dx [f(x)/g(x)] = [f'(x)·g(x) − f(x)·g'(x)] / [g(x)]²
```

**Worked example — `y = x² / sin x`:**
```
dy/dx = [d(x²)/dx · sin x − x² · d(sin x)/dx] / (sin x)²
      = [2x·sin x − x²·cos x] / (sin x)²
```

---

## 11. Chain Rule

```
d/dx (f(g(x))) = f'(g(x)) · g'(x)
```
Also written as:
```
y = f(g(x))
dy/dx = (df/dg) · (dg/dx)
```

**Example — `y = sin(x²)`:**
```
g(x) = x²
f(g(x)) = sin(x²)

dy/dx = d[sin(x²)]/d(x²) · d(x²)/dx
      = cos(x²) · 2x
      = 2x·cos(x²)
```
Visualized as a chain of transformations: `x → x² → sin(x²)`, each link contributing its own derivative multiplicatively.

**Nested example — `y = sin(log(x²))`:**
```
x → x² → log(x²) → sin(log(x²))

dy/dx = d[sin(log x²)]/d(log x²) · d(log x²)/d(x²) · d(x²)/dx
      = cos(log x²) · (1/x²) · 2x
      = (1/x) · cos(log x²)
```

### Chain Rule in Deep Learning (Backpropagation intuition)

The chain rule is exactly how gradients flow backward through a neural network:
```
dL/dw = (dL/dŷ) · (dŷ/dw₀) · (dw₀/dw₂) · ...
```
Each layer's weight update depends on chaining together the derivative of the loss with respect to each intermediate quantity, back through the network — this is the mathematical basis of backpropagation.

---

## 12. Partial Differentiation

For a **single-variable function** `y = f(x)`, the derivative `dy/dx = f'(x)` is a 2D concept (a curve in the x–y plane).

For a **multivariable function** like `y = f(x₁, x₂)` (or more generally `f(x₁, x₂, ..., xₙ)`), the graph becomes a **surface** in 3D, and differentiation w.r.t. each variable individually (holding the others fixed) is called a **partial derivative**.

**Example:**
```
z = f(x, y) = x² + y²           (a 3D "parabola"/paraboloid surface)

∂z/∂x = 2x + 0
∂z/∂y = 0 + 2y
```

**Numeric example at the point (1, 2)** — i.e. `x = 1, y = 2`:
```
∂z/∂x = 2(1) = 2
∂z/∂y = 2(2) = 4
```
(Note: worked value shown as `−2` for `∂z/∂x` at a different sign convention in the notes — worth double-checking against your own re-derivation if it matters for an assignment.)

The two partial derivatives `∂z/∂x` and `∂z/∂y` describe the surface's slope along the x-direction and y-direction separately at a given point — this is the multivariable analogue of `β₀, β₁` in a regression-style loss surface.

---

## 13. Higher-Order Derivatives

```
y = x³
dy/dx = 3x²          (1st derivative — "first order")
d²y/dx² = 6x          (2nd derivative — rate of change of the slope itself)
d³y/dx³ = ...          (3rd derivative, and so on)
```

**Physical analogy:** `distance (s) → velocity (v) → acceleration (a)` — each is the derivative of the previous.

### Using the 2nd derivative to classify maxima/minima

- At a critical point where `dy/dx = 0`:
  - If `d²y/dx² < 0` (slope going from **positive → negative**) → point is a **maxima**
  - If `d²y/dx² ≥ 0` (slope going from **negative → positive**) → point is a **minima**

```
f'(x)  → 1st order derivative
f''(x) → 2nd order derivative → rate of change of the slope
```

This generalizes to the **Newton–Hessian method** for multivariable optimization (using second-order derivative information, i.e. the Hessian matrix, to find maxima/minima more efficiently than gradient descent alone).

---

## 14. Matrix Differentiation

Useful identities for differentiating with respect to a vector — directly underlies the OLS derivation seen in Multiple Linear Regression.

### d(Ax)/dx = A

```
For A = [[a₁₁, a₁₂], [a₂₁, a₂₂]],  x = [x₁, x₂]ᵀ

Ax = [a₁₁x₁ + a₁₂x₂ , a₂₁x₁ + a₂₂x₂]ᵀ

d(Ax)/dx = A
```
(Analogous to the scalar rule `d(cx)/dx = c` and `d(5x)/dx = 5`, just generalized to matrix form.)

### d(xᵀAx)/dx = 2xᵀA  (for symmetric A, Aᵀ = A)

```
y = xᵀAx

Expanding for the 2-variable case:
y = a₁₁x₁² + a₁₂x₂² + a₂₁x₁² + a₂₂x₂²   (simplified/combined using symmetry of A)

dy/dx = [2a₁₁x₁ + 2a₁₂x₁, 2a₁₂x₂ + 2a₂₂x₂]
      = 2·A·x                              (matrix form)
      = 2xᵀA                               (equivalent transposed/row form)
```

Quick sanity check with a scalar special case: `xᵀAx → 2x²` when reduced to 1 dimension gives `d/dx(2x²) = 4x`, consistent with the general pattern.

**Why this matters:** these two identities — `d(Ax)/dx = A` and `d(xᵀAx)/dx = 2xᵀA` — are exactly the matrix calculus rules used to derive the closed-form OLS solution `β = (XᵀX)⁻¹Xᵀy` in the Multiple Linear Regression notes.

---

## Summary

| Topic | Core takeaway |
|---|---|
| What is differentiation | Finds the instantaneous rate of change of a function at a point |
| Relation with slope | The derivative at a point = slope of the tangent line there |
| Maxima/Minima | Set derivative = 0 to find turning points; this is the seed idea behind optimization |
| First principles | `dy/dx = lim(dx→0) [f(x+dx) − f(x)] / dx` |
| Derivative of a constant | Always 0 — no change means no slope |
| Power, Sum, Product, Quotient, Chain rules | Standard toolkit for differentiating combinations of functions |
| Chain rule in ML | Directly powers backpropagation in neural networks |
| Partial differentiation | Differentiate multivariable functions one variable at a time, holding others fixed |
| Higher-order derivatives | 2nd derivative tells you about curvature — used to classify maxima vs minima, and underlies Newton's method |
| Matrix differentiation | `d(Ax)/dx = A` and `d(xᵀAx)/dx = 2xᵀA` — the exact tools behind the OLS closed-form solution |
