# Bias–Variance Tradeoff — Intuition & Lecture Notes

> Session notes (17 May 2023). This README follows the lecture's story: a hidden true function, three "students" with different samples, and what happens as model complexity changes.

## Contents
1. [Why this matters](#1-why-this-matters)
2. [What we will study](#2-what-we-will-study)
3. [The hidden truth](#3-the-hidden-truth)
4. [Three students, one linear model → High Bias](#4-three-students-one-linear-model--high-bias)
5. [High-degree polynomial → High Variance](#5-high-degree-polynomial--high-variance)
6. [The tradeoff](#6-the-tradeoff)
7. [The bullseye picture](#7-the-bullseye-picture)
8. [Cheat sheet](#8-cheat-sheet)
9. [Questions to answer](#9-questions-to-answer)

---

## 1. Why this matters
Every ML model fails in one of two ways: it is **too simple** to capture the pattern, or it is **too flexible** and memorises noise. The bias–variance tradeoff explains both, and it is the reason we care about underfitting, overfitting, regularisation and model selection.

## 2. What we will study
- What bias and variance are (intuitively, then mathematically)
- How they relate to **underfitting** and **overfitting**
- Why reducing one tends to increase the other
- Bias–variance decomposition of error
- A code example

## 3. The hidden truth
In real problems the data-generating process is unknown. In this lecture we *pretend to know it*:

```
y = f(x) = x²        for x in [-15, 10]
```

The real-world (population) data is the true function **plus random error**:

```
y = x² + error       (population of ~1000 points)
```

So the points scatter around the parabola, but the parabola is the real signal.

## 4. Three students, one linear model → High Bias
Draw **3 random samples** from the population:

| Sample | Marker | Student |
|---|---|---|
| Train set 1 | blue circle | Student 1 |
| Train set 2 | orange triangle | Student 2 |
| Train set 3 | green square | Student 3 |

Each student fits a **linear regression** to their own sample. Result:

- All three lines look **very similar** to each other (the model barely changes when the data changes).
- But all three are **far from the parabola** (a straight line cannot bend).

**Definition used in class:** *Bias is the inability of an ML model to fit the training data.*

**Variance (class definition):** how the ML model's predictions change when the training data changes.

→ This case is **High Bias + Low Variance** = **Underfitting**.

## 5. High-degree polynomial → High Variance
Now each student fits a **high-degree polynomial** (the lecture explores degrees 2, 3, 4 … up to ~25).

- The curve passes through almost every training point, including the noise.
- Each student's curve looks **completely different**.
- Example from the lecture: at x ≈ −11, the three models predict roughly **220, 130 and 80**, for the same input.

→ This case is **Low Bias + High Variance** = **Overfitting**.

A polynomial of the right degree (the lecture's sketch marks degree 3; since the truth is x², degree 2 is the exact match) gives **Low Bias + Low Variance**, the ideal.

## 6. The tradeoff

> Minimising bias will usually increase variance, and vice versa.

| Model complexity | Bias | Variance | Fit |
|---|---|---|---|
| Low (linear) | High | Low | Underfitting |
| Just right | Low | Low | Good generalisation |
| High (degree 25) | Low | High | Overfitting |

As complexity grows: **bias ↓, variance ↑**. As complexity shrinks: **variance ↓, bias ↑**.

On the classic plot (x-axis = model complexity, y-axis = error):
- **Bias curve** falls from left to right.
- **Variance curve** rises from left to right.
- **Generalisation error** (red) is U-shaped, with its minimum where the two curves balance.
- Left of the minimum = **underfitting zone**; right = **overfitting zone**.

## 7. The bullseye picture
The centre of the target is the true value.

|  | **Low variance** | **High variance** |
|---|---|---|
| **Low bias** | Tight cluster on the bullseye (ideal) | Scattered around the bullseye |
| **High bias** | Tight cluster, off-centre | Scattered and off-centre (**worst**) |

## 8. Cheat sheet
- **High bias** → underfitting → poor on train *and* test.
- **High variance** → overfitting → great on train, poor on test (e.g. train 90%, test 80%).
- **Goal:** low bias *and* low variance.
- **Irreducible error:** noise in the data that no model can remove.

## 9. Questions to answer
1. How would you define bias and variance mathematically?
2. How are bias and variance related to overfitting and underfitting mathematically?
3. Why is there a tradeoff between bias and variance mathematically?

➡️ These are answered in **README_2_Math_and_Code.md**.
