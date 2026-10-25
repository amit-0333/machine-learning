# Session 3: Probability Distributions for Naive Bayes

## Recap
Naive Bayes variants assume different distributions for `P(x | class)`. This session builds up the distributions behind them.

---

## Definitions

| Distribution | Models | Example |
|---|---|---|
| **Bernoulli** | A single trial with 2 outcomes | Placed / not placed |
| **Binomial** | Number of successes in a fixed number of independent Bernoulli trials | Number placed out of 10 |
| **Categorical** | A single trial with **more than 2** outcomes (probabilities sum to 1) | Placed / opted out / tried but not placed |
| **Multinomial** | Counts of each category across a fixed number of categorical trials | 3 placed, 1 opted out, 6 not placed out of 10 |

---

## 1. Bernoulli distribution

> A college has a placement rate of 0.3. A random student is selected. What is the probability the student is (1) placed, (2) not placed?

```
P(placed)     = p     = 0.3
P(not placed) = 1 - p = 0.7
```

---

## 2. Binomial distribution

```
P(X = k) = C(n, k) · p^k · (1-p)^(n-k)
```

> Same college, 10 random students. Probability that:

**(1) 9 out of 10 are placed**

```
P(X = 9) = C(10,9) · 0.3⁹ · 0.7¹ = 10 × 0.0000197 × 0.7 ≈ 0.000138
```

**(2) 3 out of 10 are placed**

```
P(X = 3) = C(10,3) · 0.3³ · 0.7⁷ = 120 × 0.027 × 0.0824 ≈ 0.2668
```

```python
from scipy.stats import binom
binom.pmf(9, 10, 0.3)   # 0.000138
binom.pmf(3, 10, 0.3)   # 0.2668
```

---

## 3. Categorical distribution

> Any student has: 0.3 chance of being placed, 0.05 chance of opting out, 0.65 chance of trying but not getting placed. A random student is selected. Probability that the student:

| Question | Answer |
|---|---|
| 1. Gets placed | 0.3 |
| 2. Doesn't get placed but doesn't opt out either (tries but fails) | 0.65 |
| 3. Opts out | 0.05 |

Check: `0.3 + 0.05 + 0.65 = 1`

---

## 4. Multinomial distribution

```
P(n1, n2, ..., nk) = n! / (n1! · n2! · ... · nk!) · p1^n1 · p2^n2 · ... · pk^nk
```

> Same setup, 10 random students.

**(1) 3 placed, 1 opts out, 6 try but fail**

```
10! / (3! · 1! · 6!) · 0.3³ · 0.05¹ · 0.65⁶
= 840 × 0.027 × 0.05 × 0.0754
≈ 0.0855
```

**(2) 0 placed, 2 opt out, 8 try but fail**

```
10! / (0! · 2! · 8!) · 0.3⁰ · 0.05² · 0.65⁸
= 45 × 1 × 0.0025 × 0.0319
≈ 0.0036
```

```python
from scipy.stats import multinomial
p = [0.3, 0.05, 0.65]
multinomial.pmf([3, 1, 6], n=10, p=p)   # 0.0855
multinomial.pmf([0, 2, 8], n=10, p=p)   # 0.0036
```

---

## Why were these taught now?

They are the foundations of the Naive Bayes variants from Session 2:

| Distribution | Naive Bayes variant |
|---|---|
| Bernoulli | **BernoulliNB**: each feature is a binary variable (word present / absent) |
| Categorical | **CategoricalNB**: each feature takes one of several categories |
| Multinomial (and Binomial) | **MultinomialNB**: feature vector is a set of counts (word counts in a document) |
| Gaussian (Session 1) | **GaussianNB**: continuous features |

Knowing the distribution tells you which likelihood `P(x | class)` the model is really computing.
