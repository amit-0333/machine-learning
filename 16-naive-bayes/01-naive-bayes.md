# Session 1: Naive Bayes Intuition, Math, Code and Numerical Data

## Plan of attack
1. Intuition
2. Mathematical formulation
3. Code
4. How Naive Bayes handles numerical data
5. What if data is not Gaussian?
6. Naive Bayes on text data

---

## 1. Intuition

**Play Tennis toy dataset** (14 days, binary classification: `PlayTennis` = Yes / No)

| Outlook | Temperature | Humidity | Windy | PlayTennis |
|---|---|---|---|---|
| Sunny | Hot | High | False | No |
| Sunny | Hot | High | True | No |
| Overcast | Hot | High | False | Yes |
| Rainy | Mild | High | False | Yes |
| Rainy | Cool | Normal | False | Yes |
| Rainy | Cool | Normal | True | No |
| Overcast | Cool | Normal | True | Yes |
| Sunny | Mild | High | False | No |
| Sunny | Cool | Normal | False | Yes |
| Rainy | Mild | Normal | False | Yes |
| Sunny | Mild | Normal | True | Yes |
| Overcast | Mild | High | True | Yes |
| Overcast | Hot | Normal | False | Yes |
| Rainy | Mild | High | True | No |

(9 Yes, 5 No)

**Query point:** `W = {Sunny, Cool, Normal, True}`. Should we play tennis?

We compute two probabilities, `P(Yes | W)` and `P(No | W)`, and pick the larger one.

### Bayes theorem

```
P(A | B) = P(B | A) · P(A) / P(B)
```

Applied here:

```
P(Yes | W) = P(W | Yes) · P(Yes) / P(W)
P(No  | W) = P(W | No)  · P(No)  / P(W)
```

`P(W)` is the same for both classes, so it can be dropped when comparing.

---

## 2. Mathematical formulation

### Why the "naive" assumption is needed

`P(Sunny ∩ Cool ∩ Normal ∩ True | Yes)` requires an exact matching row in the data. No row matches, so this is **0** for both classes, which is useless.

**Naive assumption:** features are *conditionally independent* given the class, so the joint probability factorises:

```
P(W | Yes) = P(Sunny|Yes) · P(Cool|Yes) · P(Normal|Yes) · P(True|Yes)
```

### Worked example

**Priors:** `P(Yes) = 9/14`, `P(No) = 5/14`

**Likelihoods:**

| Feature | P(· \| Yes) | P(· \| No) |
|---|---|---|
| Outlook = Sunny | 2/9 | 3/5 |
| Temp = Cool | 3/9 | 1/5 |
| Humidity = Normal | 6/9 | 1/5 |
| Windy = True | 3/9 | 3/5 |

**Scores:**

```
Yes: 9/14 × (2/9 · 3/9 · 6/9 · 3/9) = 9/14 × 0.01646 ≈ 0.01058
No : 5/14 × (3/5 · 1/5 · 1/5 · 3/5) = 5/14 × 0.0144  ≈ 0.00514
```

**Normalise** (divide each by the sum ≈ 0.01572):

```
P(Yes | W) ≈ 0.67
P(No  | W) ≈ 0.33
```

**Prediction: Yes.**

> The values 0.82 / 0.18 on the board were only illustrative of the output format.

---

## 3. Code idea

**Training** = compute and store all probabilities (in a dictionary).
**Testing** = look up the probabilities for the query and multiply.

For `Outlook` (3 categories × 2 classes = 6 probabilities):

```
P(Sunny|Yes)    P(Sunny|No)
P(Overcast|Yes) P(Overcast|No)
P(Rainy|Yes)    P(Rainy|No)
```

Similarly `Temperature` has 3×2 = 6, `Humidity` 2×2 = 4, `Windy` 2×2 = 4.

Minimal sketch:

```python
import pandas as pd

def train(df, target):
    priors = df[target].value_counts(normalize=True).to_dict()
    probs = {}
    for col in df.columns.drop(target):
        for c in df[target].unique():
            sub = df[df[target] == c]
            for v in df[col].unique():
                probs[(col, v, c)] = (sub[col] == v).mean()
    return priors, probs

def predict(priors, probs, query):
    scores = {}
    for c, prior in priors.items():
        score = prior
        for col, v in query.items():
            score *= probs.get((col, v, c), 0)
        scores[c] = score
    total = sum(scores.values())
    return {c: s / total for c, s in scores.items()}

# df = pd.read_csv("playtennis.csv")
# priors, probs = train(df, "PlayTennis")
# predict(priors, probs, {"Outlook": "Sunny", "Temperature": "Cool",
#                         "Humidity": "Normal", "Windy": True})
```

---

## 4. How does Naive Bayes handle numerical data?

Example: predict `married` (Y/N) from `age` (27, 61, 52, ...). Query: age = 55.

```
P(Y | 55) ∝ P(55 | Y) · P(Y)
P(N | 55) ∝ P(55 | N) · P(N)
```

Age is not a category, so counting exact matches for `P(55 | Y)` would give 0. Instead:

1. Split the data by class.
2. For each class, compute the mean `μ` and standard deviation `σ` of age.
3. Assume age follows a **Gaussian distribution** in each class and plug into the density:

```
f(x) = 1 / (σ√(2π)) · exp( -½ · ((x - μ) / σ)² )
```

4. `f(55)` is used as `P(55 | class)`.

> `f(x)` is a *probability density*, not a true probability, but it works as a relative likelihood for comparing classes.

---

## 5. What if the data is not Gaussian?

1. **Data transformation**: apply log, square root or reciprocal to make the feature more normal.
2. **Alternative distributions**: if the data follows e.g. exponential or Poisson, use that distribution for the likelihood.
3. **Discretization**: bin values (equal width, equal frequency, or k-means) and use Categorical/Multinomial/Bernoulli NB.
4. **Kernel Density Estimation (KDE)**: non-parametric estimate of the density when the distribution is unknown.
5. **Use other models**: Decision Trees, Random Forests or SVMs make fewer distributional assumptions.

---

## 6. Naive Bayes on text data

Covered in [Session 2, Multinomial NB](../session-2-smoothing-and-variants/README.md#5-multinomial-naive-bayes). Idea: convert text to a Bag of Words (BoW) vector and treat word counts as features.
