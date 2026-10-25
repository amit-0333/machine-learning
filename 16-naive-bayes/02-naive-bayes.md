# Session 2: Numerical Stability, Smoothing and Types of Naive Bayes

## Contents
1. [Numerical stability (underflow)](#1-numerical-stability-underflow)
2. [Laplace additive smoothing](#2-laplace-additive-smoothing)
3. [Categorical Naive Bayes](#3-categorical-naive-bayes)
4. [Bernoulli Naive Bayes](#4-bernoulli-naive-bayes)
5. [Multinomial Naive Bayes](#5-multinomial-naive-bayes)
6. [Gaussian Naive Bayes](#6-gaussian-naive-bayes)
7. [Complement and Out-of-core NB](#7-complement-naive-bayes-and-out-of-core-naive-bayes)

---

## 1. Numerical stability (underflow)

**Underflow** happens when a number gets so close to zero that the computer's floating-point representation can no longer store it accurately, and it gets rounded to 0.

Naive Bayes multiplies many small probabilities:

```
P(c) · P(x1|c) · P(x2|c) · ... · P(xn|c)
```

With hundreds of features (e.g. words), the product can underflow to 0.

**Fix: work in the log domain.** Products become sums:

```
log P(c | x) ∝ log P(c) + Σ log P(xi | c)
```

`log` is monotonic, so the class with the highest log-score is the same class.

```python
import numpy as np
print(np.prod([1e-5] * 100))        # 0.0 (underflow)
print(np.sum(np.log([1e-5] * 100))) # -1151.29 (fine)
```

---

## 2. Laplace additive smoothing

**Problem (zero-frequency):** if a feature value never appears with a class in training, its likelihood is 0 and wipes out the whole product.

**Fix:** add a small count `α` (usually 1) to every count:

```
P(xi = v | c) = (N_{c,v} + α) / (N_c + α · n_i)
```

- `N_{c,v}`: number of rows of class `c` with feature value `v`
- `N_c`: number of rows of class `c`
- `n_i`: number of distinct values of feature `i`
- `α`: smoothing parameter

### Bias-variance trade-off

| α | Effect |
|---|---|
| `α = 0` | No smoothing, fits training data exactly: low bias, **high variance** (overfitting, zero-probability problem) |
| Small `α` (e.g. 1) | Usually a good balance |
| Very large `α` | Probabilities all pushed towards uniform: **high bias**, **low variance** (underfitting) |

So `α` is a hyperparameter that controls the bias-variance trade-off.

---

## 3. Categorical Naive Bayes

Variant designed for **categorical features**. All features must be categorical.

Using the Play Tennis data (see [Session 1](../session-1-intuition-and-gaussian/README.md)). Query: `Xq = {Sunny, Hot, High, False}`

```
P(Y | Xq) ∝ P(Y) · P(Outlook=Sunny | Y) · P(Temp=Hot | Y) · ...
P(N | Xq) ∝ P(N) · P(Outlook=Sunny | N) · P(Temp=Hot | N) · ...
```

with `P(Y) = 9/14`, `P(N) = 5/14`. The product of the feature terms is the **likelihood**.

### Worked example with Laplace smoothing (α = 1)

Among the 5 "No" rows, 3 have Outlook = Sunny, and Outlook has 3 categories:

```
P(Outlook=Sunny | N) = (3 + 1) / (5 + 1·3) = 4/8 = 1/2
```

Among the 5 "No" rows, 2 have Temp = Hot, and Temp has 3 categories:

```
P(Temp=Hot | N) = (2 + 1) / (5 + 1·3) = 3/8
```

### Q: I have 4 features, 2 categorical (gender, is married) and 2 numerical (age, height). Can I use Categorical NB?

1. **Discretize the numerical features** (bin into categories like "0-18", "19-35", "36-50", "51+"). Downside: loses information.
2. **Use a mixed Naive Bayes model**: Gaussian for numerical features, categorical/multinomial for categorical features. You may need another library or to implement it yourself.
3. **Use another model** that handles mixed types natively: Decision Trees, Random Forests, Gradient Boosted Trees.

---

## 4. Bernoulli Naive Bayes

`BernoulliNB` is for data distributed according to **multivariate Bernoulli distributions**: many features, each **binary** (0/1, True/False).

- Samples must be binary feature vectors. Otherwise `BernoulliNB` can binarize the input (see the `binarize` parameter).
- Typical use: presence/absence of words in a document.
- Unlike Multinomial NB, it **explicitly penalises the absence** of a feature.

```python
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB(alpha=1.0, binarize=0.0)
```

### Worked example (Table 13.1 from Introduction to Information Retrieval)

| | docID | Words in document | in c = China? |
|---|---|---|---|
| train | 1 | Chinese Beijing Chinese | yes |
| train | 2 | Chinese Chinese Shanghai | yes |
| train | 3 | Chinese Macao | yes |
| train | 4 | Tokyo Japan Chinese | no |
| test | 5 | Chinese Chinese Chinese Tokyo Japan | ? |

Vocabulary: Chinese, Beijing, Shanghai, Macao, Tokyo, Japan.
Priors: `P(yes) = 3/4`, `P(no) = 1/4`.

Smoothed estimate: `P(t | c) = (docs in c containing t + 1) / (docs in c + 2)`

| Term | P(t \| yes) | P(t \| no) |
|---|---|---|
| Chinese | 4/5 | 2/3 |
| Beijing | 2/5 | 1/3 |
| Shanghai | 2/5 | 1/3 |
| Macao | 2/5 | 1/3 |
| Tokyo | 1/5 | 2/3 |
| Japan | 1/5 | 2/3 |

Test doc contains Chinese, Tokyo, Japan (present) and Beijing, Shanghai, Macao (absent):

```
P(yes | d5) ∝ 3/4 · 4/5 · 1/5 · 1/5 · (1-2/5)³ ≈ 0.0052
P(no  | d5) ∝ 1/4 · 2/3 · 2/3 · 2/3 · (1-1/3)³ ≈ 0.0220
```

**Prediction: not China.**

---

## 5. Multinomial Naive Bayes

Suited for **discrete features**, especially **text classification** where features are word counts (or frequencies) in documents.

```
P(w | c) = (T_{c,w} + α) / (total_c + α · B)
```

- `T_{c,w}`: count of word `w` in all documents of class `c`
- `total_c`: total count of words in class `c`
- `B`: size of the vocabulary (so `B = n` features; with `α = 1`, `α·B = n`)

### Bag of Words (BoW)

Text is converted to numeric vectors, one column per vocabulary word. Two flavours:

- **Binary BoW**: 1 if the word is present, else 0 (use with Bernoulli NB)
- **Count BoW**: how many times it occurs (use with Multinomial NB)
- TF-IDF gives fractional values and also works reasonably with Multinomial NB

Example (movie reviews → sentiment):

| great | faltu | epic | sentiment |
|---|---|---|---|
| 2 | 0 | 0 | 1 |
| 0 | 3 | 2 | 0 |
| 1 | 1 | 1 | 1 |

### Worked example (same China data)

Vocabulary size `B = 6`. Total words in class *yes* = 8, in class *no* = 3.

```
P(Chinese | yes) = (5 + 1) / (8 + 6) = 3/7
P(Tokyo   | yes) = (0 + 1) / (8 + 6) = 1/14
P(Japan   | yes) = (0 + 1) / (8 + 6) = 1/14

P(Chinese | no)  = (1 + 1) / (3 + 6) = 2/9
P(Tokyo   | no)  = (1 + 1) / (3 + 6) = 2/9
P(Japan   | no)  = (1 + 1) / (3 + 6) = 2/9
```

Test doc: `Chinese Chinese Chinese Tokyo Japan`

```
P(yes | d5) ∝ 3/4 · (3/7)³ · 1/14 · 1/14 ≈ 0.0003
P(no  | d5) ∝ 1/4 · (2/9)³ · 2/9 · 2/9   ≈ 0.00014
```

**Prediction: China (yes).** Multinomial NB and Bernoulli NB can disagree on the same data, since Multinomial uses the repeated "Chinese" counts and ignores absent words.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai",
        "Chinese Macao", "Tokyo Japan Chinese"]
y = ["yes", "yes", "yes", "no"]

vec = CountVectorizer()
X = vec.fit_transform(docs)
model = MultinomialNB(alpha=1).fit(X, y)
print(model.predict(vec.transform(["Chinese Chinese Chinese Tokyo Japan"])))
```

---

## 6. Gaussian Naive Bayes

For **continuous features** assumed to be normally distributed within each class (see [Session 1, section 4](../session-1-intuition-and-gaussian/README.md#4-how-does-naive-bayes-handle-numerical-data)).

```
P(x | c) = 1 / (σ_c √(2π)) · exp( -(x - μ_c)² / (2σ_c²) )
```

### Q: Why is Laplace smoothing not applied to Gaussian NB?

Laplace smoothing fixes **zero counts** for discrete values. Gaussian NB doesn't count occurrences; it estimates `μ` and `σ` per class, and the density is positive for any `x` (never exactly 0), so the zero-frequency problem doesn't arise. (sklearn does have a `var_smoothing` parameter, which adds a tiny value to variances for numerical stability, but that is a different idea.)

---

## 7. Complement Naive Bayes and Out-of-core Naive Bayes

*(These two were only listed as headings in the notes, so the content below is a brief supplement.)*

- **Complement NB**: variant of Multinomial NB for **imbalanced** text datasets. Instead of estimating a word's probability from class `c`, it uses statistics from the *complement* (all other classes), which makes the weights less biased towards the majority class.
- **Out-of-core NB**: for datasets too large to fit in memory. The model is trained incrementally on chunks using `partial_fit`.

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
for X_chunk, y_chunk in data_chunks:
    model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
```
