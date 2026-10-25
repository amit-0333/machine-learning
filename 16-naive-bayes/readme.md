# Naive Bayes: Session Notes

Personal notes (June 2023) on the Naive Bayes algorithm, with worked examples.

| Session | Topic | Folder |
|---|---|---|
| 1 | Intuition, Bayes theorem, Play Tennis example, numerical data (Gaussian), non-Gaussian data, text data |
| 2 | Underflow, Laplace smoothing, Categorical / Bernoulli / Gaussian / Multinomial / Complement / Out-of-core NB  |
| 3 | Bernoulli, Binomial, Categorical, Multinomial distributions (why they matter for NB) | 

## Quick summary

For a query point `x = (x1, ..., xn)` and class `c`:

```
P(c | x)  ∝  P(c) · Π P(xi | c)
```

The "naive" part is assuming features are conditionally independent given the class. Pick the class with the highest score. Which variant you use depends on how `P(xi | c)` is modelled:

| Feature type | Variant | Likelihood model |
|---|---|---|
| Categorical | CategoricalNB | Categorical distribution |
| Binary (0/1) | BernoulliNB | Bernoulli distribution |
| Counts (e.g. word counts) | MultinomialNB | Multinomial distribution |
| Continuous | GaussianNB | Gaussian (normal) distribution |
