# Data Leakage in Machine Learning

Notes on Data Leakage in Machine Learning and Data Science.

## Table of Contents

- [What Is Data Leakage?](#what-is-data-leakage)
- [Types of Data Leakage](#types-of-data-leakage)
- [Examples of Data Leakage](#examples-of-data-leakage)
- [How to Detect Data Leakage](#how-to-detect-data-leakage)
- [How to Remove Data Leakage](#how-to-remove-data-leakage)
- [Validation Set](#validation-set)

## What Is Data Leakage?

Data leakage, in the context of machine learning and data science, refers to a problem where information from outside the training dataset is used to create the model.

This additional information can come in various forms, but the common characteristic is that it is information that the model wouldn't have access to when it's used for prediction in a real-world scenario.

This can lead to overly optimistic performance estimates during training and validation, as the model has access to extra information.

However, when the model is deployed in a production environment, that additional information is no longer available, and the performance of the model can drop significantly. This discrepancy is typically a result of mistakes in the experiment design.

## Types of Data Leakage

The notes identify five ways in which data leakage can occur:

1. **Target Leakage:** Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
2. **Multicollinearity with Target Column**
3. **Duplicated Data**
4. **Preprocessing Leakage:** Train-test contamination and improper cross-validation.
5. **Hyperparameter Tuning**

## Examples of Data Leakage

### 1. Target Leakage: Credit Card Fraud Detection

The handwritten example includes the following terms:

- Credit card fraud detection
- `value`
- `date`
- `website`
- `reversed transaction`
- Target: `fraud`
- Labels: `1` and `0`
- `500`
- `normal → 400`
- `Precision → 80%`
- `95%`
- `train 95%`

The example illustrates how a feature such as `reversed transaction` could reveal information about the fraud target.

If this information is not available when the prediction must be made, using it can cause target leakage.

The numerical annotations are retained from the handwritten notes; the diagram does not provide enough detail to reconstruct a complete evaluation.

### 2. Smartphone Price Prediction

The handwritten diagram contains:

- Brand
- RAM
- Training
- Price

The target `price` is circled in the original notes.

Review the features carefully to ensure that the target, or information derived from it, is not included among the predictors.

### 3. Duplicated Data

The handwritten example contains:

- `1000 rows`
- `200 row duplicate`
- Train
- Model
- Test
- Accuracy

The diagram illustrates possible overlap between training and testing data.

If duplicate records appear in both sets, the test results may not represent performance on genuinely unseen data.

### 4. Preprocessing Leakage

The handwritten notes mention:

- Missing values
- Scaling
- Errors

Preprocessing leakage can happen when transformations are fitted on the entire dataset before splitting it into training and testing data.

Examples include calculating imputation values or scaling parameters using information from the test set.

### 5. Hyperparameter Tuning

The handwritten example contains:

`KNN (n_neighbors = 7 / 3 / 1)`

Other handwritten terms include:

- Metric
- Training
- Test
- Extra information
- North star

Repeatedly using the test set to select hyperparameters can influence model selection and produce an overly optimistic evaluation.

Use validation data or cross-validation for tuning and reserve the test set for final evaluation.

### 6. Mean and Age Example

The handwritten preprocessing diagram contains:

- `1000 row`
- `Age`
- `mean 27`
- Extra information
- Train
- Test rows
- Standard
- Unseen data

The example appears to illustrate calculating a mean age of `27`.

Preprocessing statistics, such as the mean used to fill missing values, should be calculated using training data only and then applied to validation and test data.

## How to Detect Data Leakage

### 1. Review Your Features

Carefully review all the features being used to train your model.

Do they include any data that wouldn't be available at the time of prediction, or any data that directly or indirectly reveals the target?

Such features are common sources of data leakage.

The handwritten notes include feature labels resembling `f1, f2, ... f3`.

### 2. Unexpectedly High Performance

If your model's performance on the validation or test set is surprisingly good, this could be a sign of data leakage.

Most predictive modelling tasks are challenging, and exceptionally high performance could mean that your model has access to information it shouldn't.

### 3. Inconsistent Performance Between Training and Unseen Data

If your model performs significantly better on the training and validation data compared to new, unseen data, this might indicate that there's data leakage.

### 4. Model Interpretability

Interpretable models, or techniques like feature importance, can help understand what the model is learning.

If the model places too much importance on a feature that doesn't seem directly related to the output, it could be a sign of leakage.

The handwritten feature importance example contains:

- `f1 → 0.13`
- `f2 → 0.81`

The value for `f2` is emphasized in the original diagram.

These values illustrate what to investigate; they do not independently prove leakage.

## How to Remove Data Leakage

### 1. Understand the Data and the Task

Before starting with any kind of data processing or modelling, it's important to understand the problem, the data, and how the data was collected.

You should understand what each feature in your data represents, and whether it would be available at the time of prediction.

### 2. Careful Feature Selection

Review all the features used in your model.

If any feature includes information that wouldn't be available at the time of prediction, or that directly or indirectly gives away the target variable, it should be removed or modified.

### 3. Proper Data Splitting

Always split your data into training, validation, and testing sets at an early stage of your pipeline, before doing any pre-processing or feature extraction.

### 4. Pre-processing Inside the Cross-Validation Loop

If you're using techniques like cross-validation, make sure to do any pre-processing inside the cross-validation loop.

This ensures that the pre-processing is done separately on each fold of the data, which prevents information from the validation set leaking into the training set.

#### Incorrect Way

```python
# Incorrect way
X_normalized = normalize(X)
cross_val_score(model, X_normalized, y, cv=5)
