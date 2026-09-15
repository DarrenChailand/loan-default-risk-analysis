# Methodology

## 1. Problem definition

The project treats `Status` as a binary response and estimates the probability that `Status = 1` from information about a loan, borrower, property, and application. Because the source does not provide a rigorous target definition or feature-availability timeline, the output is best described as **risk classification**, not a verified production default model.

There are 148,670 observations and 34 original columns. The response is moderately imbalanced:

- `Status = 0`: 112,031 observations (75.36%)
- `Status = 1`: 36,639 observations (24.64%)

## 2. Data audit and leakage prevention

`ID` is an identifier and `year` is constant, so neither can provide generalizable predictive information. Three additional columns were removed:

| Removed feature | Missing when `Status = 0` | Missing when `Status = 1` | Reason |
|---|---:|---:|---|
| `Interest_rate_spread` | 0.00% | 100.00% | Missingness reveals the outcome |
| `rate_of_interest` | 0.00% | 99.45% | Missingness nearly reveals the outcome |
| `Upfront_charges` | 2.82% | 99.58% | Missingness nearly reveals the outcome |

Including these fields produced implausibly high scores. Their values may be recorded only after approval or another outcome-dependent stage, so they violate the rule that a feature must be available when a prediction is made.

### Residual proxy warning

`credit_type = EQUI` has 15,297 positive cases and only one negative case. This variable accounts for about 22.36% of the random forest's impurity-based feature importance. Its meaning and timing must be verified. The reusable script therefore offers `--strict`, which also excludes `credit_type` as a sensitivity check.

On the same held-out test split, strict-mode logistic regression reaches an area under the curve of 0.7475 and the strict random forest reaches 0.8856. The forest changes little, so its ranking performance is not driven by this one field alone. The large logistic-regression decline shows that its linear decision boundary depended much more heavily on the category.

## 3. Data splitting

The data is split once into 80% training data and 20% test data using `random_state=42`. Stratification preserves the response proportion in both partitions. The test set is not used during model choice.

Five-fold stratified cross-validation is run only on the training partition. Each fold is held out once while the other four folds train the complete preprocessing-and-model pipeline. This gives five estimates of every metric without contaminating the final test set.

## 4. Preprocessing

Preprocessing is placed inside each scikit-learn pipeline. This matters because imputers, scalers, and encoders are fitted only on the current training fold.

### Numeric variables

The seven numeric fields are `loan_amount`, `term`, `property_value`, `income`, `Credit_Score`, `LTV`, and `dtir1`.

- Missing values are replaced by the training-fold median.
- Logistic regression additionally standardizes each field to mean zero and unit variance.
- Random forests do not require scaling because tree splits depend on ordering rather than distance.

### Categorical variables

- Missing values are replaced by the training-fold mode.
- One-hot encoding creates one binary indicator per observed category.
- Unknown test-time categories are ignored rather than causing failure.

## 5. Supervised models

### Logistic regression baseline

Logistic regression estimates

$$
P(Y=1\mid x)=\frac{1}{1+e^{-(\beta_0+x^T\beta)}}.
$$

It is a useful baseline because it is simple and interpretable. `class_weight="balanced"` weights classes inversely to their frequencies, preventing the majority class from dominating training.

### Random forest

The random forest averages 200 decision trees. Each tree recursively partitions the feature space, while feature subsampling and bootstrap variation reduce correlation among trees. The main settings are:

- 200 trees
- maximum depth of 12
- minimum of 3 observations per leaf
- balanced class weights
- fixed random seed for reproducibility

This model can learn thresholds and interactions that a linear log-odds model cannot. Its stronger test performance suggests such structure is present, although some improvement may come from the proxy feature described above.

## 6. Evaluation

For true positives $TP$, false positives $FP$, true negatives $TN$, and false negatives $FN$:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
F_1=2\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
$$

The area under the receiver operating characteristic curve measures ranking quality across all thresholds. A value of 0.5 is random ranking and 1.0 is perfect ranking. It is useful here because it is threshold-independent, but it does not measure probability calibration or encode business costs.

### Held-out test results

| Model | Accuracy | Precision | Recall | F1 score | Area under curve |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.8314 | 0.6602 | 0.6507 | 0.6554 | 0.8425 |
| Random forest | 0.8774 | 0.7934 | 0.6793 | 0.7320 | 0.8872 |

The forest is better on every reported measure. Still, recall of 0.6793 means it misses about 32% of positive cases at the default 0.5 threshold. A real lender should choose a threshold using explicit costs and capacity constraints rather than accepting 0.5 automatically.

## 7. Stability analysis

The forest is refitted on ten stratified training-validation splits. Its area under the curve is 0.8829 ± 0.0025, and its F1 score is 0.7194 ± 0.0052. The low variation shows that results are not highly sensitive to the particular random split.

This check measures split stability, not temporal stability, fairness, data drift, or performance at another institution.

## 8. Borrower segmentation

K-means uses six numeric variables: loan amount, income, credit score, loan-to-value ratio, debt-to-income ratio, and property value. Median imputation prevents missing observations from being discarded; standardization prevents large-unit variables from dominating Euclidean distance.

For $k$ clusters, K-means minimizes within-cluster squared distance:

$$
\sum_{j=1}^{k}\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2.
$$

The number of clusters is selected from 2 through 6 using silhouette score on a reproducible sample of 10,000 rows. The selected value is $k=2$.

| Cluster | Count | Mean income | Mean property value | Mean loan-to-value | Positive rate |
|---|---:|---:|---:|---:|---:|
| 0 | 105,462 | 5,048.19 | 342,559.43 | 74.70 | 27.40% |
| 1 | 43,208 | 11,560.14 | 856,321.41 | 68.25 | 17.92% |

The labels have no inherent order or causal meaning. Principal component analysis is used only to visualize the six-dimensional clusters in two dimensions; it is not used to train the classifier.

## 9. What should be improved next

1. Confirm exactly what `Status` means and when every predictor becomes available.
2. Report strict results after excluding `credit_type`, then compare the performance drop.
3. Use a temporal holdout if application dates become available.
4. Add probability calibration and calibration curves.
5. Select the decision threshold from false-positive and false-negative costs.
6. Audit group-wise error rates and disparate impact before any consequential use.
7. Use permutation importance or SHAP values; impurity importance can favor high-cardinality or easily split features.
8. Validate on external data before making generalization claims.
