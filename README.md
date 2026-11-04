# Bank Marketing Subscription Prediction

<p align="center">
  <strong>ES 335 · Machine Learning Course Project</strong>
</p>

<p align="center">
  <a href="https://23110350.github.io/Bank-Marketing/">
    <strong>🌐 View the Live Project Website</strong>
  </a>
</p>

---

## Overview

This project studies whether information available before a bank marketing call can help predict whether a customer will subscribe to a term deposit. It uses the UCI Bank Marketing dataset to compare a majority-class baseline with four fixed supervised classifiers and a separately tuned Random Forest variant.

The analysis focuses on the challenges of an imbalanced classification problem: a model can achieve high overall accuracy while failing to identify customers who actually subscribe. For that reason, the comparison considers precision, recall, F1-score, balanced accuracy, and ROC-AUC alongside accuracy.

## Research Question

**How effectively can classification models predict term-deposit subscription using customer and campaign information available before a marketing call, and how does the classification threshold affect the precision–recall trade-off?**

## Dataset

The project uses the **Bank Marketing `bank-full.csv` dataset**, containing 45,211 observations. The target variable is `y`, which indicates whether a customer subscribed to a term deposit.

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Observations:** 45,211
- **Target:** `y` (`yes` / `no`)
- **Prediction task:** Binary classification

### Data-handling decisions

- The `duration` feature is excluded because call duration is only known after the call has ended. Including it would make the model unsuitable for a decision that must be made before contacting the customer.
- The `pdays = -1` value is retained to represent the absence of a previous contact. A separate indicator is used to distinguish customers with and without a previous contact.
- The target classes are imbalanced, so accuracy is not treated as the sole measure of model quality.

## Methodology

The experiment is implemented in `analysis/reproduce_analysis.py`.

1. A stratified 80:20 train–test split is used.
2. Three-fold stratified cross-validation is performed on the training partition.
3. Preprocessing is fitted within scikit-learn pipelines, so transformations are learned from the relevant training fold rather than the validation or test data.
4. A majority-class baseline is compared with Logistic Regression, Decision Tree, Random Forest, and XGBoost.
5. Class weighting is used where supported to address class imbalance.
6. The held-out test set is reserved for final model comparison and uncertainty reporting.

The browser-based threshold explorer uses **out-of-fold (OOF) predictions generated from the training partition** (`outputs/validation_oof_predictions.json`). Each training observation receives a prediction from a fold model that did not train on that observation. This supports threshold-sensitivity analysis without exposing the held-out test set to interactive threshold choices. The held-out test predictions (`outputs/test_predictions.json`) are reserved for final evaluation and are not used by the explorer. The browser recalculates classification metrics for the selected threshold; it does not train models.

## Models Compared

- **Majority-class baseline** — predicts the most frequent class for every observation.
- **Logistic Regression** — linear classification baseline.
- **Decision Tree** — rule-based nonlinear classifier.
- **Random Forest** — ensemble of decision trees.
- **XGBoost** — gradient-boosted tree ensemble.

## Results

The held-out test set contains 9,043 observations: 7,985 non-subscribers and 1,058 subscribers.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8830 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.7553 | 0.6983 | 0.2667 | 0.6238 | 0.3736 | 0.7722 |
| Decision Tree | 0.7364 | 0.6666 | 0.2394 | 0.5756 | 0.3381 | 0.7175 |
| Random Forest | 0.8251 | **0.7300** | 0.3549 | 0.6059 | **0.4476** | **0.7976** |
| XGBoost | **0.8940** | 0.5730 | **0.7181** | 0.1541 | 0.2537 | 0.7807 |
| Tuned Random Forest | 0.8443 | 0.7261 | 0.3878 | 0.5718 | **0.4622** | **0.8014** |

### Key findings

- The majority baseline achieves 88.30% accuracy simply by predicting the non-subscription class, but it identifies no subscribers. This demonstrates why accuracy alone is misleading for this dataset.
- The tuned Random Forest achieves the highest subscriber-class F1-score (0.4622) and ROC-AUC (0.8014) among the evaluated models; the fixed Random Forest remains a strong baseline.
- XGBoost achieves the highest accuracy (89.40%) and precision (0.7181), but its subscriber recall is only 0.1541 at the default threshold.
- Logistic Regression has higher subscriber recall (0.6238) than Random Forest, but lower precision and F1-score.

A training-only three-fold grid search also evaluates a tuned Random Forest. It reaches test F1 0.4622 and ROC-AUC 0.8014 in this split. The fixed Random Forest's bootstrap 95% intervals are 0.425–0.469 for F1 and 0.782–0.813 for ROC-AUC. The project also reports descriptive subgroup error rates; these are diagnostics, not causal findings.

The preferred model therefore depends on the campaign objective. If missing likely subscribers is especially costly, recall may receive greater weight; if contacting unlikely subscribers is costly, precision may matter more. The threshold explorer illustrates this trade-off.

## Project Website

The interactive website presents the project background, dataset, exploratory analysis, preprocessing, model explanations, results, and conclusion.

**[Open the live Bank Marketing ML project website](https://23110350.github.io/Bank-Marketing/)**

## Repository Structure

```text
Bank-Marketing-ML-Project/
├── analysis/
│   └── reproduce_analysis.py
├── data/
│   └── bank-full.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── Reproduce_Experiment.ipynb
│   └── archive/                 # Historical model-specific notebooks
├── outputs/
│   ├── model_comparison.csv
│   ├── test_predictions.json
│   ├── validation_oof_predictions.json
│   ├── random_forest_grid_search.csv
│   ├── test_metric_bootstrap_ci.csv
│   ├── random_forest_error_by_group.csv
│   └── experiment_metadata.json
├── scripts/
│   └── sync_website.py
├── website/                     # Website source, assets and notebook copies
├── docs/                        # GitHub Pages publishing copy (main → /docs)
├── requirements.txt
└── README.md
```

`docs/` contains the published website and is the GitHub Pages deployment directory. The root `notebooks/Reproduce_Experiment.ipynb` is the current notebook entry point. Earlier model-specific notebooks are retained under `notebooks/archive/` for historical reference and are not the source of the reported results.

## Reproducing the Analysis

### Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

### Steps

1. Clone or download this repository.
2. From the repository root, create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Confirm that `data/bank-full.csv` is present. If needed, obtain the dataset from the [UCI Bank Marketing page](https://archive.ics.uci.edu/dataset/222/bank+marketing).
5. Run the experiment:

   ```bash
   python analysis/reproduce_analysis.py
   ```

6. Review the generated comparison and prediction files in `outputs/`.

The primary notebook, `notebooks/Reproduce_Experiment.ipynb`, provides a notebook-based entry point to the experiment.

## Individual Defense Preparation

This section is a preparation aid for explaining the project's decisions. Do not memorize it word for word. Be ready to explain the ideas in your own words and point to the relevant code, results, and limitations.

### 1. Why is `duration` excluded?

The project frames prediction as something made **before a marketing call**. Call duration is only known after the call ends and is strongly associated with the outcome. Including it would create target leakage for the intended use case because the model would rely on information unavailable when the prediction is supposed to be made. Excluding it makes the task more realistic, even if the measured predictive performance is lower.

### 2. Why is accuracy misleading?

The held-out test set contains 7,985 non-subscribers and 1,058 subscribers. Predicting "no" for everyone gives about 88.3% accuracy while identifying **zero** subscribers. Accuracy is dominated by the majority class, so the project also reports precision, recall, F1-score, balanced accuracy, ROC-AUC, and average precision.

### 3. What do false positives and false negatives mean here?

- **False positive (FP):** the model predicts subscription, but the customer does not subscribe. A campaign may spend contact time or resources on someone who does not convert.
- **False negative (FN):** the model predicts no subscription, but the customer would have subscribed. The campaign may miss a potentially valuable customer.

Neither error is automatically worse. Their relative cost depends on contact cost, campaign capacity, subscription value, and customer experience. This project describes the trade-off but does not have the economic data needed to assign monetary costs or claim a business-optimal threshold.

### 4. Why report precision, recall, and F1 together?

Precision asks: "Of customers predicted to subscribe, how many actually subscribed?" Recall asks: "Of all actual subscribers, how many did the model identify?" Raising the threshold usually makes positive predictions more selective, often increasing precision while reducing recall. F1 is the harmonic mean of precision and recall. It is a useful summary when both matter, but it does not encode the bank's actual costs.

### 5. Why use average precision for Random Forest tuning?

Subscribers are the minority class, and the project needs to assess how well models identify and rank likely subscribers. Average precision summarizes the precision–recall curve across thresholds and focuses on positive-class retrieval, making it more informative than accuracy for this imbalanced task. The grid search uses training-only cross-validation to select hyperparameters. Average precision is a modelling choice, not proof of business value; a cost-based objective could select a different model.

### 6. Why compare these particular models?

The models provide complementary assumptions and complexity:

- **Majority baseline:** establishes the performance of a trivial majority-class prediction.
- **Logistic Regression:** a relatively simple linear baseline.
- **Decision Tree:** captures nonlinear splits and interactions in a rule-based model.
- **Random Forest:** averages many trees to reduce the instability of a single tree and capture nonlinear patterns.
- **XGBoost:** uses sequential boosting to build a more complex tree ensemble.

This is a practical comparison, not an exhaustive search of every possible algorithm.

### 7. Why is Random Forest selected, and what does "best" mean?

The tuned Random Forest has the highest F1-score (0.4622) and ROC-AUC (0.8014) among the reported models on this held-out split. It is a reasonable choice if the evaluation emphasizes the precision–recall balance and ranking discrimination. It is not best on every metric: XGBoost has higher accuracy and precision, while Logistic Regression has higher recall. The operational choice depends on campaign costs and capacity.

### 8. What is cross-validation, and why use it?

The training data is divided into three stratified folds. In each round, the model is fitted on two folds and evaluated on the remaining fold; the held-out fold rotates. Stratification helps preserve class proportions. Cross-validation supports model comparison and hyperparameter selection without using the final test set. The held-out test partition remains separate for final reporting.

### 9. What is the difference between OOF and test predictions?

**Out-of-fold (OOF) training predictions** are generated so that each training observation is predicted by a fold model that did not train on that observation. They power the website threshold explorer. **Test predictions** come from the final fitted model on the untouched test partition and are reserved for final evaluation. Keeping these roles separate reduces the risk of adapting choices to the test set.

### 10. Why is the tuned Random Forest not in the threshold explorer?

The Random Forest hyperparameters were selected using cross-validation on the training data. Reusing those selected hyperparameters to generate ordinary OOF predictions for the same observations can make threshold analysis optimistic because model selection has already used those folds. The explorer therefore uses the documented OOF predictions for the original fixed candidate models rather than presenting a potentially selection-biased tuned-model curve as independent.

### 11. What do the bootstrap confidence intervals tell us—and not tell us?

The project resamples held-out test predictions and outcomes with replacement to estimate intervals for selected metrics. These intervals describe sampling variability conditional on this particular split and the fitted models. They do **not** capture all uncertainty from a different train–test split, repeating model selection, a different data period, or a different customer population. They are not a guarantee of future performance.

### 12. How should subgroup error analysis be interpreted?

The subgroup table reports descriptive false-positive and false-negative counts or rates across categories such as contact type and age band. It can identify where errors are concentrated and motivate further investigation. It does not prove why those errors occur, establish causality, or by itself demonstrate fairness or unfairness. Consider subgroup sizes and denominators alongside rates.

### 13. What are the main generalisation limitations?

The evaluation uses one stratified 80:20 split from one historical dataset. The bootstrap intervals are conditional on that split. Campaign practices, customer populations, and relationships can change over time. Stronger evidence would require repeated or time-based validation, evaluation on a later or external dataset, and monitoring after deployment. The current results should not be assumed to transfer automatically to another bank, period, or campaign.

### 14. What would be needed before real-world use?

A bank would need to define the objective and relative costs of FP and FN errors, choose a threshold using training/validation data rather than the final test set, assess calibration and subgroup performance, test on a later period, address privacy and regulatory requirements, and monitor drift. A controlled pilot or A/B test would be needed to determine whether model-guided targeting improves outcomes. Predictive performance alone does not establish causal impact or financial return.

### A concise project explanation

> "We framed the task as predicting subscription before a marketing call, so we excluded call duration because it is only known afterward. The data is imbalanced, which is why the majority baseline's 88.3% accuracy is not meaningful on its own. We compared a simple linear model with tree-based models, used training-only cross-validation to tune Random Forest using average precision, and kept the test set for final evaluation. The tuned Random Forest gives the strongest F1 and ROC-AUC in this split, but the best operational choice depends on the relative costs of missed subscribers and unnecessary contacts. OOF predictions power the threshold explorer, while held-out test predictions remain reserved for final reporting. One historical split cannot establish performance on future campaigns or other populations."

**Defense advice:** Explain the metrics in your own words, locate the corresponding implementation and output, and state what the experiment does *not* prove. If asked for a business-optimal threshold or monetary error cost, explain that the required campaign cost and value data were not available rather than inventing figures.

---

## Limitations

- The results measure predictive performance on a held-out split from this dataset; they do not establish causal relationships.
- Performance may change with a different train–test split, data period, customer population, or campaign strategy.
- The experiment uses a fixed dataset and does not evaluate deployment-time drift, operational costs, or the financial return of a targeting campaign.
- Precision and recall depend on the classification threshold. A threshold suitable for one campaign objective may not be suitable for another.

---

## Project Information

**Course:** ES 335 — Machine Learning  
**Institution:** Indian Institute of Technology Gandhinagar
