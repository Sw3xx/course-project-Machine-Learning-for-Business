# RepayPro | Machine Learning for Business

**Loan Repayment Prediction · Credit Risk Analytics · Business Decision Support**

RepayPro is an academic machine learning project exploring how predictive models can support data-informed decisions in consumer lending. The project benchmarks five classification algorithms for predicting whether a loan will be repaid and translates model performance into practical risk-management considerations.

> **Project objective:** Evaluate predictive performance while considering the asymmetric business costs of misclassifying borrowers.

## At a Glance

| Dimension | Details |
| --- | --- |
| Business problem | Loan repayment prediction / credit risk screening |
| Target variable | `loan_paid_back` |
| Models compared | Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP |
| Techniques | Data preprocessing, supervised classification, feature analysis, model evaluation |
| Core stack | Python, pandas, NumPy, scikit-learn, XGBoost |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |

## Business Context

Lenders must balance credit accessibility with the costs of missed repayment risk. A predictive model can provide *decision support* in three stages:

- **Before lending:** Identify cases that warrant additional review.
- **During the loan:** Inform risk monitoring and prioritization.
- **After lending:** Support portfolio analysis and learning from outcomes.

Model predictions should be reviewed alongside policy constraints, fairness assessments, and human judgment—not treated as automatic lending decisions.

## Modeling Approach

1. **Prepare features and target.** Work with a cleaned, encoded tabular dataset and binary target `loan_paid_back`.
2. **Establish baselines.** Evaluate Logistic Regression and Decision Tree classifiers.
3. **Test stronger alternatives.** Compare Random Forest, XGBoost, and a Multi-Layer Perceptron (MLP).
4. **Evaluate trade-offs.** Examine Accuracy, Precision, Recall, F1-score and, as discussed in the course report, ROC-AUC.
5. **Interpret the results.** Explore feature relationships and discuss business implications, including the different costs of false positives and false negatives.

## Selected Results

The **course report** identifies **XGBoost** as the strongest overall model on several reported metrics:

| Reported metric | XGBoost |
| --- | ---: |
| Accuracy | 0.9030 |
| Precision | 0.9100 |
| Recall | 0.9020 |
| ROC-AUC | 0.8944 |

The course report also identifies **Decision Tree** as the model with the highest reported F1-score (**0.8851**).

These figures are transcribed from the academic report. They are **not** independently reproduced in this repository and may use reporting conventions that differ across models or classes. Consult the scripts and stored outputs before drawing deployment conclusions.

### Why accuracy alone is not enough

In the stored MLP evaluation, one class has materially lower recall than the other. This highlights a real business issue: models can achieve high overall accuracy while overlooking a meaningful subset of borrowers. Class-specific recall, threshold selection, and the operational cost of each error type matter in credit-risk applications.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── logistic_regression.py
│   ├── main.py                # Decision Tree, Random Forest, correlation work
│   ├── XGBoost.py
│   ├── train.py               # MLP
│   └── summary.md             # Additional MLP evaluation notes
├── results/                   # Selected text evaluation outputs
├── docs/                      # Project overview and legacy notes
└── data/                      # Local dataset location (excluded from Git)
```

## Getting Started

**Prerequisite:** Python 3.9+ is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the cleaned dataset at:

```text
data/loan_dataset_cleaned_encoded.csv
```

The input must contain the binary target column `loan_paid_back`.

Run the models **from the repository root**:

```bash
python src/logistic_regression.py
python src/main.py
python src/XGBoost.py
python src/train.py
```

Selected outputs are written to `results/`.

**Data availability:** The dataset is not provided in this public-ready repository because redistribution permission has not been confirmed. A fresh checkout cannot reproduce results without the appropriate dataset.

## Key Takeaways

- Compare interpretable baselines with ensemble and neural-network approaches instead of assuming one algorithm will work best.
- Evaluate class-specific outcomes alongside aggregate scores, particularly where class imbalance exists.
- Translate predictive metrics into lending-process questions: What kind of mistake is costlier? When should a case be reviewed by a human?
- Treat explainability and fairness as essential follow-up work before any real lending use.

## Scope and Limitations

This is a **course project / portfolio demonstration**, not a production-ready credit scoring system. Its outputs should not be used to make real lending decisions. Independent reproduction, external validation, bias/fairness testing, data rights checks, and operational monitoring would be required before considering real-world use.

The original course PDF is intentionally omitted because it includes personal/team identifiers. Dataset files are also omitted pending confirmation of sharing rights.
