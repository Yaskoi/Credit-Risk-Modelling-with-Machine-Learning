# Credit Risk Modelling with Machine Learning

Predicting loan default probability on 32,581 historical loan records.
Built as part of a credit risk portfolio project demonstrating end-to-end
ML pipeline design, domain-aware preprocessing, and regulatory explainability.

---

## Results

| Model               | AUC-ROC | KS     | Precision (Default) | Recall (Default) | F1 (Default) |
|---------------------|---------|--------|---------------------|------------------|--------------|
| Logistic Regression | 0.8682  | 0.5870 | 0.51                | 0.79             | 0.62         |
| Random Forest       | 0.9346  | 0.7199 | 0.85                | 0.76             | 0.80         |
| **XGBoost**         | **0.9523**  | **0.7703** | **0.96**        | **0.75**         | **0.85**     |

> XGBoost selected as production model at threshold **0.294**, achieving
> 80% recall on defaults with 85.5% precision — the recommended operating
> point for a retail credit scorecard.

---

## Project structure
```
Credit-Risk-Modelling-with-Machine-Learning/
│
├── credit_scoring_ml.ipynb     # Full pipeline — single notebook
├── requirements.txt            # Python dependencies
│
├── README.md
│
└── outputs/
    ├── eda_report.html          # Auto-generated EDA report (ydata-profiling)
    ├── evaluation_summary.csv   # Metrics at 3 decision thresholds
    └── shap_values.csv          # SHAP values for test set
```

---

## Pipeline overview

| Step | Description | Key tools |
|------|-------------|-----------|
| 1. EDA | Distributions, class imbalance, correlations | pandas, seaborn, ydata-profiling |
| 2. Cleaning | Outlier removal, group-based imputation | pandas, numpy |
| 3. Preprocessing | Ordinal encoding, one-hot encoding, scaling | scikit-learn |
| 4. Resampling | SMOTE oversampling of minority class | imbalanced-learn |
| 5. Modelling | Logistic Regression, Random Forest, XGBoost | scikit-learn, xgboost |
| 6. Evaluation | AUC-ROC, KS statistic, Precision-Recall, calibration | scikit-learn |
| 7. Explainability | SHAP global + individual waterfall plots | shap |

---

## Key design decisions

**No data leakage**
SMOTE applied strictly after train/test split. StandardScaler fitted on
training data only and applied to test set — a common mistake in student
projects that inflates all reported metrics.

**Domain-appropriate metrics**
Accuracy is meaningless on a 78/22 imbalanced dataset. Evaluation focuses
on AUC-ROC and KS statistic — the two standard metrics in retail credit
scoring — alongside threshold-specific Precision/Recall analysis.

**Threshold optimisation**
The default 0.5 threshold is not optimal for credit risk. Three operating
points were evaluated:

| Threshold | Precision | Recall | Use case |
|-----------|-----------|--------|----------|
| 0.500 | 0.961 | 0.755 | Maximum precision |
| 0.522 | 0.968 | 0.753 | Optimal F1 |
| 0.294 | 0.855 | 0.800 | Bank-oriented (recommended) |

**Regulatory explainability**
SHAP waterfall plots generated for individual predictions satisfy EU AI Act
Article 22 and Basel III IRB requirements for explainable credit decisions.
Logistic Regression retained as a fully auditable shadow model.

---

## Setup
```bash
git clone https://github.com/Yaskoi/Credit-Risk-Modelling-with-Machine-Learning.git
cd Credit-Risk-Modelling-with-Machine-Learning
pip install -r requirements.txt
jupyter notebook credit_scoring_ml.ipynb
```

---

## Dataset

Source: [Kaggle — Credit Risk Dataset](https://www.kaggle.com/laotse/credit-risk-dataset)  
32,581 loan records | 12 features | Binary target: `loan_status` (0 = non-default, 1 = default)  
Class distribution: 78.2% non-default / 21.8% default

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-purple)
