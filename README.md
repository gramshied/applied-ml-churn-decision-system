# Applied ML Churn Decision System

## Problem
Customer churn leads to revenue loss. This project builds an end-to-end
machine learning decision system to identify high-risk customers early
and support retention strategies.

## Dataset
IBM Telco Customer Churn dataset (7043 customers, mixed numerical and
categorical features).

## Approach
- Data understanding & quality checks
- Reproducible preprocessing pipeline
- Feature engineering using ColumnTransformer
- Multiple model comparison under class imbalance
- Explainability using SHAP
- Interactive deployment with Streamlit

## Model & Evaluation
Models evaluated:
- Logistic Regression (interpretable baseline)
- Random Forest (non-linear decision power)

Metrics:
- Precision, Recall, F1-score
- ROC-AUC (primary selection metric)

## Explainability
SHAP was used to identify key churn drivers:
- Month-to-month contracts increase churn risk
- Low tenure strongly correlates with churn
- High monthly charges increase churn probability

Additivity checks were disabled to handle numerical instability
in tree-based models within preprocessing pipelines.

## Deployment
Live app:
👉 https://applied-ml-churn-decision-system-sabvjjfixfoqe87jork49m.streamlit.app/

## Tech Stack
Python, Pandas, Scikit-learn, SHAP, Streamlit
