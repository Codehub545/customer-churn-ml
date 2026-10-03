Customer Churn Prediction Using Machine Learning
Project Overview

This project develops a machine learning system to predict customer churn using the IBM Telco Customer Churn sample dataset.

The project demonstrates an end-to-end machine learning workflow including data cleaning, exploratory data analysis, feature engineering, preprocessing, model comparison, cross-validation, hyperparameter optimization, probability threshold analysis, and model evaluation.

Dataset

The dataset contains 7,043 telecom customers with information about:

Customer demographics

Tenure

Contract type

Internet services

Payment method

Monthly charges

Total charges

Customer churn

The target variable is Churn.

Machine Learning Workflow
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Model Training
     ↓
Cross-Validation
     ↓
Hyperparameter Optimization
     ↓
Final Evaluation
     ↓
Saved ML Model

Models Compared

Three classification algorithms were evaluated:

Logistic Regression

Random Forest

XGBoost

Model performance was evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

PR-AUC

Feature Engineering

Additional features were created:

AverageMonthlySpend

ServiceCount

TenureGroup

The customer ID was removed because it is an identifier rather than a useful predictive feature.

Hyperparameter Optimization

RandomizedSearchCV with 5-fold cross-validation was used to optimize the XGBoost model.

Best parameters:

n_estimators = 200
max_depth = 2
learning_rate = 0.03
subsample = 0.8
colsample_bytree = 0.8


Best cross-validation ROC-AUC:

0.8492

Final Test Results

The tuned XGBoost model achieved:

Metric	Result
Accuracy	0.80
ROC-AUC	0.846
PR-AUC	0.6629
Churn Precision	0.66
Churn Recall	0.52
Churn F1	0.58
Threshold Analysis

The project also evaluated different probability thresholds instead of relying only on the default 0.50 threshold.

A threshold of 0.30 produced a higher recall/F1 trade-off for the original XGBoost configuration, demonstrating how classification thresholds can be adjusted according to the business objective.

Key Machine Learning Concepts Demonstrated

Data preprocessing

Missing-value handling

Categorical encoding

Feature scaling

Feature engineering

Class imbalance

Stratified train/test splitting

Machine learning pipelines

Ensemble learning

Gradient boosting

Cross-validation

Hyperparameter optimization

ROC-AUC

PR-AUC

Precision/Recall trade-offs

Classification threshold optimization

Model persistence

Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

Seaborn

Joblib

How to Run

Install dependencies:

pip install -r requirements.txt


Run the project:

python churn_prediction.py


The trained model is saved as:

models/churn_model.pkl

Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using multiple machine learning algorithms and systematic model evaluation. The final tuned XGBoost model achieved a test ROC-AUC of 0.846 and PR-AUC of 0.6629.