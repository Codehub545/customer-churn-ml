Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are more likely to leave can help businesses take preventive retention actions.

In this project, different machine learning classification models are trained and compared to predict customer churn.

The project focuses on:

Understanding the customer dataset

Cleaning and preparing the data

Performing exploratory data analysis

Creating useful features

Training multiple classification models

Comparing model performance

Evaluating the final model

Identifying important features related to churn

Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains 7,043 customer records and 21 columns, including:

Customer demographics

Account information

Internet and phone services

Contract type

Payment method

Monthly charges

Total charges

Churn status

The target variable is:

Churn

0 → Customer stayed

1 → Customer churned

Project Workflow

The project follows a complete machine learning workflow:

Data Loading
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Data Preprocessing
     ↓
Model Training
     ↓
Model Comparison
     ↓
Model Evaluation
     ↓
Feature Importance
     ↓
Final Conclusion

Data Preparation

During data preparation:

TotalCharges was converted from text to numeric format.

Missing values were identified and handled through the preprocessing pipeline.

customerID was removed because it is an identifier rather than a predictive feature.

Numerical features were standardized.

Categorical features were converted using one-hot encoding.

Additional features were created, including service count and average monthly spending.

Exploratory Data Analysis

Several relationships were explored, including:

Overall customer churn distribution

Contract type and churn

Monthly charges and churn

Customer tenure and churn

The dataset contains:

Customer Status	Customers	Percentage
Stayed	5,174	73.46%
Churned	1,869	26.54%

This shows that the dataset is imbalanced, with more customers staying than churning.

Machine Learning Models

Three classification models were compared:

Logistic Regression

Random Forest

XGBoost

Model Performance
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Logistic Regression	0.7331	0.4983	0.7914	0.6116	0.8419
Random Forest	0.7594	0.5323	0.7701	0.6295	0.8404
XGBoost	0.8020	0.6632	0.5160	0.5805	0.8425
Final Model

Random Forest was selected for further analysis because it achieved the highest F1-score among the three initial models while maintaining strong recall.

For a churn prediction problem, recall is important because failing to identify a customer who is actually going to churn may result in a missed opportunity for customer retention.

Final Model Evaluation

The Random Forest model produced the following confusion matrix on the test set:

                 Predicted
                 Stayed  Churned

Actual Stayed       782      253
Actual Churned       86      288


The model correctly identified 288 of the 374 actual churned customers in the test set.

The project also evaluates the final model using:

Confusion Matrix

ROC Curve

Precision-Recall Curve

Feature Importance

Key Takeaway

The results show that machine learning can help identify customers who are at higher risk of churn.

The comparison also demonstrates why accuracy alone should not be used to select a churn prediction model. Different models performed differently across precision, recall, F1-score, and ROC-AUC.

In a real business setting, churn predictions could be used to support customer retention strategies by helping companies focus their efforts on customers who are more likely to leave.

Project Structure
customer-churn-ml/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── Customer_Churn_Prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Joblib

Jupyter Notebook

How to Run

Clone the repository and install the required Python packages:

pip install -r requirements.txt


Open the notebook:

jupyter notebook Customer_Churn_Prediction.ipynb


Run the notebook cells from top to bottom.

Author

Customer Churn Prediction — Machine Learning Project

Built as an end-to-end machine learning project using Python and scikit-learn.
