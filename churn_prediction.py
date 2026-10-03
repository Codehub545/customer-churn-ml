# ============================================================
# CUSTOMER CHURN PREDICTION
# End-to-End Data Science & Machine Learning Project
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n================ DATASET ================\n")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\n================ DATA QUALITY ================\n")

print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. CLEANING
# ============================================================

# TotalCharges is sometimes stored as object/string
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Convert target
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Customer ID is an identifier, not a predictive feature
df = df.drop(columns=["customerID"])

print("\nAfter cleaning:")
print(df.info())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

df["AverageMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, np.nan)
)

df["AverageMonthlySpend"] = (
    df["AverageMonthlySpend"]
    .replace([np.inf, -np.inf], np.nan)
)

# Number of additional services
service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

# Tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 100],
    labels=["0-1 Year", "1-2 Years", "2-4 Years", "4+ Years"]
)

print("\nEngineered features added.")


# ============================================================
# 5. BASIC EDA
# ============================================================

print("\n================ CHURN DISTRIBUTION ================\n")

print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# Churn distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()


# Contract vs churn
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# Monthly charges
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()


# ============================================================
# 6. SPLIT FEATURES / TARGET
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. LEAKAGE-SAFE PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 9. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 10. TRAIN + EVALUATE
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(
            y_test,
            probabilities
        )
    }

    results.append(metrics)
    trained_models[name] = pipeline


results_df = pd.DataFrame(results)

print("\n================ MODEL COMPARISON ================\n")

print(
    results_df
    .sort_values("ROC_AUC", ascending=False)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 11. SELECT MODEL BASED ON ROC-AUC
# ============================================================

best_model_name = (
    results_df
    .sort_values("ROC_AUC", ascending=False)
    .iloc[0]["Model"]
)

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# ============================================================
# 12. THRESHOLD OPTIMIZATION
# ============================================================

probabilities = best_model.predict_proba(X_test)[:, 1]

threshold_results = []

for threshold in np.arange(0.20, 0.71, 0.05):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    threshold_results.append({

        "Threshold": round(threshold, 2),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })


threshold_df = pd.DataFrame(threshold_results)

print("\n================ THRESHOLD ANALYSIS ================\n")

print(threshold_df.round(3).to_string(index=False))


# Select threshold with highest F1
best_threshold = threshold_df.loc[
    threshold_df["F1"].idxmax(),
    "Threshold"
]

final_predictions = (
    probabilities >= best_threshold
).astype(int)

print(
    f"\nSelected probability threshold: "
    f"{best_threshold}"
)


# ============================================================
# 13. FINAL CLASSIFICATION REPORT
# ============================================================

print("\n================ FINAL REPORT ================\n")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=["Stayed", "Churned"]
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 15. ROC CURVE
# ============================================================

plt.figure(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title(
    f"ROC Curve - {best_model_name}"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(7, 5))

PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title(
    f"Precision-Recall Curve - {best_model_name}"
)

plt.tight_layout()
plt.show()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

try:

    model = best_model.named_steps["model"]

    feature_names = (
        best_model
        .named_steps["preprocessing"]
        .get_feature_names_out()
    )

    if hasattr(model, "feature_importances_"):

        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        )

        importance = (
            importance
            .sort_values(ascending=False)
            .head(15)
        )

        plt.figure(figsize=(9, 6))

        importance.sort_values().plot(
            kind="barh"
        )

        plt.title(
            "Top 15 Feature Importances"
        )

        plt.tight_layout()
        plt.show()

except Exception as e:

    print(
        "\nFeature importance unavailable:",
        e
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("CUSTOMER CHURN PROJECT COMPLETED")
print("==========================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "ROC-AUC:",
    round(
        results_df.loc[
            results_df["Model"] == best_model_name,
            "ROC_AUC"
        ].iloc[0],
        4
    )
)

print(
    "PR-AUC:",
    round(
        results_df.loc[
            results_df["Model"] == best_model_name,
            "PR_AUC"
        ].iloc[0],
        4
    )
)

print(
    "Optimized Threshold:",
    best_threshold
)
# ============================================================
# 19. STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n================ CROSS-VALIDATION ================\n")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],
        n_jobs=-1
    )

    print(f"\n{name}")

    print(
        "Accuracy:",
        round(scores["test_accuracy"].mean(), 4)
    )

    print(
        "Precision:",
        round(scores["test_precision"].mean(), 4)
    )

    print(
        "Recall:",
        round(scores["test_recall"].mean(), 4)
    )

    print(
        "F1:",
        round(scores["test_f1"].mean(), 4)
    )

    print(
        "ROC-AUC:",
        round(scores["test_roc_auc"].mean(), 4)
    )
# ============================================================
# 20. XGBOOST HYPERPARAMETER TUNING
# ============================================================

print("\n================ HYPERPARAMETER TUNING ================\n")

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(round(search.best_score_, 4))
# ============================================================
# 21. FINAL TUNED MODEL EVALUATION
# ============================================================

print("\n================ FINAL TUNED MODEL ================\n")

best_xgb = search.best_estimator_

# Predict probabilities on untouched test data
tuned_probabilities = best_xgb.predict_proba(X_test)[:, 1]

# Default threshold
tuned_predictions = (
    tuned_probabilities >= 0.50
).astype(int)

print("Test ROC-AUC:",
      round(
          roc_auc_score(
              y_test,
              tuned_probabilities
          ),
          4
      ))

print("Test PR-AUC:",
      round(
          average_precision_score(
              y_test,
              tuned_probabilities
          ),
          4
      ))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tuned_predictions,
        target_names=["Stayed", "Churned"]
    )
)
# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

import joblib

joblib.dump(
    best_xgb,
    "models/churn_model.pkl"
)

print("\nFinal model saved to models/churn_model.pkl")
