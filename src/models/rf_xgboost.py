import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


def load_data(file_path='../../data/processed/heart_disease_dummyEncoding.csv'):
    df = pd.read_csv(file_path)
    print("Spalten im DataFrame:")
    print(df.columns.tolist())
    return df


data = load_data()

# train test split
X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ))
])

xgb_pipeline.fit(X_train, y_train)

y_pred = xgb_pipeline.predict(X_test)
y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]


print("\n=== XGBoost Metrics ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Feature Importance
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_pipeline.named_steps['model'].feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n=== Top 10 Feature Importances (XGBoost) ===")
print(feature_importance_df.head(10))


# Model speichern
model_path = Path('../../models/xgboost.joblib')
joblib.dump(xgb_pipeline, model_path)
print(f"\nModel saved to {model_path}")