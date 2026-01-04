import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# StandardScaler für Normalisierung 
# SimpleImputer für NAN Werte in numerischen Spalten
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)


# Predictions
y_pred = pipeline.predict(X_test)                 # Klassen (0/1)
y_proba = pipeline.predict_proba(X_test)[:, 1]    # Wahrscheinlichkeiten

# Print Classification
print("\n=== Classification Metrics ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print WS
print("\n=== Regression-style Metrics (on probabilities) ===")
print("MAE :", mean_absolute_error(y_test, y_proba))
print("MSE :", mean_squared_error(y_test, y_proba))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_proba)))
print("R2  :", r2_score(y_test, y_proba))

# Model speichern
model_path = Path('../../models/logistic_regression_model.joblib')
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")

