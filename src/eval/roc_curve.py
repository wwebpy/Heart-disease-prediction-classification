import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


def load_data(file_path='../../data/processed/heart_disease_dummyEncoding.csv'):
    return pd.read_csv(file_path)

data = load_data()

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Model laden
model_path = Path('../../models/')

logreg = joblib.load(model_path / 'logistic_regression_model.joblib')
rf = joblib.load(model_path / 'random_forest_model.joblib')
xgb = joblib.load(model_path / 'xgboost_model.joblib')  

# Vorhersage-Wahrscheinlichkeiten
proba_logreg = logreg.predict_proba(X_test)[:, 1]
proba_rf = rf.predict_proba(X_test)[:, 1]
proba_xgb = xgb.predict_proba(X_test)[:, 1]


# Roc Kurven rechnen
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proba_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)

auc_logreg = roc_auc_score(y_test, proba_logreg)
auc_rf = roc_auc_score(y_test, proba_rf)
auc_xgb = roc_auc_score(y_test, proba_xgb)

# Roc Kurven plotten
plt.figure(figsize=(8, 6))

plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()