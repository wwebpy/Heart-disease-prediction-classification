import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv('../../data/processed/heart_disease_dummyEncoding.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_pipeline = joblib.load('../../models/random_forest_model.joblib')

imputer = rf_pipeline.named_steps['imputer']
rf_model = rf_pipeline.named_steps['model']

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)


explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_imputed)

# Herzkranke only
shap_values_class1 = shap_values[:, :, 1]

shap.summary_plot(
    shap_values_class1,
    X_test_imputed,
    plot_type="dot"
)

shap.summary_plot(
    shap_values_class1,
    X_test_imputed,
    plot_type="bar"
)

index = 0

shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[index],
    X_test_imputed.iloc[index],
    matplotlib=True
)