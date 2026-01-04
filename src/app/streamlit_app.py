import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("../../models/random_forest_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("../../data/processed/heart_disease_dummyEncoding.csv")
model_pipeline = load_model()
data = load_data()

imputer = model_pipeline.named_steps["imputer"]
model = model_pipeline.named_steps["model"]

X = data.drop(columns=["target"])
y = data["target"]


st.sidebar.header("Patient Input")

age = st.sidebar.slider("Age", 20, 90, 50)
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal"])
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])

# Input
input_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalch,
    "oldpeak": oldpeak,
    "slope_missing": 0,
    "sex_Male": 1 if sex == "Male" else 0,
    "cp_atypical angina": 1 if cp == "atypical angina" else 0,
    "cp_non-anginal": 1 if cp == "non-anginal" else 0,
    "cp_typical angina": 1 if cp == "typical angina" else 0,
    "fbs_True": 0,
    "restecg_normal": 1,
    "restecg_st-t abnormality": 0,
    "exang_True": 1 if exang == "Yes" else 0,
    "slope_flat": 1,
    "slope_upsloping": 0,
}

input_df = pd.DataFrame([input_dict])[X.columns]
input_imputed = pd.DataFrame(imputer.transform(input_df), columns=X.columns)

# Model
proba = model.predict_proba(input_imputed)[0][1]

st.title("Heart Disease Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Risk", f"{proba*100:.1f} %")
    if proba >= 0.5:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")

# Feature Importance
with col2:
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(importance_df.set_index("Feature").head(10))

# Force Plot
st.subheader("Prediction Explanation (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_imputed)
shap_values_class1 = shap_values[:, :, 1]

fig = shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[0],
    input_imputed.iloc[0],
    matplotlib=True
)
st.pyplot(fig)

# roc curve
st.subheader("Model Performance (ROC Curve)")

y_proba_all = model.predict_proba(imputer.transform(X))[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba_all)
auc = roc_auc_score(y, y_proba_all)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)