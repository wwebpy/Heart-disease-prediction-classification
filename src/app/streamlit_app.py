import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("../../models/random_forest_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("../../data/processed/heart_disease_dummyEncoding.csv")

pipeline = load_model()
data = load_data()

imputer = pipeline.named_steps["imputer"]
model = pipeline.named_steps["model"]

X = data.drop(columns=["target"])
y = data["target"]

st.title("Heart Disease Risk Prediction")
st.caption("Enter your health data to receive a risk assessment and explanation.")

# Input
input_col, result_col = st.columns([2, 1])

with input_col:
    st.subheader("Patient Information")

    age = st.slider("Age", 20, 90, 50)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    thalch = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal"])
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

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

proba = model.predict_proba(input_imputed)[0][1]
risk_pct = proba * 100

with result_col:
    st.subheader("Risk Result")

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="font-size:64px; margin-bottom:0;">{risk_pct:.1f}%</h1>
            <p style="font-size:18px; color:gray;">Heart Disease Risk</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(int(risk_pct))

    if proba < 0.3:
        st.success("Low Risk")
    elif proba < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

# Kundeninsights
st.divider()
st.header("Your Risk Explained")

st.subheader("Your Values vs Population Average")

compare_features = ["age", "chol", "thalch", "trestbps"]
compare_df = pd.DataFrame({
    "Average": X[compare_features].mean(),
    "You": input_df[compare_features].iloc[0]
})

st.bar_chart(compare_df)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_imputed)
shap_values_class1 = shap_values[:, :, 1]

st.subheader("Individual Prediction Explanation")

shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[0],
    input_imputed.iloc[0],
    matplotlib=True
)
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Top Contributors")

local_shap_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Value": shap_values_class1[0]
}).sort_values("SHAP Value", key=abs, ascending=False)

st.bar_chart(local_shap_df.head(10).set_index("Feature"))

# Modelinsights
st.divider()
st.header("Model Insights")

st.subheader("Global Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_df.head(10).set_index("Feature"))

st.subheader("ROC Curve")

y_proba_all = model.predict_proba(imputer.transform(X))[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba_all)
auc = roc_auc_score(y, y_proba_all)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.legend()
st.pyplot(fig)

st.subheader("Confusion Matrix")

y_pred_all = model.predict(imputer.transform(X))
cm = confusion_matrix(y, y_pred_all)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
    ax=ax
)
st.pyplot(fig)