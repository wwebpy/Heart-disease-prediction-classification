# Heart Disease Risk Prediction

This project is a machine learning application that predicts the risk of heart disease based on patient health data.  
The goal is to build an explainable classification model and present the results in an interactive Streamlit app.

The focus is not only on prediction accuracy, but also on model transparency using SHAP explanations.

<p align="center">
  <img src="streamlit Input.png" width="49%">
  <img src="streamlit data.png" width="49%">
</p>

## Project Overview

- Binary classification: heart disease vs. no heart disease  
- Models trained and compared:
  - Logistic Regression (baseline)
  - Random Forest (Best)
  - XGBoost
- Best model selected based on ROC AUC and overall performance  
- Interactive Streamlit app for end users

---

## Modeling Approach

1. Data cleaning and preprocessing  
2. Train / test split with stratification  
3. Model training and evaluation  
4. Comparison using:
   - Accuracy
   - Precision / Recall / F1-score
   - ROC AUC  
5. Model explainability with SHAP:
   - Global feature importance
   - Local (individual) explanations  

<p align="center">
  <img src="roc model curve.png" width="49%">
  <img src="shap mean feature impact.png" width="49%">
</p>

---

## Streamlit Application

The Streamlit app allows users to:

- Enter personal health values  
- Receive a predicted heart disease risk (in %)  
- See a clear risk category (low / medium / high)  
- Understand *why* the model made this prediction using:
  - SHAP force plot
  - Top contributing features  
- Explore global model insights:
  - Feature importance
  - ROC curve
  - Confusion matrix  

---

