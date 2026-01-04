# Heart Disease Risk Prediction

This project is a machine learning application that predicts the risk of heart disease based on patient health data.  
The goal is to build an explainable classification model and present the results in an interactive Streamlit app.

The focus is not only on prediction accuracy, but also on model transparency using SHAP explanations.

<img src="streamlit Input.png" width="49%"/>
<img src="streamlit data.png" width="49%"/>
---

## Project Overview

- Binary classification: heart disease vs. no heart disease  
- Models trained and compared:
  - Logistic Regression (baseline)
  - Random Forest (Best)
  - XGBoost
- Best model selected based on ROC AUC and overall performance  
- Interactive Streamlit app for end users

---

## Dataset

- Source: Kaggle – Heart Disease Dataset  
- Target variable:
  - `target = 1` → heart disease present  
  - `target = 0` → no heart disease  

**Features include:**
- Age  
- Blood pressure  
- Cholesterol  
- Maximum heart rate  
- ST depression  
- Chest pain type  
- Exercise-induced angina  
- Sex and ECG-related features  

Missing values were handled using imputation, and categorical variables were encoded before training.

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

The Random Forest model was selected as the final model due to strong performance and good interpretability.

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

