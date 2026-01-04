import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
plt.style.use("dark_background")

st.set_page_config(page_title="HeartDiseasePrediction", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("../../data/processed/heart_disease_dummyEncoding.csv")
    return data

data = load_data()

