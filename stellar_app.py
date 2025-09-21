import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Exoplanet Habitability Classifier", layout="wide")
st.title("🪐 Exoplanet Habitability Classifier")
st.write("Predict whether discovered exoplanets are **Habitable, Semi-Habitable, or Non-Habitable**")

# ---------------------------------------------------
# Dataset Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Kepler Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔭 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)

    # ---------------------------------------------------
    # Preprocessing (simplified demo version)
    # ---------------------------------------------------
    # Keep important columns (adjust depending on your Kaggle dataset schema)
    features = ["koi_period", "koi_prad", "koi_teq", "koi_srad", "koi_steff"]
    target = "koi_disposition"   # usually "CONFIRMED", "CANDIDATE", "FALSE POSITIVE"

    df = df[features + [target]].dropna()

    # Map target into habitability-like classes
    df[target] = df[target].map({
        "CONFIRMED": "Habitable",
        "CANDIDATE": "Semi-Habitable",
        "FALSE POSITIVE": "Non-Habitable"
    })

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------
    # Model Training
    # ---------------------------------------------------
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # ---------------------------------------------------
    # Evaluation
    # ---------------------------------------------------
    st.subheader("🌌 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    # ---------------------------------------------------
    # Single Prediction
    # ---------------------------------------------------
    st.subheader("🔮 Try Your Own Exoplanet")

    col1, col2, col3 = st.columns(3)
    with col1:
        koi_period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=100.0)
    with col2:
        koi_prad = st.number_input("Planet Radius (Earth radii)", min_value=0.1, max_value=20.0, value=1.0)
    with col3:
        koi_teq = st.number_input("Equilibrium Temperature (K)", min_value=100.0, max_value=10000.0, value=288.0)

    col4, col5 = st.columns(2)
    with col4:
        koi_srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=10.0, value=1.0)
    with col5:
        koi_steff = st.number_input("Stellar Temperature (K)", min_value=2000.0, max_value=10000.0, value=5778.0)

    if st.button("Predict Habitability"):
        sample = np.array([[koi_period, koi_prad, koi_teq, koi_srad, koi_steff]])
        pred = clf.predict(sample)[0]
        st.success(f"Predicted Habitability: **{pred}**")

