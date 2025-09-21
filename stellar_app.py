import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------
# Streamlit App
# --------------------------------
st.set_page_config(page_title="🌌 Stellar Time Capsules", layout="wide")
st.title("🔭 Stellar Time Capsules – Predicting Civilizational Echoes")

# File Upload
uploaded_file = st.file_uploader("stellar_time_capsule", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Drop StarSystem (ID-like column)
    X = data.drop(["PredictedEchoType", "StarSystem"], axis=1)
    y = data["PredictedEchoType"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Choose model
    model_choice = st.radio("Select Model:", ["Random Forest", "XGBoost"])

    if model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5],
        }
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 6, 10],
            "clf__learning_rate": [0.01, 0.1, 0.3],
        }

    # Full pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])

    # Hyperparameter tuning
    st.info("⏳ Training model... This may take a moment.")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("🌌 Model Performance")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text(classification_report(y_test, y_pred))

    # Feature importance (for Random Forest / XGBoost only)
    if model_choice == "Random Forest":
        clf = best_model.named_steps["clf"]
        importances = clf.feature_importances_
    else:
        clf = best_model.named_steps["clf"]
        importances = clf.feature_importances_

    # Plot
    st.subheader("🔑 Feature Importance")
    feature_names = (
        grid_search.best_estimator_.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out(categorical_cols)
    )
    all_features = numeric_cols + feature_names.tolist()

    fi = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=fi.values, y=fi.index, ax=ax, palette="viridis")
    st.pyplot(fig)

    # Custom prediction
    st.subheader("🛰 Predict New Star System")
    input_data = {}
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", data[col].dropna().unique())

    if st.button("Predict Echo Type"):
        sample = pd.DataFrame([input_data])
        prediction = best_model.predict(sample)[0]
        st.success(f"🔮 Predicted Echo Type: **{prediction}**")

