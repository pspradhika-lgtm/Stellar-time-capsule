# stellar_time_capsules_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Streamlit UI
st.title("🌌 Stellar Time Capsules – Predicting Civilizational Echoes in Space")

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your stellar_time_capsules.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(data.head())

    # Encode categorical columns
    label_encoders = {}
    for col in ['RadiationLevel', 'SignalAnomaly', 'InfraredExcess',
                'ChemicalAnomaly', 'PredictedEchoType']:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # Features and target
    X = data.drop(columns=['PredictedEchoType', 'StarSystem'])
    y = data['PredictedEchoType']

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # XGBoost Classifier with GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"✅ Improved Model Accuracy: {acc * 100:.2f}%")

    # Classification Report
    st.write("### Classification Report")
    target_names = label_encoders['PredictedEchoType'].classes_
    st.text(classification_report(y_test, y_pred, target_names=target_names))

    # Feature importance
    st.write("### Feature Importance")
    importances = best_model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=features, hue=features,
                dodge=False, legend=False, ax=ax, palette="viridis")
    st.pyplot(fig)

else:
    st.info("👆 Upload your dataset (`stellar_time_capsules.csv`) to get started.")
