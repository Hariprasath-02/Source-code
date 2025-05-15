import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Streamlit page config
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🔍 Customer Churn Prediction Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset uploaded. Rows: {len(df)}")

    # Drop non-informative columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # --- Visualization 1: Churn Distribution ---
    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Exited', data=df, ax=ax1)
    ax1.set_title('Churn Distribution')
    st.pyplot(fig1)

    # --- Visualization 2: Age by Churn ---
    st.subheader("Age Distribution by Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Exited', y='Age', data=df, ax=ax2)
    ax2.set_title('Age by Churn')
    st.pyplot(fig2)

    # --- Visualization 3: Correlation Heatmap ---
    st.subheader("Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    # Features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    st.sidebar.header("Choose a Model")
    model_name = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    # Model definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    if st.sidebar.button("Train and Evaluate"):
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader(f"{model_name} Results")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues', ax=ax4)
        ax4.set_title(f'Confusion Matrix: {model_name}')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        st.pyplot(fig4)

        # Feature importance for Random Forest
        if model_name == "Random Forest":
            importances = model.feature_importances_
            features = X.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df.sort_values(by='Importance', ascending=False, inplace=True)

            # --- Visualization 5: Feature Importance ---
            st.subheader("Top 15 Important Features - Random Forest")
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', ax=ax5)
            st.pyplot(fig5)
