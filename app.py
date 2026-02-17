import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Anomaly Detection App", layout="wide")

st.title("🔍 Anomaly Detection using Isolation Forest")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        st.error("No numeric columns found!")
    else:
        st.subheader("Model Parameters")

        n_estimators = st.slider("Number of Trees", 50, 300, 100)
        contamination = st.slider("Contamination (Outlier %)", 0.01, 0.2, 0.05)
        sample_size = st.selectbox("Max Samples", ["auto", 100, 200, 500])

        if st.button("Train Model"):
            
            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_numeric)

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples=sample_size,
                random_state=42
            )

            model.fit(X_scaled)

            # Predict
            df["Anomaly"] = model.predict(X_scaled)

            st.success("Model Trained Successfully!")

            # Count anomalies
            anomaly_count = df["Anomaly"].value_counts()
            st.subheader("Anomaly Summary")
            st.write(anomaly_count)

            # Plot first two features
            if df_numeric.shape[1] >= 2:
                st.subheader("Anomaly Visualization")

                fig, ax = plt.subplots()
                ax.scatter(
                    df_numeric.iloc[:, 0],
                    df_numeric.iloc[:, 1],
                    c=df["Anomaly"]
                )
                ax.set_xlabel(df_numeric.columns[0])
                ax.set_ylabel(df_numeric.columns[1])

                st.pyplot(fig)

            st.subheader("Updated Dataset")
            st.write(df.head())
