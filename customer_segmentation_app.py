
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def load_model_and_data():
    data = pd.read_csv("customer_data.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_scaled)
    return scaler, model

scaler, kmeans_model = load_model_and_data()

# Streamlit UI
st.title("🧠 Customer Segmentation App")
st.write("Input customer behavior data to determine the segment they belong to.")

# Input sliders
recency = st.slider("Recency (days since last purchase)", 1, 365, 90)
frequency = st.slider("Frequency (number of purchases)", 1, 50, 10)
monetary = st.slider("Monetary Value (₹)", 100.0, 20000.0, 5000.0, step=100.0)
tenure = st.slider("Tenure (days since first purchase)", 30, 2000, 365)

# Prepare input and predict
input_data = np.array([[recency, frequency, monetary, tenure]])
scaled_input = scaler.transform(input_data)
predicted_cluster = kmeans_model.predict(scaled_input)[0]

# Output result
st.subheader("🧩 Assigned Customer Segment:")
st.success(f"Segment {predicted_cluster}")
