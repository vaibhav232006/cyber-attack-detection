import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- UI DESIGN --------------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:bold;
    color:#00FFFF;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🔐 Cyber Attack Detection System</p>', unsafe_allow_html=True)

st.write("""
This project uses **Unsupervised Machine Learning (Isolation Forest)** 
to detect unusual patterns in network traffic, which may indicate cyber attacks.
""")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    data = pd.read_csv("KDDTest+.txt", header=None, encoding='latin-1')
    return data

data = load_data()

st.subheader("📊 Dataset Preview")
st.write(data.head())

# -------------------- PREPROCESSING --------------------
# Remove last column (labels)
data = data.iloc[:, :-1]

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -------------------- MODEL --------------------
model = IsolationForest(contamination=0.1)
model.fit(scaled_data)

# Predict anomalies
data['Anomaly'] = model.predict(scaled_data)

# -------------------- RESULTS --------------------
st.subheader("📌 Results")
st.write(data['Anomaly'].value_counts())

# -------------------- SAMPLE OUTPUT --------------------
st.subheader("📋 Sample Data with Predictions")
st.write(data.head())

# -------------------- VISUALIZATION --------------------
st.subheader("📈 Visualization")
st.scatter_chart(data[[0, 1]])