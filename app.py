import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and selected features
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("logistic_scaler.pkl")
selected_features = joblib.load("logistic_features.pkl")

# Sample CSV
sample_csv = pd.DataFrame(columns=selected_features)
sample_csv_path = "sample_input.csv"
sample_csv.to_csv(sample_csv_path, index=False)

# Streamlit setup
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight:bold; color:#2c3e50; }
    .result { font-size:20px; padding:10px; border-radius:8px; background-color:#f9f9f9; border:1px solid #ddd; }
    .section { margin-top:30px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Employee Attrition Predictor")

st.markdown("Predict whether an employee is **likely to leave or stay** based on workplace features.")

threshold = 0.6  # Best threshold determined earlier

def predict(df):
    X = df[selected_features]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    df["Attrition_Probability"] = probs
    df["Prediction"] = np.where(probs >= threshold, "❌ Likely to Leave", "✅ Likely to Stay")
    return df

# Section: Input Method
st.subheader("📂 Upload CSV or 🔢 Enter Manually")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Upload CSV file with required columns:", type=["csv"])
with col2:
    with open(sample_csv_path, "rb") as file:
        st.download_button("📥 Download Sample CSV", file, file_name="sample_input.csv", mime="text/csv")

# --- CSV Prediction ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        else:
            result_df = predict(df)
            st.success("✅ Prediction completed!")
            st.dataframe(result_df.style.highlight_max(axis=0, subset=["Attrition_Probability"], color="lightcoral"))
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result CSV", csv, "attrition_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Manual Input ---
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📝 Manual Input")

with st.form("manual_input"):
    inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            inputs[feature] = st.number_input(feature, value=0)
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_df = pd.DataFrame([inputs])
    result = predict(input_df)
    st.markdown("### 🔮 Prediction Result")
    st.markdown(
        f"<div class='result'>🎯 <b>Attrition Probability:</b> {result['Attrition_Probability'].iloc[0]:.2f}<br>"
        f"📝 <b>Prediction:</b> {result['Prediction'].iloc[0]}</div>", unsafe_allow_html=True
    )

# --- Model Comparison ---
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📊 Final Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["✅ Logistic Regression", "Random Forest", "XGBoost"],
    "Threshold": [0.60, 0.20, 0.20],
    "Accuracy": [0.8027, 0.7585, 0.8129],
    "F1-Score (Class 1)": [0.5167, 0.4818, 0.4954],
    "Recall (Class 1)": [0.6596, 0.7021, 0.5745],
    "Precision (Class 1)": [0.4247, 0.3708, 0.4355],
    "Comments": [
        "🎯 Best F1. Balanced & interpretable.",
        "Good recall, but lower overall accuracy.",
        "Best accuracy, but slightly lower F1."
    ]
})

st.dataframe(comparison_df)

# --- Feature Selection Explanation ---
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📌 Feature Selection Rationale")

st.markdown("""
These 14 features were selected based on their **correlation with attrition**:

### ➕ Positively Correlated (more = more likely to leave):
- `OverTime`
- `MaritalStatus_Single`
- `JobRole_Sales Representative`
- `BusinessTravel_Travel_Frequently`

### ➖ Negatively Correlated (more = more likely to stay):
- `TotalWorkingYears`
- `JobLevel`
- `YearsInCurrentRole`
- `MonthlyIncome`
- `Age`
- `YearsWithCurrManager`
- `YearsAtCompany`
- `JobInvolvement`
- `JobSatisfaction`
- `EnvironmentSatisfaction`

This mix of features balances **interpretability** and **performance**.
""")

