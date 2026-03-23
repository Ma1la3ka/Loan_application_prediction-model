import streamlit as st
import pandas as pd
import joblib

model = joblib.load('loan_approval_model.pkl')

st.title("Loan Approval Predictor")
st.write("Enter the details of the loan application to predict approval status.")
# Collecting user input
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income", min_value=0, value=500000)
    loan_amt = st.number_input("Loan Amount", min_value=0, value=200000)
    cibil = st.slider("CIBIL Score", 300, 900, 700)
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])

with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employed = st.selectbox("Self Employed?", ["Yes", "No"])
    term = st.number_input("Loan Term (Years)", min_value=1, max_value=20, value=10)
    luxury_assets = st.number_input("Luxury Assets Value", min_value=0, value=100000)

if st.button("Predict Approval"):
    # Create a dictionary matching your CSV column names exactly
    user_data = {
        'no_of_dependents': dependents,
        'education': education,
        'self_employed': employed,
        'income_annum': income,
        'loan_amount': loan_amt,
        'loan_term': term,
        'cibil_score': cibil,
        'residential_assets_value': 0, # Defaulting these for simplicity
        'commercial_assets_value': 0,
        'luxury_assets_value': luxury_assets,
        'bank_asset_value': 0
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    
    # Make Prediction
    prediction = model.predict(df)
    prob = model.predict_proba(df)[0][1] # Probability of Approval
    
    # Display Result
    if prediction[0] == 1:
        st.success(f"✅ Approved! (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ Rejected. (Confidence: {1-prob:.2%})")