import streamlit as st
import requests

st.title("Loan Approval Prediction")

person_age = st.number_input("Age", min_value=1, max_value=110, step=1)
person_gender = st.selectbox("Gender", options=["Male", "Female"])
person_education = st.selectbox("Education Level", options=["High School", "Associate", "Bachelor", "Master"])
person_income = st.number_input("Monthly Income", min_value=0.0, step=100.0, format="%.2f")
person_emp_exp = st.number_input("Monthly Expenses", min_value=0, step=1)
person_home_ownership = st.selectbox("Home Ownership Status", options=["OWN", "RENT", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan Amount", min_value=0.0, step=100.0, format="%.2f")
loan_intent = st.text_input("Reason for Loan")
loan_int_rate = st.slider("Loan Interest Rate (%)", min_value=1.0, max_value=20.0, step=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, step=1)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", options=["YES", "NO"])

if st.button("Predict Loan Approval"):
    payload = {
        "person_age": person_age,
        "person_gender": person_gender.lower(),
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }
    try:
        response = requests.post(" https://loan-backend-production-aeab.up.railway.app/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        prediction = result.get("loan_status_prediction", None)
        if prediction is not None:
            if prediction == 1:
                st.success("Loan Approved ✅")
            else:
                st.warning("Loan Not Approved ❌")
        else:
            st.error("Unexpected response from the API.")
    except requests.exceptions.HTTPError as http_err:
        try:
            error_detail = response.json()
            st.error(f"API validation error: {error_detail}")
        except Exception:
            st.error(f"HTTP error: {http_err}")
    except Exception as e:
        st.error(f"API request failed: {e}")

