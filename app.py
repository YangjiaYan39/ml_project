import streamlit as st
import requests

st.title("Loan Default Prediction App")
st.write("Enter customer information and get a prediction from the trained model.")

income = st.number_input("Income", value=5000.0)
age = st.number_input("Age", value=30, step=1)
credit_score = st.number_input("Credit Score", value=600.0)

col1, col2 = st.columns(2)

with col1:
    if st.button("Train Model"):
        r = requests.get("http://127.0.0.1:8000/train")
        st.write(r.json())

with col2:
    if st.button("Compare Models"):
        r = requests.get("http://127.0.0.1:8000/compare")
        st.write(r.json())

if st.button("Cross Validation"):
    r = requests.get("http://127.0.0.1:8000/cv")
    st.write(r.json())

if st.button("Predict"):
    r = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "income": income,
            "age": int(age),
            "credit_score": credit_score
        }
    )
    result = r.json()

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader(f"Prediction: {result['label']}")
        st.write("Reason:")
        for reason in result["reasons"]:
            st.write(f"- {reason}")