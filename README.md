# Loan Default Prediction System

An end-to-end machine learning application for predicting loan default risk, built with FastAPI and Streamlit.

This project demonstrates how to design a simple ML pipeline, evaluate multiple models, and deploy a real-time prediction service with an interactive frontend.

---

## Overview

This system predicts whether a customer is likely to default on a loan based on basic financial features.

It integrates:

- model training and evaluation
- API-based prediction service
- interactive user interface
- simple explainability for prediction results

---
## Demo
![alt text](image-1.png)

## Tech Stack

- Python
- FastAPI
- Streamlit
- scikit-learn
- NumPy
- joblib

---

## Features

- Train multiple models and select the best one
- Compare model performance
- Perform 10-fold cross-validation
- Real-time prediction via API
- Interactive frontend with Streamlit
- Basic explanation of prediction results

---

## Project Structure

- `main.py` — FastAPI entry point
- `api.py` — API routes and prediction logic
- `model.py` — data generation, preprocessing, and model setup
- `app.py` — Streamlit frontend
- `best_model.pkl` — saved trained model
- `scaler.pkl` — saved scaler

---

## How It Works

1. Generate customer data with features such as income, age, and credit score
2. Split data into training and testing sets
3. Apply feature scaling
4. Train multiple models, including Logistic Regression and KNN
5. Evaluate performance and select the best model
6. Save the trained model and scaler
7. Serve predictions through FastAPI and Streamlit

---

## Running the Project

Start the backend:

`python -m uvicorn main:app --reload`

Start the frontend:

`python -m streamlit run app.py`

Open API docs:

http://127.0.0.1:8000/docs

Open Streamlit app:

http://localhost:8501

---

## API Endpoints

- `GET /train` — train models and save the best one
- `GET /compare` — compare model performance
- `GET /cv` — run 10-fold cross-validation
- `POST /predict` — predict loan default risk

---

## Input Features

The prediction uses:

- `income`
- `age`
- `credit_score`

---

## Example Prediction Output

The system returns:

- predicted class
- risk label
- simple explanation of the result

Example:

- Prediction: Low Risk
- Reason: Financial profile looks stable

---

## Highlights

- Built a complete ML pipeline from preprocessing to deployment
- Combined FastAPI backend with Streamlit frontend
- Added model persistence for reuse after training
- Included simple explainability to make prediction results easier to understand
