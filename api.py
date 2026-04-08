from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from model import generate_data, preprocess_data, get_models
import joblib, os

app = FastAPI()

class InputData(BaseModel):
    income: float
    age: int
    credit_score: float

@app.get("/")
def root():
    return {"message": "API is working"}

@app.get("/train")
def train():
    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    models = get_models()
    best_model_obj = None
    best_score = 0
    best_name = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_score:
            best_score = acc
            best_model_obj = model
            best_name = name
    joblib.dump(best_model_obj, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return {
        "status": "trained",
        "best_model": best_name,
        "accuracy": round(best_score, 3)
    }

@app.get("/compare")
def compare_models():
    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = round(acc, 3)
    return results

@app.get("/cv")
def cross_validation():
    X, y = generate_data()
    models = get_models()
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
        results[name] = round(scores.mean(), 3)
    return results

@app.post("/predict")
def predict(data: InputData):
    if not os.path.exists("best_model.pkl") or not os.path.exists("scaler.pkl"):
        return {"error": "Model not found. Call /train first."}
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    new_data = scaler.transform([[data.income, data.age, data.credit_score]])
    pred = int(model.predict(new_data)[0])
    reasons = []
    if data.income < 4000:
        reasons.append("Low income")
    if data.credit_score < 550:
        reasons.append("Low credit score")
    return {
        "prediction": pred,
        "label": "High Risk" if pred == 1 else "Low Risk",
        "reasons": reasons if pred == 1 else ["Financial profile looks stable"]
    }