import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def generate_data():
    np.random.seed(42)
    n=500
    income=np.random.normal(5000,2000,n)
    age=np.random.randint(18,65,n)
    credit_score=np.random.normal(600,100,n)
    X=np.column_stack((income,age,credit_score))
    y=((income<4000)|(credit_score<550)).astype(int)
    return X,y

def preprocess_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test,y_train,y_test,scaler

def get_models():
    return{
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "KNN (k=3)":KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)":KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)":KNeighborsClassifier(n_neighbors=7),
    }