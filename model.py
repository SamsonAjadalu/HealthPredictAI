import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Load datasets and preprocess
df_diabetes = pd.read_csv('data/diabetes.csv')
df_heart = pd.read_csv('data/processed.cleveland.data', header=None, names=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'num'
])
print(df_diabetes.head())
# Selected features for Heart Disease
heart_selected_features = ['age', 'sex', 'cp', 'chol', 'fbs', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
print(df_heart.head())
# Preprocess datasets (heart disease) - create a copy of the selected features
X_heart = df_heart[heart_selected_features].copy()
y_heart = (df_heart['num'] > 0).astype(int)
X_heart.replace('?', np.nan, inplace=True)  # Now safely modify the copy
X_heart = X_heart.apply(pd.to_numeric, errors='coerce')
X_heart.fillna(X_heart.mean(), inplace=True)

# Selected features for Diabetes
diabetes_selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Preprocess datasets (diabetes) - no warning needed as this is a new DataFrame
X_diabetes = df_diabetes[diabetes_selected_features]
y_diabetes = df_diabetes['Outcome']

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_heart_poly = poly.fit_transform(X_heart)
X_diabetes_poly = poly2.fit_transform(X_diabetes)

# Scale the data
scaler = StandardScaler()
scaler2 = StandardScaler()
X_heart_scaled = scaler.fit_transform(X_heart_poly)
X_diabetes_scaled = scaler2.fit_transform(X_diabetes_poly)

# Train Logistic Regression models
model_heart = LogisticRegression(C=0.01, max_iter=5000, class_weight='balanced')
model_diabetes = LogisticRegression(C=100, max_iter=2000)

model_heart.fit(X_heart_scaled, y_heart)
model_diabetes.fit(X_diabetes_scaled, y_diabetes)


def predict_heart_disease(input_data):
    # Transform and scale the input data
    input_poly = poly.transform([input_data])
    input_scaled = scaler.transform(input_poly)
    return model_heart.predict(input_scaled)


def predict_diabetes(input_data):
    # Transform and scale the input data
    input_poly = poly2.transform([input_data])
    input_scaled = scaler2.transform(input_poly)
    return model_diabetes.predict(input_scaled)
