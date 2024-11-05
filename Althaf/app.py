from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Sample dataset with patient names
np.random.seed(42)
names = [f"Patient_{i}" for i in range(1, 1001)]
data = {
    'patient_name': names,
    'age': np.random.randint(20, 90, 1000),
    'bmi': np.random.uniform(18, 35, 1000),
    'hospital_visits_last_year': np.random.randint(0, 10, 1000),
    'days_in_hospital': np.random.randint(1, 30, 1000),
    'chronic_conditions': np.random.randint(0, 5, 1000),
    'readmitted': np.random.choice([0, 1], 1000)
}

df = pd.DataFrame(data)

# Prepare Data for Model
X = df.drop(columns=['readmitted', 'patient_name'])
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prediction function
def predict_readmission(patient_name):
    patient_data = df[df['patient_name'] == patient_name]
    if patient_data.empty:
        return None
    
    patient_features = patient_data.drop(columns=['readmitted', 'patient_name']).values
    patient_features_scaled = scaler.transform(patient_features)
    prediction = log_reg.predict(patient_features_scaled)
    prediction_proba = log_reg.predict_proba(patient_features_scaled)[0][1]
    
    return {"prediction": prediction[0], "probability": prediction_proba}

# Define a route for the frontend
@app.route("/")
def index():
    return render_template("index.html")

# Define the API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    patient_name = data.get("patient_name")
    result = predict_readmission(patient_name)
    
    if result is None:
        return jsonify({"error": f"No data found for {patient_name}."})
    
    response = {
        "name": patient_name,
        "result": "will be re-admitted" if result["prediction"] == 1 else "will not be re-admitted",
        "probability": f"{result['probability']:.2f}"
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)