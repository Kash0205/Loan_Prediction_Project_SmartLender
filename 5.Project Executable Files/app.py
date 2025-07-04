from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        # Extract form data
        form = request.form

        # Construct the input dict manually
        input_dict = {
            'Dependents': int(form.get("dependents")),
            'ApplicantIncome': float(form.get("applicant_income")),
            'CoapplicantIncome': float(form.get("coapplicant_income")),
            'LoanAmount': float(form.get("loan_amount")),
            'Loan_Amount_Term': float(form.get("loan_term")),
            'Credit_History': float(form.get("credit_history")),
            'Gender_Male': 1 if form.get("gender") == 'Male' else 0,
            'Married_Yes': 1 if form.get("married") == 'Yes' else 0,
            'Education_Not Graduate': 1 if form.get("education") == 'Not Graduate' else 0,
            'Self_Employed_Yes': 1 if form.get("self_employed") == 'Yes' else 0,
            'Property_Area_Semiurban': 1 if form.get("property_area") == 'Semiurban' else 0,
            'Property_Area_Urban': 1 if form.get("property_area") == 'Urban' else 0
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        expected_columns = [
            'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Gender_Male', 'Married_Yes', 'Education_Not Graduate',
            'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban'
        ]
        input_df = input_df[expected_columns]

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        print("User input dict:\n", input_dict)
        print("DataFrame before scaling:\n", input_df)
        print("Scaled input:\n", input_scaled)
        print("Prediction result:", prediction)
        # Show result
        result = "✅ Loan will be Approved" if prediction == 1 else "❌ Loan will Not be Approved"
        return render_template('output.html', result=result)

    return render_template('home.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)

