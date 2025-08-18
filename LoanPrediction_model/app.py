import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ApplicantIncome = float(request.form['ApplicantIncome'])
    CoapplicantIncome = float(request.form['CoapplicantIncome'])
    LoanAmount = float(request.form['LoanAmount'])
    Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
    Credit_History = int(request.form['Credit_History'])
    final_features = [np.array([LoanAmount, Credit_History, ApplicantIncome, CoapplicantIncome , Loan_Amount_Term])]

    prediction = model.predict(final_features)

    if prediction == 1:
        result = "Yes ✅ Loan Approved"
    else:
        result = "No ❌ Loan Not Approved"

    return render_template('index.html', prediction_text='{}'.format(result))



if __name__ == "__main__":
    app.run(debug=True)