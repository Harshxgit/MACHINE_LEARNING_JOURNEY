# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix,classification_report
import pickle

# Sample house dataset (you can replace this with your CSV file)


dataset = pd.read_csv('loan_prediction1.csv')
dataset.fillna({'Gender':"Male" ,
             'Married':"Yes",
             'Dependents':0,
             'Self_Employed':"No",
             'Loan_Amount_Term':360,
             'Credit_History':1,
             'LoanAmount':146.41},inplace=True)

X = dataset[['LoanAmount', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome','Loan_Amount_Term']]
y = dataset['Loan_Status']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=42)
# Initialize and train Logistic Regression model
regressor = LogisticRegression(max_iter=500)
regressor.fit(x_train, y_train)

# Save the trained model to disk
pickle.dump(regressor, open('loan_model.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open('loan_model.pkl', 'rb'))
y_pred=model.predict(x_test)

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)

# Predict price for a new house: 2000 sqft, 3 bedrooms, 10 years old

