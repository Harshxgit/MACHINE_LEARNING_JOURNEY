# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Sample house dataset (you can replace this with your CSV file)
data = {
    'size_sqft': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'age': [10, 15, 20, 5, 8],
    'area':[0,0,1,1,1],
    'price': [400000, 500000, 600000, 650000, 700000],
}

dataset = pd.DataFrame(data)

# Features and target variable
X = dataset[['size_sqft', 'bedrooms', 'age', 'area']]
y = dataset['price']

# Train-test split (optional; you can train on all data if small)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Save the trained model to disk
pickle.dump(regressor, open('house_price_model.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open('house_price_model.pkl', 'rb'))

# Predict price for a new house: 2000 sqft, 3 bedrooms, 10 years old
predicted_price = model.predict([[2000, 3, 10,1]])
print(f"Predicted price: ${predicted_price[0]:,.2f}")
