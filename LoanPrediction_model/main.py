import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("loan_data.csv")
data.head()

data.info()       # check data types
data.describe()   # summary statistics
data.isnull().sum()  # missing values

data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)


le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Property_Area']:
    data[col] = le.fit_transform(data[col])
X = data.drop('Loan_Status', axis=1)  # features
y = data['Loan_Status']               # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
