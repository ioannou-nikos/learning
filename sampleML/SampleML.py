import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load data
data = pd.read_excel('./data/test.xlsx')

# Preprocess data
X = data.drop('Y', axis=1)
y = data['Y']

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
#model = LinearRegression(positive=True)  # Use LinearRegression for regression tasks
model = LinearRegression()
#model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Compute and print coefficients
coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.3f}')
print(f'Y Description: {data['Y'].describe()}')

# Normalize RMSE
Min = data['Y'].min()
Max = data['Y'].max()
nmse = mse / (Max - Min)
print(f'Normalized MSE: {nmse:.3f}')