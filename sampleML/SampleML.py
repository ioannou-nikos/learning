import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree # For decision tree
from sklearn.linear_model import LinearRegression # For regression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def my_tree(X_train, X_test, y_train, y_test):
    # Decision Tree model
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    # Print the decision tree rules
    tree_rules = export_text(tree_model, feature_names=list(X_train.columns))
    #print(tree_rules)

    # Evaluate decision tree model
    y_pred = tree_model.predict(X_test)
    tree_mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {tree_mse:.3f}')
    

    # Normalize RMSE
    Min = data['Y'].min()
    Max = data['Y'].max()
    tree_nmse = tree_mse / (Max - Min)
    print(f'Normalized MSE: {tree_nmse:.3f}')

    

    # Visualizing decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=["A",'B'],
        filled=True,
        rounded=True,
        fontsize=7
    )
    plt.title("Decision Tree Structure")
    #plt.show()


def my_linear(X_train, X_test,y_train,y_test):
    # Linear Regression model
    #model = LinearRegression(positive=True)  # Use LinearRegression for regression tasks
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Compute and print linear model coefficients
    coefficients = linear_model.coef_
    intercept = linear_model.intercept_
    print(f'Coefficients: {coefficients}')
    print(f'Intercept: {intercept}')

    # Evaluate linear model
    y_pred = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {linear_mse:.3f}')

    #posible = 100 / y_pred
    #print(f'Possible: {posible}')
    #existing = 100 / pd.Series(y_train).to_numpy()
    #print(f'Existing: {existing}')
    

    # Normalize RMSE
    Min = data['Y'].min()
    Max = data['Y'].max()
    linear_nmse = linear_mse / (Max - Min)
    print(f'Normalized MSE: {linear_nmse:.3f}')

    print(f'Prediction is:{linear_model.predict([[1.01, 1.13]])}')
if __name__ == "__main__":
    # Load data
    data = pd.read_excel('./data/test.xlsx')

    # Preprocess data
    X = data.drop('Y', axis=1)
    y = data['Y']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'Y Description: {data['Y'].describe()}')
    
    my_linear(X_train, X_test, y_train, y_test)
    print("=======================================")
    my_tree(X_train, X_test, y_train, y_test)