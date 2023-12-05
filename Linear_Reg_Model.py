import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define the features and target variable
features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
            'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

X_train = train_data[features].copy()  # Create a copy to avoid warnings
y_train = train_data['SalePrice']
X_test = test_data[features].copy()  # Create a copy to avoid warnings

# Handle missing values
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Linear Regression model
model = LinearRegression()

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit the model on the training data
model.fit(X_train_split, y_train_split)

# Make predictions on the validation data
y_val_pred = model.predict(X_val_split)

# Evaluate the model on the validation data
mse = mean_squared_error(y_val_split, y_val_pred)
r2 = r2_score(y_val_split, y_val_pred)

print("Validation Mean Squared Error:", mse)
print("Validation R-squared:", r2)

# Train the model on the entire training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Create a DataFrame with 'ID' and 'SalePrice' columns
predicted_prices = pd.DataFrame({'ID': test_data['Id'], 'SalePrice': y_test_pred})

# Save the DataFrame to a CSV file
predicted_prices.to_csv('predicted_price.csv', index=False)
