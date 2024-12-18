import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load dataset
dataset = pd.read_csv("/home/pranjul_khankriyal/Desktop/4thyrMinor/StockMarketPricePrediction/predictor/DataSets/Reliance.csv")

# Convert 'Date' column to datetime and then to ordinal (numerical format)
dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)

# Convert Date into ordinal
dataset['Date_ordinal'] = dataset['Date'].apply(lambda date: date.toordinal())

# Split dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Extract features and target for training and testing
X_train = train_set['Date_ordinal'].values.reshape(-1, 1)
y_train = train_set['Close']

X_test = test_set['Date_ordinal'].values.reshape(-1, 1)
y_test = test_set['Close']

# Model creation and training
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Convert X_train and X_test back to datetime for plotting
X_train_dates = train_set['Date']
X_test_dates = test_set['Date']


'''
# Sort the test dates and corresponding predictions for a cleaner plot
sorted_test_indices = np.argsort(X_test_dates)
X_test_dates_sorted = X_test_dates.iloc[sorted_test_indices]
y_test_sorted = y_test.iloc[sorted_test_indices]
y_pred_sorted = y_pred[sorted_test_indices]
'''

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot actual stock prices (training and testing)
plt.scatter(X_train_dates, y_train, color='green', s=10, label='Training Data (Actual)')
plt.scatter(X_test_dates, y_test, color='red', s=25, label='Testing Data (Actual)')

# Plot predicted stock prices
plt.scatter(X_test_dates, y_pred, color='blue', s=15, linestyle='-', label='Predicted Prices')

# Add labels and title
plt.legend()
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.title('Reliance Stock Price Prediction (Decision Tree)')
plt.grid(True)
plt.show()

# ---- Predict stock price for a specific year based on user input ----
def predict_stock_price(year):
    # Convert user input year to ordinal value
    input_date = pd.to_datetime(f'{year}-01-01')  # Assuming Jan 1st of the year
    input_ordinal = input_date.toordinal()

    # Predict stock price for the given year
    predicted_price = model.predict([[input_ordinal]])
    return predicted_price[0]

# Add a loop to keep asking until the input is valid
while True:
    try:
        user_input_year = int(input("Enter a year to predict stock price (e.g., 2025): ").strip())
        break  # Break the loop if input is valid
    except ValueError:
        print("Invalid input. Please enter a valid year.")

# Make prediction for the user-input year
predicted_stock_price = predict_stock_price(user_input_year)

# Output the prediction
print(f"Predicted stock price for the year {user_input_year}: {predicted_stock_price:.2f}")


# ---- Evaluate the model ----
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
