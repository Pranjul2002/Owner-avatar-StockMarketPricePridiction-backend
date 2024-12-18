import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv("../DataSets/Reliance.csv")

# Convert 'Date' column to datetime and then to ordinal (numerical format)
dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
dataset['Date_ordinal'] = dataset['Date'].apply(lambda date: date.toordinal())

# Split dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Extract features and target for training and testing
X_train = train_set['Date_ordinal'].values.reshape(-1, 1)
y_train = train_set['Close']

X_test = test_set['Date_ordinal'].values.reshape(-1, 1)
y_test = test_set['Close']

# Model creation and training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Convert X_train and X_test back to datetime for plotting
X_train_dates = train_set['Date']
X_test_dates = test_set['Date']

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(X_train_dates, y_train, color='green', s=10, label='Date vs Price (Training)')
plt.scatter(X_test_dates, y_test, color='red', s=25, label='Date vs Price (Testing)')
plt.plot(X_test_dates, y_pred, linestyle='--', color='black', label='Line Of Best-fit')

plt.legend()
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.title('Reliance Stock Price Prediction')
plt.grid(True)
plt.show()

# Function to predict stock price for a given year
def predict_stock_price(year):
    input_date = pd.to_datetime(f'{year}-01-01')  # Assuming Jan 1st of the year
    input_ordinal = input_date.toordinal()
    predicted_price = model.predict([[input_ordinal]])
    return predicted_price[0]

# Get year input from user with error handling
try:
    user_input_year = int(input("Enter a year to predict stock price (e.g., 2025): "))
    predicted_stock_price = predict_stock_price(user_input_year)

    # Output the prediction
    print(f"Predicted stock price for the year {user_input_year}: {predicted_stock_price:.2f}")

    input_date = pd.to_datetime(f'{user_input_year}-01-01').date()

    # Check if the date exists in the dataset
    if input_date in dataset['Date'].values:
        closing_price = dataset.loc[dataset['Date'] == input_date, 'Close'].values[0]
        print(f"Closing price on {input_date.strftime('%d-%m-%Y')}: {closing_price:.2f}")
    else:
        print(f"No closing price data available for {input_date.strftime('%d-%m-%Y')}.")

except ValueError:
    print("Please enter a valid year.")
