import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv("DataSets/Reliance.csv")

# Convert 'Date' column to datetime and then to ordinal (numerical format)
dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)

#-----------------Coverting date into ordinal
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

# Convert X_train and X_test back to datetime for plotting----->>>
X_train_dates = train_set['Date']
X_test_dates = test_set['Date']

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot training  & testing data
plt.scatter(X_train_dates, y_train, color='green', s=10, label='Date vs Price (Training)')
plt.scatter(X_test_dates, y_test, color='red', s=25, label='Date vs Price (Testing)')

# Plot predicted data
plt.plot(X_test_dates, y_pred, linestyle='--', color='black', label='Line Of Best-fit')

plt.legend()
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.title('Reliance Stock Price Prediction')
plt.grid(True)
plt.show()

# ---- Make prediction based on user input ----
# Function to predict stock price for a given year
def predict_stock_price(year):
    # Convert the user input year to an ordinal value
    input_date = pd.to_datetime(f'{year}-01-01')  # Assuming Jan 1st of the year
    input_ordinal = input_date.toordinal()

    # Predict the stock price for the given year
    predicted_price = model.predict([[input_ordinal]])
    return predicted_price[0]

# Get year input from user
user_input_year = int(input("Enter a year to predict stock price (e.g., 2025): "))
predicted_stock_price = predict_stock_price(user_input_year)

# Output the prediction
print(f"Predicted stock price for the year {user_input_year}: {predicted_stock_price:.2f}")


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
