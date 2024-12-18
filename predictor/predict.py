import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
def prediction(stockName):

    valid_stocks = ["Google", "META", "NVIDIA", "Reliance", "Tesla"]
    if stockName not in valid_stocks:
        return "STOCK_NOT_PRESENT"

    datasetFileURL = "/home/pranjul_khankriyal/Desktop/4thyrMinor/StockMarketPricePrediction/predictor/DataSets/"+stockName+".csv"
    # Load dataset
    dataset = pd.read_csv(datasetFileURL)

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
    global y_test
    y_test= test_set['Close']

    # Model and training
    model.fit(X_train, y_train)

    # Predictions on the test set
    global y_pred
    y_pred = model.predict(X_test)

    # Convert X_train and X_test back to datetime for plotting
    X_train_dates = train_set['Date']
    X_test_dates = test_set['Date']

    # Convert X_test_dates to string format and sort in increasing order
    X_test_dates_str = X_test_dates.dt.strftime('%Y-%m-%d')

    # Create a DataFrame to pair the dates with predictions
    results = pd.DataFrame({
        'Date': X_test_dates_str,
        'Prediction': y_pred
    })

    # Sort the DataFrame by Date
    results_sorted = results.sort_values(by='Date')

    # Convert sorted Date column back to list of strings
    dates = results_sorted['Date'].tolist()  #sorted_dates
    price = results_sorted['Prediction'].tolist()  #sorted_predictions

    return dates, price


# ---- Predict stock price for a specific year based on user input ----
def predict_price_on_year(year):
    # Convert user input year to ordinal value
    input_date = pd.to_datetime(f'{year}-01-01')  # Assuming Jan 1st of the year
    input_ordinal = input_date.toordinal()

    # Predict stock price for the given year
    predicted_price = model.predict([[input_ordinal]])

    # ---- Evaluate the model ----
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Evaluate the acuracy of model:")
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return predicted_price.tolist()
