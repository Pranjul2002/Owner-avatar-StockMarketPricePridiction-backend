from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor.predict import prediction,predict_price_on_year

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chart-data', methods=['GET'])
def get_chart_data():
    # Retrieve the 'stock' parameter from the URL
    stock_name = request.args.get('stock')  # Returns None if 'stock' is not in the query
    print("requested stock is- ",stock_name)

    if not stock_name:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    # Backend data for labels and dataset
    x,y=prediction(stock_name)

    data = {
        "stock_name": stock_name,
        "labels": x,
        "dataset": y
    }
    return jsonify(data)  # Return data as JSON


@app.route('/predict-year', methods=['POST'])
def predict_year():
    # Get the year from the request data
    data = request.get_json()
    print(data)
    year = data.get('year')
    print(year)

    if not year:
        return jsonify({"error": "Year is required"}), 400

    try:
        # Call the function to predict stock prices for the given year
        labels="prediction"
        predicted_prices = predict_price_on_year(year)
        print(predicted_prices)
        response = {
            "labels": labels,
            "predicted_prices": predicted_prices
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':

    app.run(debug=True)