<<<<<<< HEAD
# ----------------------------------------
# House Price Prediction App (Final Version with ₹ Output)
# ----------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import traceback
import locale

# Set Indian number format
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

app = Flask(__name__)
CORS(app)

# Load Model
MODEL_PATH = 'random_forest_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as file:
        model_data = pickle.load(file)

    model = model_data['model']
    MODEL_FEATURES = model_data['features']
    print("Model loaded successfully!")

except Exception as e:
    print("Model loading error:", e)
    model = None
    MODEL_FEATURES = []

@app.route('/')
def home():
    return jsonify({"message": "House Price Prediction API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data received.'}), 400

        df = pd.DataFrame([data])

        prediction = model.predict(df)
        price = float(prediction[0])

        # Format to Indian Rupees format
        formatted = locale.format_string("₹%.2f", price, grouping=True)

        return jsonify({
            'prediction': formatted,
            'raw_value': price
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
=======
# ----------------------------------------
# House Price Prediction App (Final Version with ₹ Output)
# ----------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import traceback
import locale

# Set Indian number format
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

app = Flask(__name__)
CORS(app)

# Load Model
MODEL_PATH = 'random_forest_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as file:
        model_data = pickle.load(file)

    model = model_data['model']
    MODEL_FEATURES = model_data['features']
    print("Model loaded successfully!")

except Exception as e:
    print("Model loading error:", e)
    model = None
    MODEL_FEATURES = []

@app.route('/')
def home():
    return jsonify({"message": "House Price Prediction API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data received.'}), 400

        df = pd.DataFrame([data])

        prediction = model.predict(df)
        price = float(prediction[0])

        # Format to Indian Rupees format
        formatted = locale.format_string("₹%.2f", price, grouping=True)

        return jsonify({
            'prediction': formatted,
            'raw_value': price
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> ecd2909c32b6a754e02c050b36674d3e8499487e
