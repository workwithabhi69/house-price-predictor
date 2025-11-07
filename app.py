from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    
    # Make prediction
    input_data = np.array([[area, bedrooms, bathrooms, stories]])
    prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction_text=f"Predicted House Price: ₹{round(prediction, 2)}")

if __name__ == "__main__":
    app.run(debug=True)
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset (area, bedrooms, bathrooms, stories, price)
X = np.array([
    [1000, 2, 1, 1],
    [1500, 3, 2, 1],
    [2000, 3, 2, 2],
    [2500, 4, 3, 2],
    [3000, 4, 4, 3],
    [3500, 5, 4, 3]
])
y = np.array([50, 75, 100, 130, 150, 180])  # in lakhs

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model saved as model.pkl")
