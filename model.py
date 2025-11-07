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

# Save model to a file
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model trained and saved as model.pkl")
