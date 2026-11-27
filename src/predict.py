import csv
import numpy as np

# Get the feature names in the same order used during training
with open("../dataset/House Price India.csv", "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    feature_names = [name for name in reader.fieldnames if name not in ("id", "Price")]

# Load the trained model and its scaling values
model = np.load("../models/house_price_model.npz")
weights = model["weights"]
X_mean = model["X_mean"]
X_std = model["X_std"]
y_mean = model["y_mean"]
y_std = model["y_std"]

# Ask for the house details
values = []

print("Enter the following house details:")
print(feature_names)

for name in feature_names:
    value = float(input(f"{name}: "))
    values.append(value)

# Prepare the input in the same way as the training data
house = np.array(values, dtype=float)
house = (house - X_mean) / X_std
house = np.insert(house, 0, 1)

# Predict the house price
scaled_prediction = house @ weights
predicted_price = scaled_prediction * y_std + y_mean

print("Predicted House Price:", round(float(predicted_price), 2))