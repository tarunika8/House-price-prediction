import numpy as np
from preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

# Scale features using the training data
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Train using log(price) to keep predictions positive
y_train_log = np.log(y_train)

# Add bias column
X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_test = np.column_stack((np.ones(len(X_test)), X_test))

weights = np.zeros(X_train.shape[1])
learning_rate = 0.01
epochs = 3000

for epoch in range(epochs):
    predictions = X_train @ weights
    errors = predictions - y_train_log
    gradients = (2 / len(X_train)) * (X_train.T @ errors)
    weights -= learning_rate * gradients

# Evaluate on the test set
predicted_prices = np.exp(X_test @ weights)

mae = np.mean(np.abs(y_test - predicted_prices))
ss_total = np.sum((y_test - y_test.mean()) ** 2)
ss_residual = np.sum((y_test - predicted_prices) ** 2)
r2 = 1 - (ss_residual / ss_total)

print("Mean Absolute Error:", round(float(mae), 2))
print("R2 Score:", round(float(r2), 4))

# Save the trained model
np.savez(
    "../models/house_price_model.npz",
    weights=weights,
    X_mean=X_mean,
    X_std=X_std
)

print("Model saved successfully!")