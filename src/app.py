from flask import Flask, request, render_template_string
import numpy as np

app = Flask(__name__)

# Load the model trained with six features
model = np.load("../models/house_price_model.npz")
weights = model["weights"]
X_mean = model["X_mean"]
X_std = model["X_std"]

# These must match the feature order in preprocessing.py
FEATURES = [
    ("bedrooms", "Number of bedrooms"),
    ("bathrooms", "Number of bathrooms"),
    ("floors", "Number of floors"),
    ("condition", "House condition (1–5)"),
    ("grade", "House grade"),
    ("built_year", "Year built")
]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 35px auto;
            padding: 20px;
            background: #f3f6f4;
            color: #222;
        }

        h1 {
            color: #246b45;
        }

        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        label {
            font-size: 14px;
        }

        input {
            box-sizing: border-box;
            width: 100%;
            padding: 10px;
            margin-top: 5px;
        }

        button {
            grid-column: 1 / -1;
            padding: 12px;
            background: #246b45;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }

        .result {
            margin-top: 24px;
            padding: 18px;
            background: white;
            border-radius: 8px;
        }

        .result li {
            margin: 8px 0;
        }

        .price {
            color: #246b45;
        }

        .error {
            color: #b00020;
            font-weight: bold;
        }

        @media (max-width: 550px) {
            form {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <h1>House Price Prediction</h1>
    <p>Enter the house details below.</p>

    <form method="POST">
        {% for key, label in features %}
            <label>
                {{ label }}
                <input
                    type="number"
                    step="any"
                    name="{{ key }}"
                    value="{{ form_values.get(key, '') }}"
                    required
                >
            </label>
        {% endfor %}

        <button type="submit">Predict Price</button>
    </form>

    {% if submitted %}
        <div class="result">
            <h2>Entered House Details</h2>

            <ul>
                {% for key, label in features %}
                    <li>
                        <strong>{{ label }}:</strong>
                        {{ form_values.get(key, '') }}
                    </li>
                {% endfor %}
            </ul>

            {% if prediction is not none %}
                <h2 class="price">Predicted House Price: {{ prediction }}</h2>
            {% endif %}

            {% if error %}
                <p class="error">{{ error }}</p>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    submitted = request.method == "POST"
    form_values = {}

    if submitted:
        # Keep the user's entered values available for display
        form_values = {
            key: request.form.get(key, "")
            for key, label in FEATURES
        }

        try:
            # Convert the entered values to numbers in model feature order
            values = [
                float(form_values[key])
                for key, label in FEATURES
            ]

            house = np.array(values, dtype=float)

            # Apply the same feature scaling used during training
            house = (house - X_mean) / X_std

            # Add the bias value
            house = np.insert(house, 0, 1)

            # Predict log(price), then convert back to price
            log_price = house @ weights
            predicted_price = np.exp(log_price)

            prediction = f"{float(predicted_price):,.2f}"

        except ValueError:
            error = "Please enter valid numbers in all six fields."

    return render_template_string(
        HTML,
        features=FEATURES,
        form_values=form_values,
        prediction=prediction,
        error=error,
        submitted=submitted
    )


if __name__ == "__main__":
    app.run(debug=True)