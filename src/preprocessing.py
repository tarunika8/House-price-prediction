import csv
import numpy as np

def prepare_data():
    with open("../dataset/House Price India.csv", "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames

    # Six selected parameters, matched without relying on capitalization
    selected_features = [
        "number of bedrooms",
        "number of bathrooms",
        "number of floors",
        "condition of the house",
        "grade of the house",
        "built year"
    ]

    actual_columns = {
        column.strip().lower(): column
        for column in columns
    }

    feature_columns = [actual_columns[name] for name in selected_features]

    X = np.array(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=float
    )
    y = np.array([float(row["Price"]) for row in rows], dtype=float)

    # Shuffle and split into training and testing data
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * 0.8)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]