import numpy as np
import pandas as pd
from keras.datasets import mnist
import pickle

from nest_asyncio import apply
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "This is the index page"
@app.route('/training', methods=['POST'])

def training():
    #data = request.get_json()  # status code
    data = request.get_json()  # Extract JSON data from request body

    # Extract values from JSON
    n_estimators = data.get("n_estimators")
    max_depth = data.get("max_depth")
    min_samples_split = data.get("min_samples_split")
    min_samples_leaf = data.get("min_samples_leaf")

    parameter  = jsonify({
        "message": "Data received successfully",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    })

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten data for RandomForest  (convert 28x28 to 784 features)

    x_train, y_train = x_train[:200], y_train[:200]
    x_test, y_test = x_test[:200], y_test[:200]

    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    # Normalize pixel values (0-255) to (0-1)
    X_train = x_train_flat / 255.0
    X_test = x_test_flat / 255.0

    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    with mlflow.start_run():
        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth =  max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf )
        # Fit GridSearchCV
        rf.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_param("param_grid", parameter)
        # Evaluate on test data
        y_pred = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log the best model
        mlflow.sklearn.log_model(rf, "model")

        print("Test Accuracy:", test_accuracy)

    # Start MLflow tracking
    mlflow.set_experiment("Random Forest Hyperparameter Tuning on mnist")
    with open("D:/ML/mlruns/models/random_classifier.pkl", "wb") as f:
        pickle.dump(rf, f)

    return "success review it or working"


def get_best_model_params():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten data for RandomForest  (convert 28x28 to 784 features)
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    # Normalize pixel values (0-255) to (0-1)
    X_train = x_train_flat / 255.0
    X_test = x_test_flat / 255.0


    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300,
                                random_state=42)

    # train the best model
    rf.fit(X_train, y_train)

    with open("./model_best/random_best_classifier.pkl", "wb") as f:
        pickle.dump(rf, f)


# driver function
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)