import numpy as np
import pandas as pd
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


def training():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten data for RandomForest  (convert 28x28 to 784 features)
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    # Normalize pixel values (0-255) to (0-1)
    X_train = x_train_flat / 255.0
    X_test = x_test_flat / 255.0

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        # 'bootstrap': [True, False]
    }
    param_grid = {
        'n_estimators' : [n_estimators],
        'max_depth' : [max_depth],
        'min_samples_split' : [min_samples_split],
        'min_samples_leaf' : [min_samples_leaf]
    }

    mlflow.sklearn.autolog()

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV (taking the cross validation as cv = 5)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    with mlflow.start_run():
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_param("param_grid", param_grid)
        mlflow.log_params(grid_search.best_params_)

        # Evaluate on test data
        y_pred = grid_search.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log the best model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")

        # Report results
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validated Accuracy:", grid_search.best_score_)
        print("Test Accuracy:", test_accuracy)

    # Start MLflow tracking
    mlflow.set_experiment("Random Forest Hyperparameter Tuning on mnist")
    with open("./model_plain/random_classifier.pkl", "wb") as f:
        pickle.dump(rf, f)


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