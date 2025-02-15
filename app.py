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

    rf = RandomForestClassifier(random_state=42)

    # train the model
    rf.fit(X_train, y_train)

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