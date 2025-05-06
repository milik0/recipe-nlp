import time
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from pages.regression.utils.utils_train import build_features_tfidf, build_features_word2vec, get_metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def train_test_split_custom(data, feature_matrice):
    X = np.hstack(feature_matrice)
    y = data["minutes"].values

    X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = (
        train_test_split(X, y, data.index, test_size=0.2, random_state=42)
    )
    return X_train, X_test, y_train, y_test, X_train_indices, X_test_indices


def create_feature_matrices(data, embedding):
    if embedding == "w2v":
        return build_features_word2vec(data)
    elif embedding == "tf":
        return build_features_tfidf(data)
    else:
        return []

## FEEDFORWARD NEURAL NETWORKS FUNCTIONS
def train_feedforward(embedding, X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/bayes_model_{embedding}.keras"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

        model = Sequential(
            [
                # Input layer
                # l2 => to prevent overfitting
                Dense(
                    128,
                    activation="relu",
                    input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                Dropout(0.3),
                # Hidden layers
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                # Output layer (single neuron for regression)
                Dense(1),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",  # MSE for regression
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )
        model.save(model_path)
    
    y_pred = model.predict(X_test_scaled).flatten()
    return get_metrics(y_test, y_pred)

def forward_nn(embedding, X_train, y_train, X_test, y_test):

    print(f"\n{embedding} - FeedForward NN Start")
    feedforward_metrics = train_feedforward(embedding, X_train, y_train, X_test, y_test)
    print(f"{embedding} - FeedForward NN Done")

    return feedforward_metrics

## NAÏVE BAYES FUNCTIONS
def train_bayes(embedding, X_train, y_train, X_test, y_test):
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/bayes_model_{embedding}.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = BayesianRidge()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def bayesian(embedding, X_train, y_train, X_test, y_test):

    print(f"\n{embedding} - Bayesian Ridge Start")
    # Bayesian Ridge
    bayesian_metrics = train_bayes(
        embedding, X_train, y_train, X_test, y_test
    )
    print(f"{embedding} - Bayesian Ridge Done")

    return bayesian_metrics


## LINEAR REGRESSION FUNCTIONS
def train_linear_regression(embedding, X_train, y_train, X_test, y_test):
    dir_path = "models/saved"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f"{dir_path}/lr_model_{embedding}.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def linear(embedding, X_train, X_test, y_train, y_test):

    print(f"\n{embedding} - Linear Regression Start")
    
    linear_metrics = train_linear_regression(
        embedding, X_train, y_train, X_test, y_test
    )

    print(f"{embedding} - Linear Regression Done")

    return linear_metrics


def select_model(model: str, X_train, X_test, y_train, y_test):
    match model:
        # TF-IDF Models
        case "lr_tf":
            return linear("tf", X_train, X_test, y_train, y_test)
        case "bayesian_tf":
            return bayesian("tf", X_train, X_test, y_train, y_test)
        case "forward_nn_tf":
            return forward_nn("tf", X_train, X_test, y_train, y_test)

        # Word2Vec Models
        case "lr_w2v":
            return linear("w2v", X_train, X_test, y_train, y_test)
        case "bayesian_w2v":
            return bayesian("w2v", X_train, X_test, y_train, y_test)
        case "forward_nn_w2v":
            return forward_nn("w2v", X_train, X_test, y_train, y_test)

        # Default
        case _:
            print(f"ERROR - model {model} should not exists")
            return {"weights": []}
    return []
