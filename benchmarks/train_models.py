import time
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_features(data):
    steps_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    steps_features = steps_vectorizer.fit_transform(data["steps_string_standardize"])

    ingredients_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    ingredients_features = ingredients_vectorizer.fit_transform(data["ingredients_text"])

    tags_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    tags_features = tags_vectorizer.fit_transform(data["tags_text"])
    
    numerical_features = data[['n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'saturated_fat', 'carbohydrates']].values

    feature_matrices = [steps_features.toarray(), ingredients_features.toarray(),
                    tags_features.toarray(),
                   numerical_features]
    
    return feature_matrices



def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "MAPE": mape
    }


def train_simple_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)

def train_feedforward(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    model = Sequential([
            # Input layer
            # l2 => to prevent overfitting
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer (single neuron for regression)
            Dense(1)
        ])

    model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'  # MSE for regression
    )

    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
    )

    history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
    )
    y_pred = model.predict(X_test_scaled).flatten()
    return get_metrics(y_test, y_pred)

def train_models(data):
    feature_matrices = build_features(data)
    
    X = np.hstack(feature_matrices)
    y = data["minutes"].values
    
    X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = train_test_split(
        X, y, data.index, test_size=0.2, random_state=42
    )    
    
    results = []

    # Linear Regression
    start = time.time()
    linear_metrics = train_simple_model(LinearRegression(), X_train, y_train, X_test, y_test)
    linear_metrics["Model"] = "Linear Regression"
    linear_metrics["Train Time (s)"] = round(time.time() - start, 2)
    results.append(linear_metrics)

    # Bayesian Ridge
    start = time.time()
    bayesian_metrics = train_simple_model(BayesianRidge(), X_train, y_train, X_test, y_test)
    bayesian_metrics["Model"] = "Bayesian Ridge"
    bayesian_metrics["Train Time (s)"] = round(time.time() - start, 2)
    results.append(bayesian_metrics)

    # Feedforward NN
    start = time.time()
    feedforward_metrics = train_feedforward(X_train, y_train, X_test, y_test)
    feedforward_metrics["Model"] = "Feedforward NN"
    feedforward_metrics["Train Time (s)"] = round(time.time() - start, 2)
    results.append(feedforward_metrics)

    return pd.DataFrame(results).set_index("Model")

