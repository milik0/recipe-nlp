import pandas as pd
import numpy as np
import nltk
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
import os

class TFIDFPreprocessing:
    def __init__(self):
        self.steps_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.ingredients_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.valid_cuisines = {}

    def fit(self, data: pd.DataFrame):
        print("Extracting primary cuisine...")
        data['primary_cuisine'] = data['cuisines'].apply(lambda cuisines: cuisines[0] if len(cuisines) > 0 else None)
        self.valid_cuisines = set(data['primary_cuisine'].unique())

        print("Fitting TF-IDF vectorizers...")
        self.steps_vectorizer.fit(data["steps_strings"])
        self.ingredients_vectorizer.fit(data["ingredients_text"])
        
        print("Fitting label encoder...")
        self.label_encoder.fit(data['primary_cuisine'])
        
        return self

    def transform(self, data) -> Tuple[np.ndarray, np.ndarray]:
        print("Starting transform process...")
        data = self._common_preprocessing(data)
        
        if self.valid_cuisines is None:
            raise ValueError("The preprocessor has not been fitted. Call fit() first.")

        print("Removing invalid cuisines: {data['cuisines'].isin(self.valid_cuisines)}")
        data = data[data['cuisines'].isin(self.valid_cuisines)].copy()

        # Extract primary cuisine
        data['primary_cuisine'] = data['cuisines'].apply(lambda cuisines: cuisines[0] if len(cuisines) > 0 else None)
        
        # Transform text data using fitted vectorizers
        print("Applying TF-IDF transformations...")
        steps_features = self.steps_vectorizer.transform(data["steps_strings"]).toarray()
        ingredients_features = self.ingredients_vectorizer.transform(data["ingredients_text"]).toarray()
        
        # Extract numerical features
        numerical_features = data[['n_steps', 'n_ingredients']].values
        
        # Combine all features
        print("Building feature matrix...")
        feature_matrices = [
            steps_features,
            ingredients_features,
            numerical_features
        ]

        X = np.hstack(feature_matrices)
        y = self.label_encoder.transform(data['primary_cuisine'])
        return X, y

    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(data)
        return self.transform(data)
    
    def save_pipeline(self, directory="./preprocessor"):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "steps_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.steps_vectorizer, f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.ingredients_vectorizer, f)

        with open(os.path.join(directory, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "wb") as f:
            pickle.dump(self.valid_cuisines, f)

        print(f"Preprocessor saved to {directory}")
        
    def load_pipeline(self, directory="./preprocessor"):
        with open(os.path.join(directory, "steps_vectorizer.pkl"), "rb") as f:
            self.steps_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "ingredients_vectorizer.pkl"), "rb") as f:
            self.ingredients_vectorizer = pickle.load(f)

        with open(os.path.join(directory, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(os.path.join(directory, "valid_cuisines.pkl"), "rb") as f:
            self.valid_cuisines = pickle.load(f)

        print(f"Preprocessor loaded from {directory}")
        return self
