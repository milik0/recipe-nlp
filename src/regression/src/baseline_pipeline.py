import pandas as pd
import numpy as np
from typing import Self, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import re
import nltk
from nltk.stem import WordNetLemmatizer
import ast
from scipy.sparse import hstack
import os

from exchange.abstract_pipeline import AbstractPipeline
from exchange.constants import STOPWORDS

class BaselinePipelineParams:
    """Configuration for the simple pipeline."""

    name: str = "BaselinePipelineParams"
    type: str = "baseline"
    file_path: str = "../data/dataset.csv"
    train_size: float = 0.8

class BaselinePipeline(AbstractPipeline):
    """A simple pipeline implementing data loading, transformation, and splitting."""

    def __init__(self: Self, config: BaselinePipelineParams) -> None:
        super().__init__(config)
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

    def load(self: Self) -> pd.DataFrame:
        """Load the dataset."""
        self.logger.info(f"Loading data from {os.path.join(os.getcwd(), self.config.file_path)}")
        if self.config.file_path.endswith(".csv"):
            return pd.read_csv(self.config.file_path)
        raise Exception(f"Unable to load file {self.config.file_path}")

    def _clean_recipe_names(self: Self, recipes) -> List:
        """Basic cleaning and lemmatization of recipe names."""
        cleaned_recipes = []
        for recipe in recipes:
            recipe = recipe.lower()
            recipe = re.sub(r'[^a-z\s]', '', recipe)
            words = recipe.split()
            words = [self.lemmatizer.lemmatize(word) for word in words]
            words = [word for word in words if word not in STOPWORDS]
            cleaned_recipes.append(" ".join(words))
        return cleaned_recipes

    def transform(self: Self, data: pd.DataFrame) -> np.ndarray:
        """Transform data by cleaning and feature engineering."""
        self.logger.info("Transforming data")

        data.set_index('id', inplace=True)
        for col in ["tags", "steps", "ingredients", "nutrition"]:
            data[col] = data[col].apply(ast.literal_eval)

        data.drop(columns=["contributor_id", "submitted"], inplace=True, errors="ignore")
        data.dropna(subset=["name"], inplace=True)
        data['n_steps'] = data['steps'].apply(len)
        data['n_ingredients'] = data['ingredients'].apply(len)

        data['cleaned_name'] = self._clean_recipe_names(data['name'])
        data["description"] = data["description"].apply(lambda x: x.lower() if isinstance(x, str) else "")

        for col in ["tags", "steps", "ingredients"]:
            data[col] = data[col].apply(lambda x: [s.lower() for s in x if isinstance(s, str)] if isinstance(x, list) else [])

        data.dropna(subset=["tags", "steps", "ingredients"], inplace=True, how="any")
        data.reset_index(inplace=True)

        data['combined_text'] = (
            data['cleaned_name'] + ' ' +
            data['description'].fillna('') + ' ' +
            data['ingredients'].apply(lambda x: " ".join(x)) + ' ' +
            data['steps'].apply(lambda x: " ".join(x))
        )

        self.logger.info("Applying TF-IDF")
        tfidf = TfidfVectorizer(max_features=1000)
        X_text = tfidf.fit_transform(data['combined_text'])

        X_numeric = data[["n_steps", "n_ingredients", "minutes"]].fillna(0).to_numpy()

        return hstack([X_text, X_numeric])
    
    def fit(self: Self, X: np.ndarray, y:np.ndarray) -> Any:
        pass

    def split(self: Self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        """Split data into train/test sets."""
        self.logger.info(f"Splitting data into train/test sets")
        return train_test_split(X, y, train_size=self.config.train_size, random_state=42)

    def save(self: Self, data):
        """Placeholder for saving model or results."""
        pass
