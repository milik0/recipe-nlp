import pandas as pd
import numpy as np
from typing import Self, List
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.stem import WordNetLemmatizer
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

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

    def load(self: Self) -> pd.DataFrame:
        """Simulate loading a dataset."""
        self.logger.info(f"loading data  {self.config.file_path}")
        self.logger.info(f"Loading data from {self.config.file_path} where pwd is {os.getcwd()}")
        self.logger.info(f"{os.path.join(os.getcwd(), self.config.file_path)}")
        if self.config.file_path.endswith(".csv"):
            return pd.read_csv(self.config.file_path)
        raise Exception(f"Unable to load file {self.config.file_path}")

    def _clean_recipe_names(self: Self, recipes) -> List:
        cleaned_recipes = []
        
        for recipe in recipes:
            recipe = recipe.lower()
            recipe = re.sub(r'[^a-z\s]', '', recipe)
            
            recipe_words = recipe.split()
            
            recipe_words = [self.lemmatizer.lemmatize(word) for word in recipe_words]
            
            recipe_words = [word for word in recipe_words if word not in STOPWORDS]
            
            cleaned_recipe = " ".join(recipe_words)
            cleaned_recipes.append(cleaned_recipe)
        
        return cleaned_recipes

    def transform(self: Self, data: pd.DataFrame) -> np.ndarray:
        """Apply simple transformations (e.g., encoding labels)."""
        data.set_index('id', inplace=True)
        for col in ["tags", "steps", "ingredients", "nutrition"]:
            data[col] = data[col].apply(ast.literal_eval)

        data.drop(columns=["contributor_id", "submitted"], inplace=True, errors="ignore")
        data.dropna(subset=["name"], inplace=True)

        data['name'] = self._clean_recipe_names(data['name'])

        for col in ["tags", "steps", "ingredients"]:
            data[col] = data[col].apply(lambda x: [s.lower() for s in x if isinstance(s, str)] if isinstance(x, list) else [])

        data['n_steps'] = data['steps'].apply(len)
        data['n_ingredients'] = data['ingredients'].apply(len)

        data.dropna(subset=["tags", "steps", "ingredients"], inplace=True, how="any")
        data.reset_index(inplace=True)

        self.logger.info("Applying Bag of Words (CountVectorizer)")
        bow = CountVectorizer(max_features=1000)
        X_text = bow.fit_transform(data['ingredients'].apply(lambda x: " ".join(x)))

        X_numeric = data[["n_steps", "n_ingredients", "minutes"]].fillna(0).to_numpy()

        return hstack([X_text, X_numeric])

    def split(self: Self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Split data into train/test sets."""
        self.logger.info(f"split data")
        return train_test_split(X, y, train_size=self.config.train_size)

    def save(self: Self, data):
        pass
