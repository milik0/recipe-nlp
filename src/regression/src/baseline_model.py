import joblib
import os
import numpy as np
from pydantic import BaseModel, Field
from typing import Self, Any
from sklearn.linear_model import LinearRegression

from exchange.abstract_model import AbstractModel

class BaselineParams(BaseModel):
    """Configuration for the baseline model."""

    type: str = "baseline"
    name: str = "Time"


class BaseLineModel(AbstractModel):
    def __init__(self, config: BaselineParams):
        super().__init__(config)
        self.model = LinearRegression()

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        self.model.fit(X, y)
        self.logger.info("Training completed successfully")
        return self

    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        self.logger.info(f"Making predictions with model: {self.config.name}")
        predictions = self.model.predict(X)
        return predictions

    def save(self: Self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self: Self, path: str) -> Self:
        if not os.path.exists(path):
            self.logger.error(f"Model file not found at path: {path}")
            raise FileNotFoundError(f"Model file not found at path: {path}")

        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
        return self
