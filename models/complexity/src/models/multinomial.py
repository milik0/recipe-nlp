import joblib
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, Self, Any
from models.model import AbstractModel

class MultinomialParams(BaseModel):
    """Configuration for the multinomial logistic regression model."""
    type: str = "multinomial"
    name: str
    solver: Literal["lbfgs", "saga", "newton-cg"] = "lbfgs"
    penalty: Literal["l2", "none"] = "l2"
    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength")

class Multinomial(AbstractModel):
    def __init__(self, config: MultinomialParams):
        super().__init__(config)
        self.logger.info(f"Initializing {self.config.name} model with configuration: {self.config.model_dump()}")
        self.model = LogisticRegression(**self.config.model_dump(exclude={"name"}))

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        self.logger.info(f"Starting training with config: {self.config.model_dump()}")
        self.model.fit(X, y)
        self.logger.info("Training completed successfully")
        return self

    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        self.logger.info(f"Making predictions with model: {self.config.name}")
        predictions = self.model.predict(X)
        self.logger.info(f"Predictions made for {len(X)} samples")
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
