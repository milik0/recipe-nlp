import pandas as pd
import numpy as np
from typing import Self, Field
from pipeline import AbstractPipeline
from sklearn.model_selection import train_test_split

class SimplePipelineParams:
    """Configuration for the simple pipeline."""
    name: str
    type: str = "simple"
    file_path : str
    train_size: float = Field(default=0.8, gt=0, lt=1, description="Train/test split ratio")

class SimplePipeline(AbstractPipeline):
    """A simple pipeline implementing data loading, transformation, and splitting."""

    def __init__(self: Self, config: SimplePipelineParams) -> None:
        super().__init__(config)

    def load(self: Self) -> pd.DataFrame:
        """Simulate loading a dataset."""
        self.logger.info(f"loading data")
        if self.config.file_path.endswith(".csv"):
            return pd.read_csv(self.config.file_path)    
        raise Exception(f"Unable to load file {self.config.file_path}")

    def transform(self: Self, data: pd.DataFrame) -> np.ndarray:
        """Apply simple transformations (e.g., encoding labels)."""
        self.logger.info(f"transforming data")
        return data.to_numpy()

    def split(self: Self, data: np.ndarray) -> tuple:
        """Split data into train/test sets."""
        self.logger.info(f"split data")
        return train_test_split(data, self.config.train_size)
