from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Self
from logger import get_logger

class AbstractPipeline(ABC):
    """Abstract class for a data processing pipeline."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = get_logger(self.config.name)

    @abstractmethod
    def load(self : Self) -> Any:
        """Load data from a source."""
        pass

    @abstractmethod
    def transform(self : Self, X: Any, y: Any) -> Any:
        """Apply transformations to the data."""
        pass

    @abstractmethod
    def split(self : Self, X: Any, y: Any) -> tuple:
        """Split data into training and test sets."""
        pass
