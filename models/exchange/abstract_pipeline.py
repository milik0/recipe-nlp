from abc import ABC, abstractmethod
from typing import Any, Tuple, Self
from pydantic import BaseModel
from exchange.logger import get_logger

class AbstractPipeline(ABC):
    """Abstract base class for a data processing pipeline."""

    def __init__(self, config: BaseModel) -> None:
        self.config = config
        self.logger = get_logger(self.config.name)

    @abstractmethod
    def load(self: Self) -> Any:
        """Load raw data."""
        pass

    @abstractmethod
    def transform(self: Self, data: Any) -> Tuple[Any, Any]:
        """Transform raw data into features (X) and target (y)."""
        pass

    @abstractmethod
    def split(self: Self, X: Any, y: Any) -> Tuple[Any, Any, Any, Any]:
        """Split features and target into training and test sets."""
        pass

    @abstractmethod
    def fit(self: Self, X: Any, y: Any) -> Any:
        """Train model or pipeline components."""
        pass

    @abstractmethod
    def save(self: Self, path: str) -> None:
        """Save the model or outputs."""
        pass
