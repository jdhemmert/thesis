from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional

import torch
from omegaconf import DictConfig


class BaseInitializer(ABC):
    """
    Abstract base class for prompt initializers.
    """
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config

    @abstractmethod
    def initialize(self, **kwargs) -> torch.Tensor:
        """
        Initializes and returns a tensor for the virtual prompt.
        """
        print(f"Warning: unused initializer arguments {list(kwargs.keys())}")
