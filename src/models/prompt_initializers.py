from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
from enum import Enum
import torch
from transformers import LlamaModel
from omegaconf import DictConfig

from .initializers.composable import ComposablePromptInitializer

class PromptInitializerName(str, Enum):
    """
    Enum for names of available prompt initializers.
    """
    RANDOM = "random"
    COMPOSABLE = "composable"

class BasePromptInitializer(ABC):
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
        pass

class RandomInitializer(BasePromptInitializer):
    """
    Initializes the virtual prompt with random noise.
    """
    def initialize(self, model: LlamaModel, virtual_token_count: int, **kwargs) -> torch.Tensor:
        noise_level = self.config.get("noise_level", 0.01) if self.config else 0.01
        hidden_size = model.config.hidden_size
        return torch.randn(virtual_token_count, hidden_size) * noise_level

INITIALIZER_MAP: Dict[str, Type[BasePromptInitializer]] = {
    "random": RandomInitializer,
    "composable": ComposablePromptInitializer,
}


