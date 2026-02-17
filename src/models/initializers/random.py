import torch
from transformers import LlamaModel

from .base import BaseInitializer


class RandomInitializer(BaseInitializer):
    """
    Initializes the virtual prompt with random noise.
    """
    def initialize(self, model: LlamaModel, virtual_token_count: int, noise_level: int, **kwargs) -> torch.Tensor:
        super().initialize(**kwargs)
        return torch.randn(virtual_token_count, model.config.hidden_size) * noise_level
