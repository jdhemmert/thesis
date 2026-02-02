from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, LlamaModel


class PromptInitializer(ABC):
    """
    Abstract base class for prompt initializers.
    """
    @abstractmethod
    def initialize(self, model: LlamaModel, **kwargs) -> torch.Tensor:
        """
        Initializes and returns a tensor for the virtual prompt.
        """
        pass

class InitializerFactory:
    _initializers: Dict[str, Type[PromptInitializer]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(initializer_cls: Type[PromptInitializer]):
            cls._initializers[name] = initializer_cls
            return initializer_cls
        return wrapper

    @classmethod
    def create(cls, name: str) -> PromptInitializer:
        initializer_cls = cls._initializers.get(name)
        if not initializer_cls:
            raise ValueError(f"Unknown initializer: {name}")
        return initializer_cls()

@InitializerFactory.register("random")
class RandomInitializer(PromptInitializer):
    """
    Initializes the virtual prompt with random noise.
    """
    def initialize(self, model: LlamaModel, virtual_token_count: int, noise_level: float = 0.01, **kwargs) -> torch.Tensor:
        hidden_size = model.config.hidden_size
        return torch.randn(virtual_token_count, hidden_size) * noise_level

@InitializerFactory.register("text")
class TextInitializer(PromptInitializer):
    """
    Initializes the virtual prompt from a piece of text.
    An optional noise level can be added to the embeddings.
    """
    def initialize(self, model: LlamaModel, tokenizer: PreTrainedTokenizer, context_text: str, noise_level: float = 0.0, **kwargs) -> torch.Tensor:
        if not context_text:
            raise ValueError("Context text cannot be empty for TextInitializer.")
        
        token_ids = tokenizer(context_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            initial_weights = embedding_layer(token_ids.to(model.device)).squeeze(0)

        if noise_level > 0.0:
            noise = torch.randn_like(initial_weights) * noise_level
            initial_weights += noise
        
        return initial_weights

