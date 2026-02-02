from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List
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

@InitializerFactory.register("windowed_average")
class WindowedAverageInitializer(PromptInitializer):
    """
    Initializes the virtual prompt by creating windowed averages of the context text embeddings.
    """
    def initialize(self, model: LlamaModel, tokenizer: PreTrainedTokenizer, context_text: str, virtual_token_count: int, **kwargs) -> torch.Tensor:
        if not context_text:
            raise ValueError("Context text cannot be empty for WindowedAverageInitializer.")
        if virtual_token_count <= 0:
            raise ValueError("virtual_token_count must be greater than 0.")

        token_ids = tokenizer(context_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            context_embeddings = embedding_layer(token_ids.to(model.device)).squeeze(0) # Shape: (N, hidden_size)

        N = context_embeddings.shape[0] # Number of context tokens

        if N < virtual_token_count:
            raise ValueError(
                f"Number of context tokens ({N}) is less than virtual_token_count ({virtual_token_count}). "
                "Cannot perform windowed averaging."
            )
        
        averaged_embeddings: List[torch.Tensor] = []
        current_idx = 0

        base_size = N // virtual_token_count
        remainder = N % virtual_token_count

        for i in range(virtual_token_count):
            window_size = base_size + (1 if i < remainder else 0)
            
            if window_size == 0: # Should not happen if N >= virtual_token_count
                raise ValueError("Calculated window size is zero, unexpected error in windowing logic.")

            window_embeddings = context_embeddings[current_idx : current_idx + window_size]
            averaged_embeddings.append(torch.mean(window_embeddings, dim=0))
            current_idx += window_size
        
        return torch.stack(averaged_embeddings)

