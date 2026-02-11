from abc import ABC, abstractmethod
import torch
from typing import List
from omegaconf import DictConfig

class BaseEmbeddingPooler(ABC):
    @abstractmethod
    def pool(self, embeddings: torch.Tensor, virtual_token_count: int) -> torch.Tensor:
        pass

class WindowedAveragePooler(BaseEmbeddingPooler):
    def __init__(self, config: DictConfig):
        pass

    def pool(self, embeddings: torch.Tensor, virtual_token_count: int) -> torch.Tensor:
        N = embeddings.shape[0]
        if N < virtual_token_count:
            raise ValueError(f"WindowedAveragePooler requires at least as many input embeddings ({N}) as virtual tokens ({virtual_token_count}).")
        
        base_size = N // virtual_token_count
        remainder = N % virtual_token_count
        averaged: List[torch.Tensor] = []
        current_idx = 0
        for i in range(virtual_token_count):
            window_size = base_size + (1 if i < remainder else 0)
            window = embeddings[current_idx : current_idx + window_size]
            averaged.append(torch.mean(window, dim=0))
            current_idx += window_size
        return torch.stack(averaged)

class PadPooler(BaseEmbeddingPooler):
    def __init__(self, config: DictConfig):
        self.pad_strategy = getattr(config, "strategy", "zeros")
        self.noise_level = getattr(config, "noise_level", 0.01)

    def pool(self, embeddings: torch.Tensor, virtual_token_count: int) -> torch.Tensor:
        n, h = embeddings.shape
        if n >= virtual_token_count:
            raise ValueError(f"PadPooler requires fewer input embeddings ({n}) than virtual tokens ({virtual_token_count}).")

        padding_size = virtual_token_count - n
        if self.pad_strategy == "zeros":
            padding = torch.zeros(padding_size, h, device=embeddings.device)
        elif self.pad_strategy == "random":
            padding = torch.randn(padding_size, h, device=embeddings.device) * self.noise_level
        else:
            raise ValueError(f"Unknown padding strategy: {self.pad_strategy}")
        
        return torch.cat([embeddings, padding], dim=0)

class TruncatePooler(BaseEmbeddingPooler):
    def __init__(self, config: DictConfig):
        pass

    def pool(self, embeddings: torch.Tensor, virtual_token_count: int) -> torch.Tensor:
        return embeddings[:virtual_token_count]

class IdentityPooler(BaseEmbeddingPooler):
    def __init__(self, config: DictConfig):
        pass

    def pool(self, embeddings: torch.Tensor, virtual_token_count: int) -> torch.Tensor:
        if embeddings.shape[0] != virtual_token_count:
            raise ValueError(f"IdentityPooler requires input embedding count ({embeddings.shape[0]}) to equal virtual token count ({virtual_token_count}).")
        return embeddings
