import torch
from omegaconf import DictConfig
from transformers import LlamaModel, PreTrainedTokenizer

from .dispatch import GENERATOR_MAP, POOLER_MAP

class ComposableInitializer:
    def __init__(self, config: DictConfig):
        self.config = config

    def initialize(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, virtual_token_count: int, dataset_path: str) -> torch.Tensor:
        generator_config = self.config.get("embedding_generator")
        if not generator_config:
            raise ValueError("`embedding_generator` config is missing.")
        generator_type = generator_config.get("type")
        generator_class = GENERATOR_MAP.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator type: {generator_type}")
        generator = generator_class(generator_config.get("config"))

        pooler_config = self.config.get("embedding_pooler")
        if not pooler_config:
            raise ValueError("`embedding_pooler` config is missing.")
        pooler_type = pooler_config.get("type")
        pooler_class = POOLER_MAP.get(pooler_type)
        if not pooler_class:
            raise ValueError(f"Unknown pooler type: {pooler_type}")
        pooler = pooler_class(pooler_config.get("config"))

        initial_embeddings = generator.generate(main_model, main_tokenizer, dataset_path=dataset_path)
        pooled_embeddings = pooler.pool(initial_embeddings, virtual_token_count)

        noise_level = self.config.get("noise_level", 0.0)
        if noise_level > 0.0:
            noise = torch.randn_like(pooled_embeddings) * noise_level
            pooled_embeddings += noise
        
        return pooled_embeddings
