from abc import ABC, abstractmethod
import torch
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, LlamaModel, AutoModelForCausalLM

from .utils import BaseContextConfig, get_context_text_from_config

@dataclass
class TokenizerEmbeddingGeneratorConfig(BaseContextConfig):
    pass

@dataclass
class ExternalModelEmbeddingGeneratorConfig(BaseContextConfig):
    embedding_model_name_or_path: str


class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, dataset_path: Optional[str]) -> torch.Tensor:
        pass

class TokenizerEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, config: DictConfig):
        schema = OmegaConf.structured(TokenizerEmbeddingGeneratorConfig)
        self.config: TokenizerEmbeddingGeneratorConfig = OmegaConf.merge(schema, config)

    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, dataset_path: Optional[str]) -> torch.Tensor:
        context_text = get_context_text_from_config(self.config, dataset_path)
        
        token_ids = main_tokenizer(context_text, return_tensors="pt").input_ids.to(main_model.device)
        
        with torch.no_grad():
            embedding_layer = main_model.get_input_embeddings()
            return embedding_layer(token_ids).squeeze(0)

class ExternalModelEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, config: DictConfig):
        schema = OmegaConf.structured(ExternalModelEmbeddingGeneratorConfig)
        self.config: ExternalModelEmbeddingGeneratorConfig = OmegaConf.merge(schema, config)

    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, dataset_path: Optional[str]) -> torch.Tensor:
        context_text = get_context_text_from_config(self.config, dataset_path)
        
        print(f"Loading embedding model: {self.config.embedding_model_name_or_path}...")
        device = main_model.device
        embed_model = AutoModelForCausalLM.from_pretrained(self.config.embedding_model_name_or_path).to(device)
        embed_model.eval()
        
        with torch.no_grad():
            inputs = main_tokenizer(context_text, return_tensors="pt", truncation=True).to(device)
            outputs = embed_model(**inputs, output_hidden_states=True)
            return outputs.last_hidden_state.squeeze(0)
