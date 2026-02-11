from abc import ABC, abstractmethod
import torch
import datasets
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, LlamaModel, AutoModelForCausalLM

@dataclass
class TokenizerEmbeddingGeneratorConfig:
    context_text: str

@dataclass
class ExternalModelEmbeddingGeneratorConfig:
    embedding_model_name_or_path: str
    use_dataset_context: bool = False
    context_text: Optional[str] = None
    context_source_column: str = "biography"
    max_context_samples: Optional[int] = None

class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, **kwargs) -> torch.Tensor:
        pass

class TokenizerEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, config: DictConfig):
        schema = OmegaConf.structured(TokenizerEmbeddingGeneratorConfig)
        self.config: TokenizerEmbeddingGeneratorConfig = OmegaConf.merge(schema, config)

    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, **kwargs) -> torch.Tensor:
        if not self.config.context_text:
            raise ValueError("`context_text` must be provided for TokenizerEmbeddingGenerator.")
        
        token_ids = main_tokenizer(self.config.context_text, return_tensors="pt").input_ids.to(main_model.device)
        
        with torch.no_grad():
            embedding_layer = main_model.get_input_embeddings()
            return embedding_layer(token_ids).squeeze(0)

class ExternalModelEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, config: DictConfig):
        schema = OmegaConf.structured(ExternalModelEmbeddingGeneratorConfig)
        self.config: ExternalModelEmbeddingGeneratorConfig = OmegaConf.merge(schema, config)

    def generate(self, main_model: LlamaModel, main_tokenizer: PreTrainedTokenizer, **kwargs) -> torch.Tensor:
        context_text = self._get_context_text(kwargs.get("dataset_path"))
        
        print(f"Loading embedding model: {self.config.embedding_model_name_or_path}...")
        device = main_model.device
        embed_model = AutoModelForCausalLM.from_pretrained(self.config.embedding_model_name_or_path).to(device)
        embed_model.eval()
        
        with torch.no_grad():
            inputs = main_tokenizer(context_text, return_tensors="pt", truncation=True).to(device)
            outputs = embed_model(**inputs, output_hidden_states=True)
            return outputs.last_hidden_state.squeeze(0)

    def _get_context_text(self, dataset_path: Optional[str]) -> str:
        if not self.config.use_dataset_context:
            if not self.config.context_text:
                raise ValueError("`context_text` must be provided if `use_dataset_context` is false.")
            return self.config.context_text
        
        if not dataset_path:
            raise ValueError("`dataset_path` must be provided to the initializer if `use_dataset_context` is true.")
            
        print("Loading dataset to build initialization context...")
        dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]
        
        if self.config.context_source_column not in dataset.column_names:
            raise ValueError(f"Column '{self.config.context_source_column}' not found in dataset.")
            
        sample_range = range(len(dataset))
        if self.config.max_context_samples and self.config.max_context_samples < len(dataset):
            sample_range = range(self.config.max_context_samples)

        return "\n".join([dataset[i][self.config.context_source_column] for i in sample_range])
