from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List
from enum import Enum
from dataclasses import dataclass, field
import torch
import datasets
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer, PreTrainedModel, LlamaModel, AutoModelForCausalLM


class PromptInitializerName(str, Enum):
    """
    Enum for names of available prompt initializers.
    """
    RANDOM = "random"
    TEXT = "text"
    WINDOWED_AVERAGE = "windowed_average"
    EMBEDDING_MODEL = "embedding_model"

@dataclass
class EmbeddingModelInitializerConfig:
    embedding_model_name_or_path: str
    use_dataset_context: bool = False
    context_text: Optional[str] = None
    context_source_column: str = "biography"
    max_context_samples: Optional[int] = None
    downsampling_strategy: str = "average"
    upsampling_strategy: str = "pad_zeros"
    noise_level: float = 0.0


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
    _initializers: Dict[PromptInitializerName, Type[PromptInitializer]] = {}

    @classmethod
    def register(cls, name: PromptInitializerName):
        def wrapper(initializer_cls: Type[PromptInitializer]):
            cls._initializers[name] = initializer_cls
            return initializer_cls
        return wrapper

    @classmethod
    def create(cls, name: PromptInitializerName) -> PromptInitializer:
        initializer_cls = cls._initializers.get(name)
        if not initializer_cls:
            raise ValueError(f"Unknown initializer: {name}")
        return initializer_cls()

@InitializerFactory.register(PromptInitializerName.RANDOM)
class RandomInitializer(PromptInitializer):
    """
    Initializes the virtual prompt with random noise.
    """
    def initialize(self, model: LlamaModel, virtual_token_count: int, noise_level: float = 0.01, **kwargs) -> torch.Tensor:
        hidden_size = model.config.hidden_size
        return torch.randn(virtual_token_count, hidden_size) * noise_level

@InitializerFactory.register(PromptInitializerName.TEXT)
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

@InitializerFactory.register(PromptInitializerName.WINDOWED_AVERAGE)
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
            context_embeddings = embedding_layer(token_ids.to(model.device)).squeeze(0)

        N = context_embeddings.shape[0]

        if N < virtual_token_count:
            raise ValueError(f"Number of context tokens ({N}) is less than virtual_token_count ({virtual_token_count}).")
        
        # Windowed averaging logic
        base_size = N // virtual_token_count
        remainder = N % virtual_token_count
        averaged_embeddings: List[torch.Tensor] = []
        current_idx = 0
        for i in range(virtual_token_count):
            window_size = base_size + (1 if i < remainder else 0)
            window = context_embeddings[current_idx : current_idx + window_size]
            averaged_embeddings.append(torch.mean(window, dim=0))
            current_idx += window_size
        
        return torch.stack(averaged_embeddings)


@InitializerFactory.register(PromptInitializerName.EMBEDDING_MODEL)
class EmbeddingModelInitializer(PromptInitializer):
    """
    Initializes the prompt by running text through a separate embedding model.
    """
    def initialize(self, model: LlamaModel, virtual_token_count: int, initializer_config: DictConfig, dataset_path: str, **kwargs) -> torch.Tensor:
        schema = OmegaConf.structured(EmbeddingModelInitializerConfig)
        config: EmbeddingModelInitializerConfig = OmegaConf.merge(schema, initializer_config)

        context_text = self._get_context_text(config, dataset_path)
        
        print(f"Loading embedding model: {config.embedding_model_name_or_path}...")
        device = model.device
        embed_model = AutoModelForCausalLM.from_pretrained(config.embedding_model_name_or_path).to(device)
        embed_model.eval()
        tokenizer: PreTrainedTokenizer = kwargs['tokenizer']
        
        # Generate embeddings
        with torch.no_grad():
            inputs = tokenizer(context_text, return_tensors="pt", truncation=True).to(device)
            outputs = embed_model(**inputs, output_hidden_states=True)
            input_embeddings = outputs.last_hidden_state.squeeze(0)

        N = input_embeddings.shape[0]
        V = virtual_token_count
        
        if N > V:
            processed_embeddings = self._downsample(input_embeddings, V, config.downsampling_strategy)
        elif N < V:
            processed_embeddings = self._upsample(input_embeddings, V, config.upsampling_strategy, model.config.hidden_size, device)
        else:
            processed_embeddings = input_embeddings
            
        if config.noise_level > 0.0:
            noise = torch.randn_like(processed_embeddings) * config.noise_level
            processed_embeddings += noise
            
        print("Embedding model initialization complete.")
        return processed_embeddings

    def _get_context_text(self, config: EmbeddingModelInitializerConfig, dataset_path: str) -> str:
        if not config.use_dataset_context:
            if not config.context_text:
                raise ValueError("`context_text` must be provided in initializer_config if `use_dataset_context` is false.")
            return config.context_text
        
        print("Loading dataset to build initialization context...")
        dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]
        
        if config.context_source_column not in dataset.column_names:
            raise ValueError(f"Column '{config.context_source_column}' not found in dataset.")
            
        sample_range = range(len(dataset))
        if config.max_context_samples and config.max_context_samples < len(dataset):
            sample_range = range(config.max_context_samples)

        full_text = "\n".join([dataset[i][config.context_source_column] for i in sample_range])
        return full_text

    def _downsample(self, embeddings: torch.Tensor, target_len: int, strategy: str) -> torch.Tensor:
        if strategy == "average":
            N = embeddings.shape[0]
            base_size = N // target_len
            remainder = N % target_len
            averaged: List[torch.Tensor] = []
            current_idx = 0
            for i in range(target_len):
                window_size = base_size + (1 if i < remainder else 0)
                window = embeddings[current_idx : current_idx + window_size]
                averaged.append(torch.mean(window, dim=0))
                current_idx += window_size
            return torch.stack(averaged)
        elif strategy == "truncate":
            return embeddings[:target_len]
        else:
            raise ValueError(f"Unknown downsampling strategy: {strategy}")

    def _upsample(self, embeddings: torch.Tensor, target_len: int, strategy: str, hidden_size: int, device) -> torch.Tensor:
        n, h = embeddings.shape
        if strategy == "pad_zeros":
            padding = torch.zeros(target_len - n, h, device=device)
            return torch.cat([embeddings, padding], dim=0)
        elif strategy == "pad_random":
            padding = torch.randn(target_len - n, h, device=device) * 0.01
            return torch.cat([embeddings, padding], dim=0)
        elif strategy == "repeat":
            repeats = (target_len // n) + 1
            return embeddings.repeat(repeats, 1)[:target_len]
        else:
            raise ValueError(f"Unknown upsampling strategy: {strategy}")

