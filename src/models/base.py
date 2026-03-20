from abc import ABC, abstractmethod
import os
import hydra
from omegaconf import DictConfig
import torch
from typing import Dict, Type, Optional, Tuple, Any
from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, LlamaConfig
from peft import get_peft_model, PeftConfig


import time

def retry_oserror(fn, retries=3, delay_s=10):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except OSError as e:
            last_exc = e
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)
    raise last_exc


class ModelArchitecture(str, Enum):
    """Enum for names of available model architectures."""
    STANDARD = "standard"
    AUGMENTED_LLAMA = "augmented-llama"


class ModelFactory:
    """
    A factory for creating and loading models based on a specified architecture.
    Registers different model loaders and dispatches to the appropriate one.
    """
    _model_loaders: Dict[ModelArchitecture, Type["BaseModelLoader"]] = {}

    @classmethod
    def register(cls, architecture: ModelArchitecture):
        """
        Decorator to register a BaseModelLoader subclass with a given architecture.
        """
        def decorator(loader_class: Type["BaseModelLoader"]):
            if not issubclass(loader_class, BaseModelLoader):
                raise TypeError(f"Registered class must inherit from BaseModelLoader, got {loader_class.__name__}")
            cls._model_loaders[architecture] = loader_class
            return loader_class
        return decorator

    @classmethod
    def load(cls, cfg: DictConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads a model and its tokenizer based on the configuration.
        Dispatches to the specific loader registered for the given architecture.
        """
        architecture = cfg.model.architecture
        if not isinstance(architecture, ModelArchitecture):
             raise TypeError(f"Architecture must be a ModelArchitecture enum member, but got {type(architecture)}")

        loader_class = cls._model_loaders.get(architecture)
        if loader_class is None:
            raise ValueError(f"No model loader registered for architecture: '{architecture}'. "
                             f"Available architectures: {[arch.value for arch in cls._model_loaders.keys()]}")
        
        loader = loader_class(cfg)
        return loader.load()


class BaseModelLoader(ABC):
    """
    Abstract Base Class for model loaders.
    Provides generic methods for tokenizer creation, model loading, PEFT application,
    and post-processing, which can be overridden by subclasses.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_config = cfg.model.model_config
        self.model_precision = cfg.model.precision
        self.adapter_config = cfg.model.adapter_config
        self.adapter_path = getattr(cfg.model, "adapter_path", None)
        self.model_path = self.model_config.model_path

    @abstractmethod
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Main method to load the model and tokenizer.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def _create_tokenizer(self) -> PreTrainedTokenizer:
        """
        Creates and returns a tokenizer from the model path.
        Handles default padding token setup.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_dtype(self) -> torch.dtype:
        """
        Determines the torch.dtype based on the model_precision configuration.
        """
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return dtype_map.get(self.model_precision, torch.bfloat16)

    def _load_base_model_from_pretrained(self, model_class: Type[PreTrainedModel], tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Loads a base model using its from_pretrained method.
        """
        if not self.model_path:
            raise ValueError("model_path must be provided in model configuration.")

        model = retry_oserror(
            lambda: model_class.from_pretrained(
                self.model_path,
                config=self.model_config,
                dtype=self._get_dtype(),
                local_files_only=True,
            )
        )
        return model

    def _apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Applies a PEFT adapter to the model if adapter_config is provided.
        """
        from peft import PeftModel
        if self.adapter_path:
            # Load existing adapter from path (e.g. for memory training on top of finetuned model)
            abs_adapter_path = os.path.abspath(self.adapter_path)
            print(f"Loading PEFT adapter from {abs_adapter_path}...")
            model = PeftModel.from_pretrained(model, abs_adapter_path, is_trainable=True)
            model.print_trainable_parameters()
        elif self.adapter_config:
            # Initialize new adapter
            peft_config = hydra.utils.instantiate(self.adapter_config)
            model = get_peft_model(model, peft_config)
            print("PEFT adapter applied. Trainable parameters:")
            model.print_trainable_parameters()
        return model

    def _post_process_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Hook for subclasses to perform any final model adjustments.
        """
        return model
