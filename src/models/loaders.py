import torch
from typing import Tuple
from enum import Enum

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, LlamaConfig

from src.models.base import BaseModelLoader, ModelFactory
from src.models.augmented_llama import AugmentedLlamaForCausalLM, AugmentedLlamaConfig


class ModelArchitecture(str, Enum):
    """Enum for names of available model architectures."""
    STANDARD = "standard"
    AUGMENTED_LLAMA = "augmented-llama"


@ModelFactory.register(ModelArchitecture.STANDARD)
class StandardLoader(BaseModelLoader):
    """
    Loader for standard Hugging Face AutoModelForCausalLM models.
    Leverages the generic loading logic from BaseModelLoader.
    """
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = self._create_tokenizer()
        model = self._load_model(tokenizer)
        model = self._apply_peft(model)
        model = self._post_process_model(model, tokenizer)
        return model, tokenizer

    def _load_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Loads the standard AutoModelForCausalLM.
        """
        return self._load_base_model_from_pretrained(AutoModelForCausalLM, tokenizer)


@ModelFactory.register(ModelArchitecture.AUGMENTED_LLAMA)
class AugmentedLlamaLoader(BaseModelLoader):
    """
    Loader for the custom AugmentedLlamaForCausalLM model.
    Overrides _load_model. The post-processing hook is now a no-op.
    """
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = self._create_tokenizer()
        model = self._load_model(tokenizer)
        model = self._apply_peft(model)
        return model, tokenizer

    def _load_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Loads the AugmentedLlamaForCausalLM.
        """
        if not isinstance(self.model_config, AugmentedLlamaConfig):
            raise TypeError("model_config must be an instance of AugmentedLlamaConfig for AugmentedLlamaLoader.")
        
        return self._load_base_model_from_pretrained(AugmentedLlamaForCausalLM, tokenizer)
