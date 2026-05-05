import torch
from typing import Tuple

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from src.models.base import BaseModelLoader, ModelFactory, ModelArchitecture
from src.models.augmented_llama import AugmentedLlamaForCausalLM, AugmentedLlamaConfig
from src.models.augmented_gpt2 import AugmentedGPT2LMHeadModel, AugmentedGPT2Config


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


@ModelFactory.register(ModelArchitecture.AUGMENTED_GPT2)
class AugmentedGPT2Loader(BaseModelLoader):
    """
    Loader for the custom AugmentedGPT2LMHeadModel.
    """
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = self._create_tokenizer()
        model = self._load_model(tokenizer)
        model = self._apply_peft(model)
        return model, tokenizer

    def _load_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        if not isinstance(self.model_config, AugmentedGPT2Config):
            raise TypeError("model_config must be an instance of AugmentedGPT2Config for AugmentedGPT2Loader.")

        return self._load_base_model_from_pretrained(AugmentedGPT2LMHeadModel, tokenizer)
