import torch
from typing import Tuple

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, LlamaConfig

from src.models.base import BaseModelLoader, ModelFactory
from src.models.augmented_llama import AugmentedLlamaForCausalLM, AugmentedLlamaConfig
from src.models.prompt_initializers import InitializerFactory


@ModelFactory.register("standard")
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


@ModelFactory.register("augmented-llama")
class AugmentedLlamaLoader(BaseModelLoader):
    """
    Loader for the custom AugmentedLlamaForCausalLM model.
    Overrides _load_model and _post_process_model for custom logic.
    """
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = self._create_tokenizer()
        
        model = self._load_model(tokenizer)
        model = self._apply_peft(model)
        model = self._post_process_model(model, tokenizer)
        return model, tokenizer

    def _load_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Loads the AugmentedLlamaForCausalLM.
        """
        if not isinstance(self.model_config, AugmentedLlamaConfig):
            raise TypeError("model_config must be an instance of AugmentedLlamaConfig for AugmentedLlamaLoader.")
        
        return self._load_base_model_from_pretrained(AugmentedLlamaForCausalLM, tokenizer)

    def _post_process_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Initializes the soft prompt for AugmentedLlamaForCausalLM using the configured method.
        """
        if not isinstance(model, AugmentedLlamaForCausalLM):
            raise TypeError("Model must be AugmentedLlamaForCausalLM for soft prompt initialization.")

        # Ensure the model config is the correct type
        if not isinstance(self.model_config, AugmentedLlamaConfig):
            raise TypeError("model_config must be an instance of AugmentedLlamaConfig.")

        # Check if an initialization method is specified
        init_method = getattr(self.model_config, "initialization_method", None)
        if not init_method:
            return model # No initialization requested

        print(f"Performing soft prompt initialization with method: '{init_method}'...")
        
        initializer = InitializerFactory.create(init_method)
        
        # Prepare arguments for the initializer
        init_kwargs = {
            "model": model.model, # Pass the underlying LlamaModel
            "tokenizer": tokenizer,
            "virtual_token_count": self.model_config.virtual_token_count,
            "context_text": getattr(self.model_config, "initialization_context_text", None),
            "noise_level": getattr(self.model_config, "initialization_noise_level", 0.0),
        }
        
        # Generate the initial weights
        initial_weights = initializer.initialize(**init_kwargs)
        
        # Rebuild the virtual prompt with the new weights
        model.model.rebuild_virtual_prompt(weights=initial_weights)
        
        print(f"Initialized soft prompt with {model.model.virtual_token_count} tokens.")
        
        return model
