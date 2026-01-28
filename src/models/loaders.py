import torch
from typing import Tuple

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, LlamaConfig

from src.models.base import BaseModelLoader, ModelFactory
from src.models.augmented_llama import AugmentedLlamaForCausalLM, AugmentedLlamaConfig


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
        Initializes the soft prompt for AugmentedLlamaForCausalLM if configured.
        """
        if not isinstance(model, AugmentedLlamaForCausalLM):
            raise TypeError("Model must be AugmentedLlamaForCausalLM for soft prompt initialization.")

        if self.model_config.initialization_context_text:
            print("Performing soft prompt initialization...")
            context_text = self.model_config.initialization_context_text
            noise_level = self.model_config.initialization_noise_level

            context_ids = tokenizer(context_text, return_tensors="pt").input_ids
            virtual_token_count = context_ids.shape[1]

            if not hasattr(model, 'model') or not hasattr(model.model, 'rebuild_virtual_prompt'):
                raise TypeError("The provided model is not a compatible AugmentedLlamaForCausalLM instance.")
            
            model.model.rebuild_virtual_prompt(virtual_token_count)

            with torch.no_grad():
                embed_tokens_layer = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
                initial_weights = embed_tokens_layer(context_ids.to(model.device)).squeeze(0)

            if noise_level > 0.0:
                noise = torch.randn_like(initial_weights) * noise_level
                initial_weights += noise
                print(f"Applied Gaussian noise with std_dev={noise_level} to initial soft prompt.")

            model.model.set_virtual_prompt_weights(initial_weights)
            print(f"Initialized soft prompt with {virtual_token_count} tokens from context.")
        return model
