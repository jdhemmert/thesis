import torch
from torch import nn
import hydra
from typing import Optional, Any, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, LlamaConfig, PretrainedConfig
from peft import get_peft_model, PeftConfig

from src.models.augmented_llama import AugmentedLlamaForCausalLM, AugmentedLlamaConfig

MODEL_LOADER_REGISTRY = {}

def register_model_loader(config_class: type):
    """Decorator to register a model loader function for a specific base model config type."""
    def decorator(loader_fn):
        if config_class in MODEL_LOADER_REGISTRY:
            raise ValueError(f"Model loader already registered for config type: {config_class.__name__}")
        MODEL_LOADER_REGISTRY[config_class] = loader_fn
        return loader_fn
    return decorator

@register_model_loader(LlamaConfig)
def _load_base_llama(config: LlamaConfig, precision: str, **kwargs):
    """Loads a standard base Llama model."""
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(precision, torch.bfloat16)
    
    model_path = config.model_path
    if not model_path:
        raise ValueError("model_path must be provided to load a base Llama model.")

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype)
    return model

@register_model_loader(AugmentedLlamaConfig)
def _load_augmented_llama(config: AugmentedLlamaConfig, precision: str, **kwargs):
    """Loads an AugmentedLlamaForCausalLM model and initializes its soft prompt if configured."""
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(precision, torch.bfloat16)

    model_path = config.model_path
    if not model_path:
        raise ValueError("model_path must be provided to load an AugmentedLlama model.")

    model = AugmentedLlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype)
    
    return model

def load_model_from_config(model_config: Any, model_precision: str, adapter_config: Optional[PeftConfig] = None):
    """
    Loads a model and its tokenizer based on the compositional configuration.
    1. Loads the base model using the `model_config`.
    2. Applies a PEFT adapter if `adapter_config` is present.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_type = type(model_config)
    loader_fn = MODEL_LOADER_REGISTRY.get(config_type)
    
    if loader_fn is None:
        # Fallback for inheritance (e.g., if a loader is registered for LlamaConfig and we get a subclass)
        for base_class in config_type.__mro__[1:]:
            if base_class in MODEL_LOADER_REGISTRY:
                loader_fn = MODEL_LOADER_REGISTRY[base_class]
                break

    if loader_fn is None:
        raise ValueError(f"No model loader registered for config type: {config_type.__name__}. "
                         f"Available types: {[c.__name__ for c in MODEL_LOADER_REGISTRY.keys()]}")

    base_model = loader_fn(model_config, precision=model_precision)

    if adapter_config:
        peft_config = hydra.utils.instantiate(adapter_config)
        final_model = get_peft_model(base_model, peft_config)
        final_model.print_trainable_parameters()
    else:
        final_model = base_model

    if isinstance(model_config, AugmentedLlamaConfig) and model_config.initialization_context_text:
        print("Performing soft prompt initialization...")
        initialize_soft_prompt(
            model=final_model,
            tokenizer=tokenizer,
            context_text=model_config.initialization_context_text,
            noise_level=model_config.initialization_noise_level
        )

    return final_model, tokenizer

def initialize_soft_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context_text: str,
    noise_level: float = 0.0,
):
    """
    Initializes the model's internal soft prompt with embeddings derived from a context sentence.
    """
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