import torch
import importlib
import hydra
from typing import Optional, Any, Dict

from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, PeftConfig

from src.configs.models import PeftLlamaConfig, CustomLlamaConfig, AugmentedLlamaConfig

MODEL_LOADER_REGISTRY = {}

def register_model_loader(config_class: type):
    """Decorator to register a model loader function by its config class type."""
    def decorator(loader_fn):
        if config_class in MODEL_LOADER_REGISTRY:
            raise ValueError(f"Model loader already registered for config type: {config_class.__name__}")
        MODEL_LOADER_REGISTRY[config_class] = loader_fn
        return loader_fn
    return decorator

def _get_class_from_string(class_path: str):
    """Dynamically imports and returns a class from its fully qualified string path."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

@register_model_loader(PeftLlamaConfig)
def _load_peft_model(config: PeftLlamaConfig, tokenizer: AutoTokenizer):
    """Loads a base Llama model and applies PEFT adapters based on config."""
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(config.precision, torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=dtype).to(device)

    peft_configs_to_apply = []
    if config.lora_config:
        peft_configs_to_apply.append(("lora_adapter", hydra.utils.instantiate(config.lora_config)))
    if config.memory_config:
        peft_configs_to_apply.append(("memory_bank", hydra.utils.instantiate(config.memory_config)))

    if peft_configs_to_apply:
        adapter_name, peft_config = peft_configs_to_apply[0]
        model = get_peft_model(model, peft_config, adapter_name=adapter_name)

        for adapter_name, peft_config in peft_configs_to_apply[1:]:
            model.add_adapter(adapter_name, peft_config)
            
        all_adapter_names = [name for name, _ in peft_configs_to_apply]
        model.set_adapter(all_adapter_names)
        model.trainable_adapters = all_adapter_names

    return model

@register_model_loader(CustomLlamaConfig)
def _load_custom_model(config: CustomLlamaConfig, tokenizer: AutoTokenizer):
    """Loads a custom LlamaForCausalLM subclass based on config."""
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(config.precision, torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    custom_model_class = _get_class_from_string(config.custom_class_path)
    
    model = custom_model_class.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        **config.custom_params
    ).to(device)

    return model

@register_model_loader(AugmentedLlamaConfig)
def _load_augmented_llama(config: AugmentedLlamaConfig, tokenizer: AutoTokenizer):
    """Loads an AugmentedLlamaForCausalLM model and initializes its soft prompt if configured."""
    model = _load_custom_model(config, tokenizer)

    # Perform the specific initialization for the augmented model
    if config.initialization_context_text:
        print("Performing soft prompt initialization...")
        initialize_soft_prompt(
            model=model,
            tokenizer=tokenizer,
            context_text=config.initialization_context_text,
            noise_level=config.initialization_noise_level
        )
    
    return model

def load_model_from_config(model_config):
    """
    Loads a model and its tokenizer based on a provided configuration object.
    Dispatches to appropriate loader function using a type-based registry.
    """
    config_type = type(model_config)
    loader_fn = None
    if config_type in MODEL_LOADER_REGISTRY:
        loader_fn = MODEL_LOADER_REGISTRY[config_type]
    else:
        # Check for parent classes in the registry for inheritance
        for base_class in config_type.__mro__[1:]:
            if base_class in MODEL_LOADER_REGISTRY:
                loader_fn = MODEL_LOADER_REGISTRY[base_class]
                break
    
    if loader_fn is None:
        raise ValueError(f"No model loader registered for config type: {type(model_config).__name__}. "
                         f"Available types: {[c.__name__ for c in MODEL_LOADER_REGISTRY.keys()]}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = loader_fn(model_config, tokenizer)
    
    return model, tokenizer

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

    # Assumes the model has a 'model.rebuild_virtual_prompt' method.
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

def prepare_identical_context_embeddings(
    context_text: str,
    prompt_text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    virtual_token_count: Optional[int] = None
) -> tuple[nn.Embedding, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepares an nn.Embedding object for virtual tokens and prompt embeddings
    for a "no-op" comparison in the PoC script.
    """
    device = model.device

    if virtual_token_count is None:
        context_ids_for_length = tokenizer(context_text, return_tensors="pt").input_ids
        virtual_token_count = context_ids_for_length.shape[1]

    full_prompt_ids = tokenizer(context_text + prompt_text, return_tensors="pt").input_ids.to(device)

    context_ids_from_full = full_prompt_ids[:, :virtual_token_count]
    prompt_ids_from_full = full_prompt_ids[:, virtual_token_count:]
    
    with torch.no_grad():
        embed_tokens_layer = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
        initial_virtual_token_tensor = embed_tokens_layer(context_ids_from_full)
        prompt_embeds = embed_tokens_layer(prompt_ids_from_full)

    # Create an nn.Embedding layer for the virtual prompt and load the weights
    initial_virtual_prompt_embedding = nn.Embedding(virtual_token_count, model.config.hidden_size)
    initial_virtual_prompt_embedding.weight.data.copy_(initial_virtual_token_tensor.squeeze(0))

    prompt_attention_mask = torch.ones_like(prompt_ids_from_full)

    return initial_virtual_prompt_embedding, prompt_embeds, prompt_attention_mask, full_prompt_ids, virtual_token_count