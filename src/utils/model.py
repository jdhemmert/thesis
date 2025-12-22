import torch
import importlib
import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, PeftConfig # PeftConfig is used for type hinting for nested configs

from src.configs.models import PeftLlamaConfig, CustomLlamaConfig

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
def _load_peft_model(config: PeftLlamaConfig):
    """Loads a base Llama model and applies PEFT adapters based on config."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
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
def _load_custom_model(config: CustomLlamaConfig):
    """Loads a custom LlamaForCausalLM subclass based on config."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.precision, torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    custom_model_class = _get_class_from_string(config.custom_class_path)
    
    model = custom_model_class.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        **config.custom_params
    ).to(device)

    return model

def load_model_from_config(model_config):
    """
    Loads a model and its tokenizer based on a provided configuration object.
    Dispatches to appropriate loader function using a type-based registry.
    """
    config_type = type(model_config)
    if config_type not in MODEL_LOADER_REGISTRY:
        raise ValueError(f"No model loader registered for config type: {config_type.__name__}. "
                         f"Available types: {[c.__name__ for c in MODEL_LOADER_REGISTRY.keys()]}")
    
    loader_fn = MODEL_LOADER_REGISTRY[config_type]
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = loader_fn(model_config)
    
    return model, tokenizer
