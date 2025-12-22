from dataclasses import dataclass, field
from typing import Any, Optional, Dict

# Base class for common fields
@dataclass
class BaseModelConfig:
    model_path: str = "meta-llama/Llama-3-8B-Instruct"
    precision: str = "bf16"

@dataclass
class PeftLlamaConfig(BaseModelConfig):
    """Configuration for a PEFT-adapted Llama model."""
    _target_: str = "src.configs.models.PeftLlamaConfig" # Hydra target
    lora_config: Optional[Dict[str, Any]] = None # This will be hydra.instantiated
    memory_config: Optional[Dict[str, Any]] = None # This will be hydra.instantiated

@dataclass
class CustomLlamaConfig(BaseModelConfig):
    """Configuration for a custom LlamaForCausalLM subclass."""
    _target_: str = "src.configs.models.CustomLlamaConfig" # Hydra target
    custom_class_path: str = "src.models.CustomLlamaWithMemory" # Path to your custom class
    custom_params: Dict[str, Any] = field(default_factory=dict) # Params for your custom class's __init__ or from_pretrained
