from dataclasses import dataclass, field
from typing import Any, Optional, Dict

# Base class for common fields
@dataclass
class BaseModelConfig:
    model_path: str = "meta-llama/Llama-3.2-3B"
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

@dataclass
class AugmentedLlamaConfig(CustomLlamaConfig):
    """Configuration for the Augmented Llama model with soft prompts."""
    _target_: str = "src.configs.models.AugmentedLlamaConfig"
    custom_class_path: str = "src.models.augmented_llama.AugmentedLlamaForCausalLM"
    custom_params: Dict[str, Any] = field(default_factory=lambda: {"virtual_token_count": 10})

    # Configuration for soft prompt initialization
    initialization_context_text: Optional[str] = None
    initialization_noise_level: float = 0.0

