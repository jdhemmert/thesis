import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, PeftConfig # Added get_peft_model and PeftConfig for type hinting

def load_model_and_tokenizer(
    model_path: str,
    precision: str = "bf16",
    lora_config: PeftConfig = None,
    memory_config: PeftConfig = None
):
    """
    Loads a model and tokenizer with specified precision and optional PEFT adapters.

    Args:
        model_path (str): The path to the base model.
        precision (str, optional): The precision to use for model loading ('fp32', 'fp16', 'bf16'). Defaults to "bf16".
        lora_config (PeftConfig, optional): The PEFT configuration for the LoRA adapter. Defaults to None.
        memory_config (PeftConfig, optional): The PEFT configuration for the memory adapter. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine data type for model loading
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.bfloat16)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)

    peft_configs = []
    if lora_config:
        peft_configs.append(("lora_adapter", lora_config))
    if memory_config:
        peft_configs.append(("memory_adapter", memory_config))

    if peft_configs:
        # Apply the first PEFT config using get_peft_model
        adapter_name, config = peft_configs[0]
        model = get_peft_model(model, config, adapter_name=adapter_name)

        # Add any subsequent PEFT configs using add_adapter
        for adapter_name, config in peft_configs[1:]:
            model.add_adapter(adapter_name, config)

    return model, tokenizer
