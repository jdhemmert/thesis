import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model_and_tokenizer(model_path: str, adapter_path: str = None, precision: str = "bf16"):
    """
    Loads a model and tokenizer with specified precision and optional PEFT adapter.

    Args:
        model_path (str): The path to the base model.
        adapter_path (str, optional): The path to the PEFT adapter. Defaults to None.
        precision (str, optional): The precision to use for model loading ('fp32', 'fp16', 'bf16'). Defaults to "bf16".

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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)

    # Load PEFT adapter if provided
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer
