import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PrefixTuningConfig

def create_model(cfg, tokenizer):
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.task.precision, torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_path, dtype=dtype).to(device)

    return model