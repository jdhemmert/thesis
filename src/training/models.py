import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PrefixTuningConfig

def create_model(cfg, tokenizer):
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.precision, torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_path, dtype=dtype).to(device)

    if cfg.peft.method == "lora":
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    elif cfg.peft.method == "ptune":
        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            num_virtual_tokens=cfg.peft.num_virtual_tokens,
        )
        model = get_peft_model(model, peft_config)

    return model