from transformers import Trainer, TrainingArguments
from peft import get_peft_model
import hydra.utils

def train_sequential(cfg, base_model, tokenizer, train_tokenized_dataset, test_tokenized_dataset, max_steps):
    if cfg.task.peft_method == "lora":
        peft_config = hydra.utils.instantiate(cfg.peft.lora)
        model = get_peft_model(base_model, peft_config)

    elif cfg.task.peft_method == "prefix":
        peft_config = hydra.utils.instantiate(cfg.peft.prefix)
        model = get_peft_model(base_model, peft_config)
    elif cfg.task.peft_method == "prompt":
        peft_config = hydra.utils.instantiate(cfg.peft.prompt)
        model = get_peft_model(base_model, peft_config)
    else:
        model = base_model
        
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=cfg.task.output_dir,
        logging_dir=cfg.task.log_dir,
        per_device_train_batch_size=cfg.dataset.physical_batch_size,
        gradient_accumulation_steps=cfg.dataset.accumulation_steps,
        num_train_epochs=cfg.task.epochs,
        max_steps=max_steps,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=cfg.task.eval_steps,
        seed=cfg.task.seed,
        ddp_find_unused_parameters=False,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()