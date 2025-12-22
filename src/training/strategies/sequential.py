import torch
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer

from src.training.losses import self_distillation_loss
from src.training.callbacks import ExtrinsicValidationCallback
from src.utils.dataset import SelfDistillationDataCollator

class MemoryBankTrainer(Trainer):
    def __init__(self, loss_type, loss_alpha, temperature, **kwargs):
        self.loss_type = loss_type
        self.alpha = loss_alpha
        self.T = temperature
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        """
        Custom loss function for self-oracle training of the memory bank.
        """
        oracle_outputs = model(
            input_ids=inputs["oracle_input_ids"],
            attention_mask=inputs["oracle_attention_mask"],
            labels=inputs["oracle_input_ids"]
        )

        memory_outputs = model(
            input_ids=inputs["memory_input_ids"],
            attention_mask=inputs["memory_attention_mask"],
            labels=inputs["memory_input_ids"]
        )
        
        loss = self_distillation_loss(
            oracle_logits=oracle_outputs.logits,
            memory_logits=memory_outputs.logits,
            temperature=self.T,
            alpha=self.alpha,
            loss_type=self.loss_type
        )

        return (loss, memory_outputs) if return_outputs else loss

def train_sequential(cfg: DictConfig, model, tokenizer, tokenized_dataset):
    """
    Implements the sequential training strategy using the MemoryBankTrainer.
    """
    cfg_task = cfg.task
    print("Using standard fine-tuning strategy.")
    
    training_args = TrainingArguments(
        output_dir=cfg_task.output_dir,
        learning_rate=cfg_task.learning_rate,
        num_train_epochs=cfg_task.num_train_epochs,
        per_device_train_batch_size=cfg.dataset.physical_batch_size,
        gradient_accumulation_steps=cfg.dataset.accumulation_steps,
        seed=cfg_task.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=cfg_task.eval_steps,
        save_strategy="steps",
    )

    text_columns = [col for col in tokenized_dataset.column_names if tokenized_dataset.features[col].dtype == 'string']
    trainer_dataset = tokenized_dataset.remove_columns(text_columns)

    trainer = MemoryBankTrainer(
        model=model,
        loss_type=cfg_task.loss.loss_type,
        loss_alpha=cfg_task.loss.alpha,
        temperature=cfg_task.loss.temperature,
        args=training_args,
        train_dataset=trainer_dataset,
        eval_dataset=trainer_dataset,
        callbacks=[ExtrinsicValidationCallback(tokenized_dataset, tokenizer, cfg_task)],
        tokenizer=tokenizer,
        data_collator=SelfDistillationDataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model()
