import os
import hydra
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from dotenv import load_dotenv

from src.utils.dataset import load_dataset_for_task
from src.utils.model import load_model_from_config

load_dotenv()

@hydra.main(version_base=None, config_path="../../conf", config_name="config_finetune")
def main(cfg: DictConfig):
    if cfg.task.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.task.gpu_ids

    model_config = hydra.utils.instantiate(cfg.model)
    model, tokenizer = load_model_from_config(model_config)
    model.print_trainable_parameters()

    tokenized_dataset = load_dataset_for_task(
        task_type='finetune',
        dataset_path=cfg.dataset.path,
        tokenizer=tokenizer,
        max_length=cfg.task.max_seq_length,
        sample_n=cfg.task.get('sample_n'),
        seed=cfg.task.seed,
        drop_text_columns=True
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=cfg.task.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir=cfg.task.output_dir,
        learning_rate=cfg.task.learning_rate,
        num_train_epochs=cfg.task.epochs,
        per_device_train_batch_size=cfg.dataset.physical_batch_size,
        gradient_accumulation_steps=cfg.dataset.accumulation_steps,
        seed=cfg.task.seed,
        eval_strategy="steps",
        eval_steps=cfg.task.eval_steps,
        save_strategy="steps",
        logging_steps=10,
        fp16=model_config.precision == 'fp16',
        bf16=model_config.precision == 'bf16',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
