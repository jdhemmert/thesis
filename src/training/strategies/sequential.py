from transformers import Trainer, TrainingArguments

def train_sequential(cfg, model, train_tokenized_dataset, test_tokenized_dataset, max_steps):
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        logging_dir=cfg.log_dir,
        per_device_train_batch_size=cfg.dataset.physical_batch_size,
        gradient_accumulation_steps=cfg.dataset.accumulation_steps,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        seed=cfg.seed,
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