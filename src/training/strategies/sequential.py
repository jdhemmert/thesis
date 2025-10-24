from transformers import Trainer, TrainingArguments

def train_sequential(args, model, train_tokenized_dataset, test_tokenized_dataset, max_steps):
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.physical_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        seed=args.seed,
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