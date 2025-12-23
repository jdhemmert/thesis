import hydra
from omegaconf import DictConfig
from accelerate import Accelerator

from src.utils.model import load_model_from_config
from src.training.strategies.meta import train_meta
from src.utils.dataset import load_dataset_for_task
from src.training.strategies.sequential import train_sequential

@hydra.main(version_base=None, config_path="../../conf", config_name="config_train_memory")
def main(cfg: DictConfig):
    # The 'cfg' object passed here is the 'experiment' node from the main config.
    cfg_task = cfg.task
    accelerator = Accelerator()

    model_config = hydra.utils.instantiate(cfg.model)
    model, tokenizer = load_model_from_config(model_config)
    
    model.print_trainable_parameters()

    tokenized_dataset = load_dataset_for_task(
        task_type='self_distillation',
        dataset_path=cfg.dataset.path,
        tokenizer=tokenizer,
        max_length=cfg_task.max_length,
        sample_n=cfg_task.sample_n,
        sample_strategy=cfg_task.sample_strategy,
        seed=cfg_task.seed,
        drop_text_columns=False
    )

    if cfg_task.strategy.name == "meta":
        train_meta(cfg, model, tokenized_dataset, tokenizer)
    else:
        train_sequential(cfg, model, tokenized_dataset, tokenizer)

if __name__ == "__main__":
    main()