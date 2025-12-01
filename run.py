import hydra
from omegaconf import DictConfig, OmegaConf

# Import the main functions from your training scripts
from src.training.finetune import main as finetune_main
from src.training.train_memory import main as train_memory_main

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.task.name == "finetune":
        print("Executing finetune task...")
        finetune_main(cfg)
    elif cfg.task.name == "train_memory":
        print("Executing train_memory task...")
        train_memory_main(cfg)
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}. Please choose 'finetune' or 'train_memory'.")

if __name__ == "__main__":
    run_app()
