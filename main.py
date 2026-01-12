import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf # Import OmegaConf

from src.config_schemas import Config
import src.tasks

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    cfg = OmegaConf.to_object(cfg)
    cfg.task.main(cfg)

if __name__ == "__main__":
    main()
