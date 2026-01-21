import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf # Import OmegaConf

from src.config_schemas import Config
import src.tasks

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cwd = HydraConfig.get().runtime.output_dir
    cfg = hydra.utils.instantiate(cfg, _convert_="partial")

    cfg.task.main(cwd, cfg)

if __name__ == "__main__":
    main()
