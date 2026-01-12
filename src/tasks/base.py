from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

def register_task(name, group=None):
    def decorator(cls):
        cs = ConfigStore.instance()
        cs.store(name=name, group=group, node=cls)
        return cls
    return decorator

@register_task(name="default_task", group="task")
@dataclass
class BaseTaskConfig:
    _target_: str = "src.tasks.base.BaseTask"

class BaseTask:
    def __init__(self, **kwargs):
        self.config = BaseTaskConfig(**kwargs)

    def main(self, cfg):
        print("This is an empty task. You will want to select one to run, "
              "either by setting one explicitly via the `task=...` flag, "
              "or by choosing an experiment via the `+experiment=...` flag "
              "to select a pre-defined experiment in `conf/experiment`.")
