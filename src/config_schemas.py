from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from src.tasks.base import BaseTaskConfig
from src.models.loaders import ModelArchitecture

@dataclass
class DatasetConfig:
    path: str = MISSING
    test_split_ratio: float = 0.125
    streaming: bool = False
    physical_batch_size: int = 4
    accumulation_steps: int = 1
    dataset_mode: str = "qa"  # qa, wikitext, attributes (legacy; use parser+biography_task instead)
    parser: Optional[Any] = field(default=None)
    biography_task: Optional[Any] = field(default=None)

@dataclass
class ModelConfigGroup:
    """A container for the base model config and an optional adapter config."""
    architecture: ModelArchitecture = MISSING
    model_config: Optional[Any] = field(default=MISSING)
    adapter_config: Optional[Any] = field(default=None)
    adapter_path: Optional[str] = field(default=None)
    precision: str = field(default="bf16")

@dataclass
class PromptConfig:
    contextual_qa_training: str   = "Biography: {biography}\nQuestion: {question}\nAnswer: {answer}"
    contextual_qa_generation: str = "Biography: {biography}\nQuestion: {question}\nAnswer: "
    direct_qa_training: str       = "Question: {question}\nAnswer: {answer}"
    direct_qa_generation: str     = "Question: {question}\nAnswer: "


@dataclass
class Config:
    """Top-level configuration schema."""

    project_name: str = "thesis"
    seed: int = field(default=42)

    model: ModelConfigGroup = field(default_factory=ModelConfigGroup)
    task: BaseTaskConfig = field(default=MISSING)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)

    experiment_path: str = field(default="scratch")
    base_output_dir: str = field(default=MISSING)

cs = ConfigStore.instance()
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
