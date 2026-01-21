from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from peft import PeftConfig

from src.tasks.base import BaseTaskConfig

@dataclass
class DatasetConfig:
    path: str = "data/qa_finetune_dataset_12k.jsonl"
    test_split_ratio: float = 0.125
    streaming: bool = True
    physical_batch_size: int = 4
    accumulation_steps: int = 1

@dataclass
class ModelConfigGroup:
    """A container for the base model config and an optional adapter config."""
    model_config: Optional[Any] = field(default=MISSING)
    adapter_config: Optional[PeftConfig] = field(default=None)
    precision: str = field(default="bf16")

@dataclass
class Config:
    """Top-level configuration schema."""

    project_name: str = "thesis"
    seed: int = field(default=42)

    model: ModelConfigGroup = field(default_factory=ModelConfigGroup)
    task: BaseTaskConfig = field(default=MISSING)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    experiment_path: str = field(default="scratch")
    base_output_dir: str = field(default=MISSING)

