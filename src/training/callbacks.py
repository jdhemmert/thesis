import os
import json

from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import torch
from transformers import TrainerCallback
from omegaconf import DictConfig, OmegaConf

from src.eval.metrics.beaver import BeaverMetricConfig
from src.eval.metrics.base import MetricFactory, BaseMetric


class ExtrinsicValidationFrequency(Enum):
    ALWAYS = "always"
    NEVER  = "never"
    END    = "end"


@dataclass
class ExtrinsicValidationConfig:
    frequency: ExtrinsicValidationFrequency = ExtrinsicValidationFrequency.NEVER
    metric_type: str = "evaluate"
    metric_config: Optional[Any] = None


class ExtrinsicValidationCallback(TrainerCallback):
    
    def __init__(self, config: ExtrinsicValidationConfig, eval_dataset, tokenizer, output_dir, prompt_config):
        self.config = config
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.prompt_config = prompt_config

        self.metric: BaseMetric = MetricFactory.load(
            metric_type=self.config.metric_type,
            metric_config=self.config.metric_config,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            prompt_config=self.prompt_config
        )

    def on_evaluate(self, args, state, control, **kwargs):
        if self.config.frequency == ExtrinsicValidationFrequency.ALWAYS:
            self._run_validation(state, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.config.frequency != ExtrinsicValidationFrequency.NEVER:
            self._run_validation(state, **kwargs)

    def _run_validation(self, state, **kwargs):
        model = kwargs.get("model")
        metrics = kwargs.get("metrics", {})

        self.metric.compute_and_log_scores(model, state, metrics)

