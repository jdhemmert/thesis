from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Any

from src.eval.metrics.base import MetricFactory, BaseMetricLoader, BaseMetric
from src.eval.metrics.rouge import RougeMetric, RougeMetricConfig


@MetricFactory.register("rouge")
class RougeMetricLoader(BaseMetricLoader):
    """Loads and instantiates the RougeMetric."""
    def load(self) -> BaseMetric:
        return RougeMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir
        )
