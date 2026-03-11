from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Any

from src.eval.metrics.base import MetricFactory, BaseMetricLoader, BaseMetric
from src.eval.metrics.evaluate_metric import EvaluateMetric, EvaluateMetricConfig
from src.eval.metrics.beaver import BeaverMetric, BeaverMetricConfig
from src.eval.metrics.perplexity import PerplexityMetric, PerplexityMetricConfig
from src.config_schemas import PromptConfig


# TODO: these are all essentially transparent passthroughs - maybe parameterize?
@MetricFactory.register("evaluate")
class EvaluateMetricLoader(BaseMetricLoader):
    """Loads and instantiates the EvaluateMetric."""

    def load(self) -> BaseMetric:
        return EvaluateMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            prompt_config=self.prompt_config,
        )


@MetricFactory.register("beaver")
class BeaverMetricLoader(BaseMetricLoader):
    """Loads and instantiates the BeaverMetric."""

    def load(self) -> BaseMetric:
        return BeaverMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            prompt_config=self.prompt_config,
        )


@MetricFactory.register("perplexity")
class PerplexityMetricLoader(BaseMetricLoader):
    """Loads and instantiates the PerplexityMetric."""

    def load(self) -> PerplexityMetric:
        return PerplexityMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            prompt_config=self.prompt_config,
        )

