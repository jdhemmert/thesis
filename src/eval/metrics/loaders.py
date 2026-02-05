from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Any

from src.eval.metrics.base import MetricFactory, BaseMetricLoader, BaseMetric
from src.eval.metrics.evaluate_metric import EvaluateMetric, EvaluateMetricConfig
from src.eval.metrics.beaver import BeaverMetric, BeaverMetricConfig
from src.eval.metrics.perplexity import PerplexityMetric, PerplexityMetricConfig
from src.config_schemas import PromptConfig


@MetricFactory.register("evaluate")
class EvaluateMetricLoader(BaseMetricLoader):
    """Loads and instantiates the EvaluateMetric."""
    def __init__(self, metric_config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str, prompt_config: PromptConfig):
        super().__init__(metric_config, tokenizer, eval_dataset, output_dir, prompt_config)

    def load(self) -> BaseMetric:
        return EvaluateMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir
        )


@MetricFactory.register("beaver")
class BeaverMetricLoader(BaseMetricLoader):
    """Loads and instantiates the BeaverMetric."""
    def __init__(self, metric_config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str, prompt_config: PromptConfig):
        super().__init__(metric_config, tokenizer, eval_dataset, output_dir, prompt_config)

    def load(self) -> BaseMetric:
        return BeaverMetric(
            config=self.metric_config,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir
        )


@MetricFactory.register("perplexity")
class PerplexityMetricLoader(BaseMetricLoader):
    def __init__(self, metric_config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str, prompt_config: PromptConfig):
        super().__init__(metric_config, tokenizer, eval_dataset, output_dir)
        self.prompt_config = prompt_config

    def load(self) -> PerplexityMetric:
        # Convert DictConfig to the structured dataclass
        schema = OmegaConf.structured(PerplexityMetricConfig)
        config = OmegaConf.merge(schema, self.metric_config)
        
        return PerplexityMetric(config, self.tokenizer, self.eval_dataset, self.output_dir, self.prompt_config)

