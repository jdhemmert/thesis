from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, Any
from omegaconf import DictConfig


class MetricFactory:
    """
    A factory for creating and loading metric instances based on a specified metric type.
    Registers different metric loaders and dispatches to the appropriate one.
    """
    _metric_loaders: Dict[str, Type["BaseMetricLoader"]] = {}

    @classmethod
    def register(cls, metric_type: str):
        """
        Decorator to register a BaseMetricLoader subclass with a given metric type.
        """
        def decorator(loader_class: Type["BaseMetricLoader"]):
            if not issubclass(loader_class, BaseMetricLoader):
                raise TypeError(f"Registered class must inherit from BaseMetricLoader, got {loader_class.__name__}")
            cls._metric_loaders[metric_type] = loader_class
            return loader_class
        return decorator

    @classmethod
    def load(cls, metric_type: str, metric_config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str, prompt_config) -> "BaseMetric":
        """
        Loads a metric instance based on the configuration.
        Dispatches to the specific loader registered for the given metric type.
        """
        loader_class = cls._metric_loaders.get(metric_type)
        if loader_class is None:
            raise ValueError(f"No metric loader registered for metric type: '{metric_type}'. "
                             f"Available metric types: {list(cls._metric_loaders.keys())}")
        
        loader = loader_class(metric_config, tokenizer, eval_dataset, output_dir, prompt_config)
        return loader.load()


@dataclass
class BaseMetricConfig:
    _target_: str = "src.eval.metrics.base.BaseMetricConfig"


class BaseMetric(ABC):
    """Abstract base class for all metric implementations."""
    def __init__(self, config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str):
        self.config = config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

    @abstractmethod
    def compute_and_log_scores(self, model: Any, state: Any, metrics: Dict):
        """Abstract method to compute and log metric scores."""
        pass


class BaseMetricLoader(ABC):
    """Abstract base class for all metric loader implementations."""
    def __init__(self, metric_config: DictConfig, tokenizer: Any, eval_dataset: Any, output_dir: str, prompt_config):
        self.metric_config = metric_config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.prompt_config = prompt_config

    @abstractmethod
    def load(self) -> BaseMetric:
        """Abstract method to load and return a metric instance."""
        pass
