from typing import Optional
from dataclasses import dataclass
import datasets
from omegaconf import DictConfig

@dataclass
class BaseContextConfig:
    use_dataset_context: bool = False
    context_text: Optional[str] = None
    context_source_column: str = "biography"
    max_context_samples: Optional[int] = None

def get_context_text_from_config(config: BaseContextConfig, dataset_path: Optional[str]) -> str:
    """
    Retrieves context text either from a static string in the config or by loading it from a dataset.
    """
    if not config.use_dataset_context:
        if not config.context_text:
            raise ValueError("`context_text` must be provided in config if `use_dataset_context` is false.")
        return config.context_text
    
    if not dataset_path:
        raise ValueError("`dataset_path` must be provided if `use_dataset_context` is true.")
        
    print("Loading dataset to build initialization context...")
    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]
    
    if config.context_source_column not in dataset.column_names:
        raise ValueError(f"Column '{config.context_source_column}' not found in dataset: {dataset.column_names}.")
        
    sample_range = range(len(dataset))
    if config.max_context_samples and config.max_context_samples < len(dataset):
        sample_range = range(config.max_context_samples)

    full_text = "\n".join([dataset[i][config.context_source_column] for i in sample_range])
    return full_text
