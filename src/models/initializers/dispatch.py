from .generators import BaseEmbeddingGenerator, TokenizerEmbeddingGenerator, ExternalModelEmbeddingGenerator
from .poolers import BaseEmbeddingPooler, WindowedAveragePooler, PadPooler, TruncatePooler, IdentityPooler
from typing import Dict, Type

GENERATOR_MAP: Dict[str, Type[BaseEmbeddingGenerator]] = {
    "tokenizer": TokenizerEmbeddingGenerator,
    "external_model": ExternalModelEmbeddingGenerator,
}

POOLER_MAP: Dict[str, Type[BaseEmbeddingPooler]] = {
    "windowed_average": WindowedAveragePooler,
    "pad": PadPooler,
    "truncate": TruncatePooler,
    "identity": IdentityPooler,
}
