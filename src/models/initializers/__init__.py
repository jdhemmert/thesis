from enum import Enum
from typing import Dict, Type

from .base import BaseInitializer
from .random import RandomInitializer
from .composable import ComposableInitializer


class InitializerName(Enum):
    """
    Enum for names of available prompt initializers.
    """
    RANDOM = "random"
    COMPOSABLE = "composable"


INITIALIZER_MAP: Dict[str, Type[BaseInitializer]] = {
    "random": RandomInitializer,
    "composable": ComposableInitializer,
}
