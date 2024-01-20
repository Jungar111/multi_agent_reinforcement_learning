"""Module for handling different price_models."""
from enum import Enum


class PriceModel(Enum):
    """Enum for the city names and filenames."""

    ZERO_DIFF_MODEL = 1
    DIFF_MODEL = 2
    REG_MODEL = 3
