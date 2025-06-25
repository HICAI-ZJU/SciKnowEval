"""
Evaluation module for SciKnowEval benchmark.

Contains task definitions, metrics, and utility functions for evaluating
scientific knowledge of Large Language Models.
"""

from .define import get_task_func, get_task_data, reformat_result
from .metrics import *

__all__ = [
    "get_task_func",
    "get_task_data",
    "reformat_result",
]