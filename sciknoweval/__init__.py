"""
SciKnowEval: Evaluating Multi-level Scientific Knowledge of Large Language Models

A comprehensive benchmark for evaluating Large Language Models across five levels
of scientific knowledge: memory, comprehension, reasoning, discernment, and application.
"""

__version__ = "0.1.0"
__author__ = "HICAI-ZJU"
__description__ = "Evaluating Multi-level Scientific Knowledge of Large Language Models"

from .evaluation.define import get_task_func, get_task_data, reformat_result
from .evaluation.metrics import *

__all__ = [
    "get_task_func",
    "get_task_data", 
    "reformat_result",
]