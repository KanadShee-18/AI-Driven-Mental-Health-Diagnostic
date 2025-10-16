# Importing functions from data_comparison.py for easy access to other directories

from .data_comparison import (
    compare_before_after,
    get_text_stats,
    print_text_stats
)

__all__ = [
    'compare_before_after',
    'get_text_stats',
    'print_text_stats'
]