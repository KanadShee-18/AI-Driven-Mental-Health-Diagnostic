# Importing functions from text_cleaner.py for easy access to other directories

from .text_cleaner import (
    cleaned_text,
    remove_special_chars,
    to_lowercase,
    remove_extra_spaces,
    preprocess_dataframe
)

__all__ = [
    'cleaned_text',
    'remove_special_chars',
    'to_lowercase',
    'remove_extra_spaces',
    'preprocess_dataframe'
]