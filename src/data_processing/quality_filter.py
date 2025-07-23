import pandas as pd
import logging
from typing import Callable, Optional

class QualityFilter:
    def __init__(self, min_length: int = 20, language_check: Optional[Callable[[str], bool]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_length = min_length
        self.language_check = language_check

    def filter(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        self.logger.info('Filtering low-quality content...')
        initial_len = len(df)
        df = df[df[text_column].str.len() >= self.min_length]
        if self.language_check:
            df = df[df[text_column].map(self.language_check)]
        self.logger.info(f'Reduced from {initial_len} to {len(df)} rows after filtering.')
        return df 