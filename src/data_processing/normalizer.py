import pandas as pd
import logging
from transformers import AutoTokenizer

class Normalizer:
    def __init__(self, model_name: str = 'gpt2'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def normalize(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        self.logger.info('Normalizing and tokenizing text...')
        df['normalized_text'] = df[text_column].astype(str).map(self._normalize_text)
        df['tokens'] = df['normalized_text'].map(self.tokenizer.encode)
        return df

    def _normalize_text(self, text: str) -> str:
        return text.strip().replace('\r', '').replace('\n', ' ') 
