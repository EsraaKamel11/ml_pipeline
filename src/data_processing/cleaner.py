import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging
import trafilatura
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class DataCleaner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.95):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def clean_text(self, text: str, remove_boilerplate: bool = False) -> str:
        if remove_boilerplate:
            extracted = trafilatura.extract(text)
            if extracted:
                text = extracted
        return ' '.join(text.strip().lower().split())

    def filter_sentences(self, text: str, min_length: int = 20) -> str:
        try:
            sentences = nltk.sent_tokenize(text)
            filtered = [s for s in sentences if len(s) >= min_length]
            return ' '.join(filtered)
        except Exception as e:
            self.logger.warning(f"Error in sentence tokenization: {e}")
            # Fallback: simple sentence splitting
            sentences = text.split('. ')
            filtered = [s for s in sentences if len(s) >= min_length]
            return '. '.join(filtered)

    def remove_duplicates(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        self.logger.info('Removing duplicates using semantic similarity...')
        texts = df[text_column].tolist()
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        keep = []
        seen = set()
        for idx, emb in enumerate(embeddings):
            if idx in seen:
                continue
            keep.append(idx)
            sims = util.pytorch_cos_sim(emb, embeddings)[0]
            for j, score in enumerate(sims):
                if j != idx and score > self.similarity_threshold:
                    seen.add(j)
        cleaned_df = df.iloc[keep].copy()
        self.logger.info(f'Reduced from {len(df)} to {len(cleaned_df)} rows.')
        return cleaned_df

    def process(self, df: pd.DataFrame, text_column: str = 'text', remove_boilerplate: bool = False, filter_sentences: bool = False, min_length: int = 20) -> pd.DataFrame:
        df[text_column] = df[text_column].astype(str).map(lambda t: self.clean_text(t, remove_boilerplate=remove_boilerplate))
        if filter_sentences:
            df[text_column] = df[text_column].map(lambda t: self.filter_sentences(t, min_length=min_length))
        df = self.remove_duplicates(df, text_column)
        return df 