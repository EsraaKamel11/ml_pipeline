import pandas as pd
import logging
import sqlite3

class StorageManager:
    def __init__(self, db_path: str = 'processed_data.sqlite'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = db_path

    def save_to_sqlite(self, df: pd.DataFrame, table_name: str = 'data') -> None:
        self.logger.info(f'Saving data to SQLite: {self.db_path}, table: {table_name}')
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        self.logger.info('Data saved to SQLite.')

    def load_from_sqlite(self, table_name: str = 'data') -> pd.DataFrame:
        self.logger.info(f'Loading data from SQLite: {self.db_path}, table: {table_name}')
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        return df

    def save_to_parquet(self, df: pd.DataFrame, path: str) -> None:
        self.logger.info(f'Saving data to Parquet: {path}')
        df.to_parquet(path, index=False)
        self.logger.info('Data saved to Parquet.')

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        self.logger.info(f'Loading data from Parquet: {path}')
        return pd.read_parquet(path) 