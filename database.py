import sqlite3
import pandas as pd
import logging
import os

# Ensure the data directory exists
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Database:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(data_dir, "history.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        number REAL,
                        prediction TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    def save_data(self, timestamp, number, prediction=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO history (timestamp, number, prediction) VALUES (?, ?, ?)",
                             (timestamp, number, prediction))
                conn.commit()
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")

    def load_data(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM history", conn)
                return df
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            return pd.DataFrame(columns=['timestamp', 'number', 'prediction'])