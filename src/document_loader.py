"""Load csv documents from data folder"""

import pandas as pd
import os

class DocumentLoader:
    def __init__(self, documents_path: str):
        self.documents_path = documents_path

    def load_documents(self) -> list[str]:
        df = pd.read_csv(self.documents_path)
        return df["text"].tolist()
