"""Ladda csv dokument från data mapp"""

import pandas as pd

class DocumentLoader:
    def __init__(self, documents_path: str):
        self.documents_path = documents_path

    def load_documents(self) -> list[str]:
        df = pd.read_csv(self.documents_path)
        return df["text"].tolist()
