# import faiss # Keep if you are using FAISS directly later, otherwise remove if only using LangChain's FAISS
from hashlib import md5
import json
from typing import Dict, List, Union
# from langchain.embeddings import OpenAIEmbeddings
import os

class EmbeddingCache:
    def __init__(self, cache_path: str = "embeddings_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def _get_key(self, text: str) -> str:
        return md5(text.encode()).hexdigest()  # Unique key for each text

    def get_embedding(self, text: str) -> Union[List[float], None]:
        key = self._get_key(text)
        return self.cache.get(key)

    def add_embedding(self, text: str, embedding: List[float]):
        key = self._get_key(text)
        self.cache[key] = embedding
        self._save_cache()

# Global cache instance
cache = EmbeddingCache()

