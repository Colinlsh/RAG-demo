# import faiss # Keep if you are using FAISS directly later, otherwise remove if only using LangChain's FAISS
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from utils.embedding_cache import cache
import streamlit as st


class OpenAIEmbedder(Embeddings):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIEmbedder, cls).__new__(cls)
            cls._instance.embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.session_state.openai_api_key
            )
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Check cache first
        cached_results = []
        uncached_texts = []
        
        for text in texts:
            cached_embedding = cache.get_embedding(text)
            if cached_embedding:
                cached_results.append(cached_embedding)
            else:
                uncached_texts.append(text)
        
        # Call OpenAI only for uncached texts
        if uncached_texts:
            new_embeddings = self.embedding_model.embed_documents(uncached_texts)
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache.add_embedding(text, embedding)
            cached_results.extend(new_embeddings)
        
        return cached_results

    def embed_query(self, text: str) -> List[float]:
        cached_embedding = cache.get_embedding(text)
        if cached_embedding:
            return cached_embedding
        embedding = self.embedding_model.embed_query(text)
        cache.add_embedding(text, embedding)
        return embedding