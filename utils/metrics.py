from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from utils.embeddings import OpenAIEmbedder

def calculate_precision_at_k(retrieved_docs: List[Document], relevant_docs: List[str], k: int) -> float:
    """
    relevant_docs: List of ground-truth document IDs or content snippets.
    """
    top_k = [doc.metadata.get("source_file") for doc in retrieved_docs[:k]]
    relevant = set(relevant_docs)
    return len([doc for doc in top_k if doc in relevant]) / k

def calculate_recall_at_k(retrieved_docs: List[Document], relevant_docs: List[str], k: int) -> float:
    top_k = [doc.page_content for doc in retrieved_docs[:k]]
    relevant = set(relevant_docs)
    return len([doc for doc in top_k if doc in relevant]) / len(relevant_docs)

def calculate_mrr(retrieved_docs: List[Document], relevant_docs: List[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc.page_content in relevant_docs:
            return 1.0 / rank
    return 0.0

def calculate_content_recall(retrieved_docs: List[Document], query: str, threshold=0.7):
    """Measure if retrieved chunks are semantically relevant to the query."""
    embedding_model = OpenAIEmbedder()
    
    query_embed = embedding_model.embed_query(query)
    doc_embeds = embedding_model.embed_documents([doc.page_content for doc in retrieved_docs])
    
    similarities = cosine_similarity([query_embed], doc_embeds)[0]
    return sum(sim > threshold for sim in similarities) / len(retrieved_docs)

def calculate_faithfulness(answer: str, context: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([answer, context])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def calculate_answer_relevance(query: str, answer: str, reranker: CrossEncoder) -> float:
    return reranker.predict([[query, answer]], activation_fn=torch.nn.Sigmoid())[0]

def evaluate_retrieval(query: str, retrieved_docs: List[Document], relevant_docs: List[str], answer: str, context: str, reranker):
    """Comprehensive retrieval evaluation"""
    results = {
        'precision@5': calculate_precision_at_k(retrieved_docs, relevant_docs, 5),
        'recall@5': calculate_recall_at_k(retrieved_docs, relevant_docs, 5),
        'mrr': calculate_mrr(retrieved_docs, relevant_docs),
        'faithfulness': calculate_faithfulness(answer, context),
        'answer_relevance': calculate_answer_relevance(query, answer, reranker),
        'content_recall': calculate_content_recall(retrieved_docs, query),
        'coverage': len(set(d.metadata['source_file'] for d in retrieved_docs)) / len(relevant_docs),
        'novelty': len(set(d.page_content for d in retrieved_docs) - set(relevant_docs)) / len(retrieved_docs)
    }
    return results
