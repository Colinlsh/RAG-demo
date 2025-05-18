from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st
import networkx as nx
import re
import spacy
from torch import cosine_similarity, tensor
import torch
from langchain.docstore.document import Document

from utils.embeddings import OpenAIEmbedder

nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "tagger"])  # Disable unused components
nlp.add_pipe("sentencizer")

# def build_knowledge_graph(docs, embedding_model: OpenAIEmbedder):
#     G = nx.Graph()
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     id_to_doc = {}  # Map document IDs to their metadata
    
#     for doc in docs:
#         # Create a document node with all metadata
#         doc_id = f"doc_{doc.metadata['source_idx']}"
#         G.add_node(doc_id, 
#                   type="document",
#                   **doc.metadata)  # Includes source_file and source_idx
        
#         # Store document reference
#         id_to_doc[doc_id] = doc
        
#         # Process content and extract entities
#         doc_content = nlp(doc.page_content)
#         entities = [ent.text for ent in doc_content.ents]
        
#         # Add entity nodes and document-entity relationships
#         for entity in entities:
#             if entity not in G.nodes:
#                 G.add_node(entity, 
#                           type="entity",
#                           embedding=embedding_model.embed_documents([entity])[0])
            
#             # Add relationship between document and entity
#             G.add_edge(doc_id, entity, 
#                       weight=1.0,
#                       occurrences=entities.count(entity))
        
#         # Add entity-entity relationships within the same document
#         for i in range(len(entities)):
#             for j in range(i+1, len(entities)):
#                 ent1, ent2 = entities[i], entities[j]
#                 if G.has_edge(ent1, ent2):
#                     # Increase weight if relationship exists
#                     G[ent1][ent2]["weight"] += 0.1
#                 else:
#                     # Create new relationship with similarity score
#                     emb1 = torch.tensor(G.nodes[ent1]["embedding"]).to(device)
#                     emb2 = torch.tensor(G.nodes[ent2]["embedding"]).to(device)
#                     sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
#                     if sim > 0.6:  # Slightly lower threshold for doc-internal relationships
#                         G.add_edge(ent1, ent2, weight=sim)
    
#     return G, id_to_doc

def build_knowledge_graph(docs: List[Document], embedding_model: OpenAIEmbedder) -> Tuple[nx.Graph, Dict]:
    """Optimized KG construction with sentence-aware relationships."""
    G = nx.Graph()
    id_to_doc = {}
    
    # Pre-embed all unique entities across documents
    all_entities = set()
    for doc in docs:
        doc_content = nlp(doc.page_content)
        entities = [
            ent.text for ent in doc_content.ents 
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]
        ]
        all_entities.update(entities)
    
    # Bulk embed entities (reduces API calls)
    if all_entities:
        entity_embeddings = embedding_model.embed_documents(list(all_entities))
        entity_embed_map = dict(zip(all_entities, entity_embeddings))
    
    for doc in docs:
        # Document node
        doc_id = f"doc_{doc.metadata['source_idx']}"
        G.add_node(doc_id, type="document", **doc.metadata)
        id_to_doc[doc_id] = doc

        # Process entities with sentence boundaries
        doc_content = nlp(doc.page_content)
        entities_in_doc = [
            ent.text for ent in doc_content.ents 
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]
        ]
        
        # Add entities and document-entity edges
        for entity in entities_in_doc:
            if entity not in G.nodes:
                G.add_node(entity, type="entity", embedding=entity_embed_map[entity])
            G.add_edge(doc_id, entity, weight=1.0)
        
        # Sentence-level relationships
        for sent in doc_content.sents:  # Now works due to sentencizer
            sent_entities = [
                ent.text for ent in sent.ents 
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]
            ]
            
            # Compare entities within the same sentence
            for i in range(len(sent_entities)):
                for j in range(i+1, len(sent_entities)):
                    ent1, ent2 = sent_entities[i], sent_entities[j]
                    emb1 = torch.tensor(entity_embed_map[ent1])
                    emb2 = torch.tensor(entity_embed_map[ent2])
                    sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    
                    if sim > 0.6:
                        if G.has_edge(ent1, ent2):
                            G[ent1][ent2]["weight"] += sim
                        else:
                            G.add_edge(ent1, ent2, weight=sim)
    
    return G, id_to_doc

def retrieve_from_graph(queries: List[str], G: dict, doc_id: str, top_k: int = 5) -> List[dict]:
    """Process multiple query variants and aggregate results."""
    embedding_model = OpenAIEmbedder()
    all_entity_scores = defaultdict(float)
    
    # Process each query variant
    for query in queries:
        query_embedding = embedding_model.embed_query(query)
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Calculate entity scores for this query
        entity_scores = {}
        for node in G['graph'].nodes:
            if G['graph'].nodes[node].get("type") == "entity":
                entity_embedding = G['graph'].nodes[node]["embedding"]
                if not isinstance(entity_embedding, torch.Tensor):
                    entity_embedding = torch.tensor(entity_embedding, dtype=torch.float32)
                entity_embedding = entity_embedding.unsqueeze(0)
                similarity = torch.cosine_similarity(query_embedding, entity_embedding, dim=1).item()
                entity_scores[node] = similarity
        
        # Accumulate scores across all queries
        for entity, score in entity_scores.items():
            all_entity_scores[entity] += score
    
    # Get top entities across all query variants
    top_entities = sorted(all_entity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate document scores based on aggregated entity scores
    doc_scores = defaultdict(float)
    doc_entities = defaultdict(list)
    
    for entity, total_score in top_entities:
        for neighbor in G['graph'].neighbors(entity):
            if G['graph'].nodes[neighbor].get("type") == "document":
                doc_scores[neighbor] += total_score
                doc_entities[neighbor].append((entity, total_score))
    
    # Compile results
    results = []
    for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        source_file = G['metadata'].get("source_file", "Unknown")
        source_idx = G['metadata'].get("source_idx", -1)
        original_doc = G['id_map'].get(doc_id)
        
        if not original_doc:
            continue
            
        related_entities = sorted(doc_entities[doc_id], key=lambda x: x[1], reverse=True)
        
        results.append({
            "document_id": doc_id,
            "source_file": source_file,
            "source_idx": source_idx,
            "content": original_doc.page_content[:500] + "...",
            "related_entities": [e[0] for e in related_entities],
            "confidence": score / len(related_entities) if related_entities else 0,
            "metadata": G['metadata']
        })
    
    return results