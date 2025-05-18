# retriever_pipeline.py

from collections import defaultdict
import re
import time
from typing import Any, List
import streamlit as st
from utils.build_graph import retrieve_from_graph
import numpy as np
# import faiss # Keep if you are using FAISS directly later, otherwise remove if only using LangChain's FAISS
import torch
from langchain.docstore.document import Document

from utils.doc_handler import get_ai_client
from utils.embeddings import OpenAIEmbedder


def reformulate_query_with_llm(query: str, chat_history: str = "") -> List[str]:
    """Generate high-quality query variants using LLM"""
    prompt = f"""You are an expert search query reformulator. Given the following query and chat history, 
generate 3-5 alternative queries that would help retrieve the most relevant documents.

Guidelines:
1. Maintain the core intent while exploring different aspects
2. Include synonyms and related concepts
3. For comparative queries, generate both "compare X and Y" and "difference between X and Y" variants
4. For technical queries, include both specific and general versions

Chat History:
{chat_history}

Original Query: {query}

Generate variants as a numbered list. Return only the list, nothing else."""

    try:
        ai_client, model_name = get_ai_client()
        response = ai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,  # Balance creativity and relevance
            max_tokens=500
        )
        
        # Parse response into clean variants
        variants = []
        for line in response.choices[0].message.content.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                variant = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if variant and variant not in variants:
                    variants.append(variant)
        return variants[:5]  # Return up to 5 best variants
    
    except Exception as e:
        st.error(f"Query reformulation failed: {str(e)}")
        return [query]  # Fallback to original query

def reformulate_query(query: str, embedding_model: OpenAIEmbedder, seed_queries: list[str] = None) -> list[str]:
    """Generate query variants by finding semantically similar phrases from a seed set."""
    # Step 1: Embed the input query
    query_embedding = embedding_model.embed_query(query)
    
    # Step 2: Use a predefined set of seed queries (or dynamically fetch from your corpus)
    if seed_queries is None:
        seed_queries = [
            "Explain the concept of {query}",
            "What are the applications of {query}?",
            "Compare {query} with related terms",
            "Technical details about {query}",
            "History of {query}"
        ]
    
    # Step 3: Find top 3 most similar seed templates
    seed_embeddings = embedding_model.embed_documents([s.format(query="") for s in seed_queries])
    similarities = [
        torch.cosine_similarity(
            torch.tensor(query_embedding).flatten(),
            torch.tensor(seed_embedding).flatten(),
            dim=0
        ).item()
        for seed_embedding in seed_embeddings
    ]
    
    # Step 4: Format the top templates with the original query
    top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 matches
    return [seed_queries[i].format(query=query) for i in top_indices]

def rerank_with_reasonir(query: str, docs: list[str], embedding_model: OpenAIEmbedder) -> list[str]:
    """Rerank documents using ReasonIR's relevance scoring."""
    # Convert embeddings to PyTorch tensors
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Convert embeddings to PyTorch tensors on the correct device
    query_embedding = torch.tensor(embedding_model.embed_query(query)).to(device)
    doc_embeddings = torch.tensor(embedding_model.embed_documents(docs)).to(device)
    
    # Calculate cosine similarities
    scores = [
        torch.cosine_similarity(
            query_embedding, 
            doc_embed.unsqueeze(0),  # Add batch dimension
            dim=1
        ).item()
        for doc_embed in doc_embeddings
    ]
    
    # Sort docs by descending similarity score
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

def expand_query(query, top_k=3):
    """Find related terms using embedding similarity"""
    embedding_model = OpenAIEmbedder()
    
    # Convert to tensors
    query_embed = torch.tensor(embedding_model.embed_query(query)).unsqueeze(0)
    all_terms = list(st.session_state.retrieval_pipeline["knowledge_graph"].nodes())
    term_embeds = torch.tensor(embedding_model.embed_documents(all_terms))
    
    # Calculate and get top terms
    similarities = torch.cosine_similarity(query_embed, term_embeds)
    top_indices = torch.topk(similarities, k=top_k).indices.tolist()
    
    return query + " " + " ".join([all_terms[i] for i in top_indices])

def retrieve_even_contexts(
    ensemble_retriever: Any,
    query_variants: str,
    n_docs: int,
    per_doc: int,
    fetch_multiplier: int = 3
) -> List[Document]:
    """
    Retrieve exactly n_docs * per_doc chunks, evenly distributed across up to n_docs unique sources.
    Uses LLM-based query reformulation for better retrieval.

    Args:
        ensemble_retriever: your LangChain ensemble retriever
        query: the user query string
        n_docs: how many distinct documents to draw from
        per_doc: how many chunks per document
        fetch_multiplier: how many times larger to oversample for post-filtering

    Returns:
        A list of Document chunks of length ≤ n_docs * per_doc
    """
    # Step 1: Generate query variants using LLM
    
    start = time.time()
    
    total_required = n_docs * per_doc
    fetch_k = total_required * fetch_multiplier
    candidates: List[Document] = []
    
    # Step 2: Retrieve documents for each variant
    for variant in query_variants:
        try:
            docs = ensemble_retriever.invoke(variant)[:fetch_k]
            candidates.extend(docs)
            
            # Early exit if we have enough candidates
            if len(candidates) >= fetch_k * len(query_variants):
                break
        except Exception as e:
            st.error(f"Error retrieving documents for variant '{variant}': {str(e)}")
            continue

    # Step 3: Select chunks evenly across documents
    selected: List[Document] = []
    counts = defaultdict(int)
    docs_seen = set()  # Using set for faster lookups
    
    # Sort candidates by relevance score if available
    if candidates and hasattr(candidates[0], 'metadata') and 'score' in candidates[0].metadata:
        candidates.sort(key=lambda x: x.metadata['score'], reverse=True)
        
    # Add graph results
    if st.session_state.get("enable_graph_rag", False):
        
        knowledge_graphs = st.session_state.retrieval_pipeline.get("kg_collection", {})
        graph_results = {}
        
        for graph_name, graph in knowledge_graphs.items():
            graph_results[graph_name] = retrieve_from_graph(
                query_variants,
                graph,
                graph_name
            )
        
        # Convert graph results to Document format
        graph_docs = []
        for res in graph_results.values():
            for _d in res:
                # Deduplicate by document ID
                if not any(d.metadata['document_id'] == _d['document_id'] for d in graph_docs):
                    _d["metadata"]['document_id'] = _d["document_id"]
                    graph_docs.append(Document(
                        page_content=_d["content"],
                        metadata=_d["metadata"]
                    ))
        
        candidates.extend(graph_docs)
    
    for chunk in candidates:
        # Get source identifier (prioritize different metadata fields)
        src_id = (
            chunk.metadata.get("source_id") 
            or chunk.metadata.get("source_file") 
            or chunk.metadata.get("source_name")
            or str(hash(chunk.metadata.get("source", ""))))
        
        if not src_id:
            continue

        # Add new document if we haven't reached n_docs limit
        if src_id not in docs_seen:
            if len(docs_seen) >= n_docs:
                continue
            docs_seen.add(src_id)

        # Add chunk if we haven't reached per_doc limit for this document
        if counts[src_id] < per_doc:
            selected.append(chunk)
            counts[src_id] += 1

        # Early exit if we've collected enough chunks
        if len(selected) >= total_required:
            break

    # Fallback if we didn't get enough chunks
    if len(selected) < total_required and len(selected) < len(candidates):
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:total_required - len(selected)])
        
    st.write(f"Retrieval took {time.time() - start:.2f}s")

    return selected[:total_required]  # Ensure we don't return more than requested


def retrieve_documents(query: str, k: int = 5, embedding_model: OpenAIEmbedder = None) -> List[Document]:
    """Hybrid retrieval with query reformulation and reranking."""
    if "retrieval_pipeline" not in st.session_state:
        return []

    ensemble_retriever = st.session_state.retrieval_pipeline["ensemble"]
    
    # Step 1: Query Reformulation
    query_variants = reformulate_query(query, embedding_model=embedding_model)
    st.session_state.debug_variants = query_variants  # For UI display
    
    # Step 2: Retrieve for each variant
    all_docs = []
    for variant in query_variants:
        docs = ensemble_retriever.invoke(variant)
        all_docs.extend(docs)
    
    # Step 3: Rerank (if enabled)
    if st.session_state.get("enable_reranking", False):
        doc_texts = [doc.page_content for doc in all_docs]
        reranked_texts = rerank_with_reasonir(query, doc_texts)
        # Map back to original Document objects
        all_docs = [doc for doc in all_docs if doc.page_content in reranked_texts]
    
    # Step 4: Apply GraphRAG if enabled
    if st.session_state.get("enable_graph_rag", False):
        graph_nodes = retrieve_from_graph(query, st.session_state.retrieval_pipeline["knowledge_graph"])
        graph_docs = [Document(page_content=node) for node in graph_nodes]
        all_docs.extend(graph_docs)
        
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
    for i, doc in enumerate(unique_docs[:k]):
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata.update({
            'source_id': i+1,
            'source_name': f"Document {i+1}",
            # Keep original source_file if it exists
            'source_file': doc.metadata.get('source_file', 'Unknown')
        })
    
    # Deduplicate and return top-k
    return unique_docs[:k]



# Your build_system_prompt function (remains unchanged)
def build_system_prompt(prompt, chat_history, context):
    
    if "compare" in prompt.lower() or "difference" in prompt.lower():
        comparison_text = compare_documents(prompt)
        return f"""You are an AI assistant helping compare documents. 
        
        {comparison_text}

        INSTRUCTIONS:
        1. Highlight key similarities and differences
        2. Point out unique aspects of each document
        3. Note any contradictions
        4. Organize your response by themes/topics
        """
    else:
        instruction_template = f"""You are an AI assistant with access to the following context:

    CONTEXT SOURCES:
    {context}

    INSTRUCTIONS:
    1. Always ground your answers in the provided context
    2. When referencing information, cite the source using its heading (e.g., "As mentioned in Document 1...")
    3. If multiple sources agree, mention this (e.g., "Several documents confirm that...")
    4. If sources conflict, note the discrepancy

    USER PREFERENCES:
    - Response Style: {st.session_state.get('response_style', 'Professional')}
    - User Role: {st.session_state.get('user_role', 'Not specified')}
    - Additional Instructions: {st.session_state.get('custom_instructions', 'None')}

    CHAT HISTORY:
    {chat_history}

    QUESTION: {prompt}
    ANSWER:"""
        return instruction_template

def compare_documents(query: str = None) -> str:
    """Compare all uploaded documents and highlight key differences/similarities"""
    if "retrieval_pipeline" not in st.session_state:
        return "No documents loaded for comparison"
    
    # Get all document texts grouped by source
    source_groups = {}
    for doc in st.session_state.retrieval_pipeline["texts"]:
        if hasattr(doc, 'metadata') and 'source_idx' in doc.metadata:
            source_idx = doc.metadata['source_idx']
            if source_idx not in source_groups:
                source_groups[source_idx] = []
            source_groups[source_idx].append(doc.page_content)
    
    if not source_groups:
        return "No documents available for comparison"
    
    # Prepare comparison prompt
    comparison_prompt = f"""Compare the following documents and identify:
1. Key similarities
2. Key differences
3. Unique aspects of each document
4. Any contradictions between documents
"""
    
    # Add document contents
    for source_idx, texts in source_groups.items():
        doc_name = st.session_state.document_sources.get(source_idx, f"Document {source_idx+1}")
        comparison_prompt += f"\n\n=== {doc_name} ===\n" + "\n".join(texts[:3])  # First few chunks
    
    if query:
        comparison_prompt += f"\n\nFocus especially on aspects related to: {query}"
    
    return comparison_prompt