from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import spacy
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re

from utils.embeddings import OpenAIEmbedder

FILES_ID = []
kg_collection = {}
nlp = spacy.load("en_core_web_sm")

def bm25_preprocess(text):
    return re.sub(r"[^\w\s-]", "", text).split()

def get_ai_client(is_async=False):
    """
    Returns a tuple of (client_instance, model_name). When is_ollama is True,
    returns an instance of AsyncOllamaClient.
    """
    openai_api_key = st.session_state.openai_api_key
    
    if openai_api_key is None or openai_api_key == "":
        openai_api_key == os.getenv("OPENAI_API_KEY")
        
    
    model="gpt-4.1-mini"
    
    return OpenAI(api_key = openai_api_key), model

def is_valid_chunk(chunk, ai_client: OpenAI = None):
    text = chunk.page_content.strip()
    
    # Basic length checks
    if len(text) < 50 or len(text) > 2000:  # Wider range
        return False
    
    # Count meaningful words (not just tokens)
    word_count = len([w for w in text.split() if len(w) > 2])
    if word_count < 5:
        return False
    
    # Optional: Use LLM to validate chunk quality
    # This is more expensive but more accurate
    if st.session_state.get('use_llm_chunk_validation', False):
        return validate_chunk_with_llm(text, ai_client)
    
    return True

def validate_chunk_with_llm(text, ai_client: OpenAI):
    """Use LLM to determine if chunk is meaningful"""
    prompt = f"""Determine if this text chunk is meaningful and self-contained:
    
    {text}
    
    Respond with just "YES" or "NO"."""
    
    response = ai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper() == "YES"

def merge_related_chunks(chunks):
    merged = []
    buffer = ""
    
    for chunk in chunks:
        current_text = chunk.page_content
        
        # Merge if either:
        # 1. Current chunk is too short (<100 chars), OR
        # 2. Buffer ends with colon/semicolon and current starts lowercase
        if (len(current_text) < 100 or 
            (buffer and buffer[-1] in [":", ";"] and current_text[0].islower())):
            buffer += " " + current_text
        else:
            if buffer:
                merged.append(Document(
                    page_content=buffer.strip(),
                    metadata=chunk.metadata
                ))
            buffer = current_text
    
    if buffer:
        merged.append(Document(
            page_content=buffer.strip(),
            metadata=chunk.metadata
        ))
    
    return merged

def process_documents(uploaded_files, reranker):
    # if st.session_state.documents_loaded:
    #     return
    
    files_to_process = [file for file in uploaded_files if file.file_id not in FILES_ID]
    
    if len(files_to_process) == 0:
        return
    
    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process files
    for idx, file in enumerate(uploaded_files):
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source_file'] = file.name
                doc.metadata['source_idx'] = idx
            documents.extend(loaded_docs)
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return
        
    st.session_state.document_sources = {
        idx: file.name for idx, file in enumerate(uploaded_files)
    }
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Increased from 512
        chunk_overlap=512,  # Increased from 128
        separators=[
            "\n\n\n",  # Triple newlines first (major section breaks)
            "\n\n",    # Then double newlines
            "。", "．",  # Asian full stops
            "\n",      # Single newlines
            " ",       # Spaces
            ""         # Final fallback
        ],
        keep_separator=True  # Preserve separation markers
    )
    
    split_docs = text_splitter.split_documents(documents)
    merged_docs = merge_related_chunks(split_docs)
    
    texts = []
            
    texts = merged_docs
    text_contents = [doc.page_content for doc in merged_docs]

    try:
        # Create embedding wrapper
        embedding_model = OpenAIEmbedder()
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            texts,
            embedding_model,
        )
        
        # Verify index creation
        if not hasattr(vector_store, "index"):
            raise ValueError("FAISS index not properly initialized")
    except Exception as e:
        st.error(f"ReasonIR embedding failed: {str(e)}")
        return
    

    # BM25 retriever setup
    bm25_retriever = BM25Retriever.from_texts(
        text_contents,
        metadatas=[doc.metadata for doc in texts],
        bm25_impl=BM25Okapi,
        preprocess_func=bm25_preprocess
    )

    # Hybrid ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 10})
        ],
        weights=[0.3, 0.7]
        # weights=[0.3, 0.7]
    )

    kg_collection = {}
    for doc in texts:
        doc_key = f"doc_{doc.metadata['source_idx']}"
        G, id_map = build_knowledge_graph([doc], embedding_model)
        kg_collection[doc_key] = {
            "graph": G,
            "id_map": id_map,
            "metadata": doc.metadata
        }
        
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,
        "texts": texts,
        "kg_collection": kg_collection  # Store individual document graphs
    }

    # st.session_state.documents_loaded = True
    FILES_ID.append(file.file_id)
    st.session_state.processing = False
    
    # display individual graph nodes.
    
    if "kg_collection" in st.session_state.retrieval_pipeline:
        for doc_key, doc_info in st.session_state.retrieval_pipeline["kg_collection"].items():
            st.write(f"File name: {doc_info['metadata']['source_file']}")
            st.write(f"Source index: {doc_info['metadata']['source_idx']}")
            st.write(f"Total Node: {len(doc_info['graph'].nodes)}")
            st.write(f"Total Edge: {len(doc_info['graph'].edges)}")

    # Debugging output
    # if "knowledge_graph" in st.session_state.retrieval_pipeline:
    #     G = st.session_state.retrieval_pipeline["knowledge_graph"]
    #     st.write(f"🔗 Total Nodes: {len(G.nodes)}")
    #     st.write(f"🔗 Total Edges: {len(G.edges)}")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'[^\x00-\x7F]+', '', text)