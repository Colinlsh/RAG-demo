import multiprocessing as mp
from typing import List

import numpy as np

from utils.metrics import evaluate_retrieval
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")  # Only set if not already set

import os
# Disable Streamlit's file-watcher for torch internals
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import torch
torch.set_num_threads(1)  # Reduce thread contention
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable HuggingFace tokenizer parallelism


import streamlit as st
from utils.retriever_pipeline import build_system_prompt, compare_documents, reformulate_query_with_llm, retrieve_even_contexts
from utils.doc_handler import get_ai_client, process_documents
from sentence_transformers import CrossEncoder
# from langchain_openai.chat_models import ChatOpenAI


import os
from dotenv import load_dotenv, find_dotenv

# Fix for torch classes not found error
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# At the top of app.py
from collections import defaultdict
from langchain_core.documents import Document
metrics = defaultdict(list)

COMPARISON_TRIGGERS = [
    'compare', 'versus', 'vs', 'difference', 'contrast',
    'across', 'between', 'relative'
]

def is_comparison_request(query: str) -> bool:
    return any(trigger in query.lower() for trigger in COMPARISON_TRIGGERS)

VISUALIZATION_TRIGGERS = [
    'plot', 'graph', 'chart', 'visualize', 'trend',
    'engagement', 'metrics', 'analytics', 'statistics'
]

def is_visualization_request(query: str) -> bool:
    return any(trigger in query.lower() for trigger in VISUALIZATION_TRIGGERS)

if "plot_data" not in st.session_state:
    st.session_state.plot_data = {}

# Add this function to help maintain state
def handle_post_selection():
    """Reset response placeholder when post selection changes"""
    st.session_state.pop('selected_post', None)
    st.session_state.pop('selected_post_id', None)
    st.session_state.pop('compare_post', None)

# After generating a response in app.py:
def log_metrics(query: str, retrieved_docs: List[Document], answer: str, context: str, reranker):
    # Mock relevant docs (replace with your ground truth)
    # relevant_docs = ["doc1.pdf", "doc2.pdf"]  
    relevant_docs = [file["name"] for file in st.session_state.uploaded_files_info]
    
    results = evaluate_retrieval(query, retrieved_docs, relevant_docs, answer, context, reranker)
    
    # Log retrieval metrics
    metrics["precision@5"].append(results["precision@5"])
    metrics["recall@5"].append(results["recall@5"])
    metrics["mrr"].append(results["mrr"])
    
    # Log generation metrics
    metrics["faithfulness"].append(results["faithfulness"])
    metrics["answer_relevance"].append(results["answer_relevance"])
    
    metrics["content_recall"].append(results["content_recall"])
    metrics['novelty'].append(results['novelty'])
    metrics['coverage'].append(results['coverage'])
            
    # Display in Streamlit
    with st.expander("📊 Performance Metrics"):
        st.metric("Precision@5", f"{np.mean(metrics['precision@5']):.2f}")
        st.metric("Recall@5", f"{np.mean(metrics['recall@5']):.2f}")
        st.metric("Faithfulness", f"{np.mean(metrics['faithfulness']):.2f}")
        st.metric("Answer Relevance", f"{np.mean(metrics['answer_relevance']):.2f}")
        st.metric("Content Recall", f"{np.mean(metrics['content_recall']):.2f}")
        st.metric("Novelty", f"{np.mean(metrics['novelty']):.2f}")
        st.metric("Coverage", f"{np.mean(metrics['coverage']):.2f}")
        st.metric("MRR", f"{np.mean(metrics['mrr']):.2f}")



# Load environment
load_dotenv(find_dotenv())  
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:32b")  # Ollama model for generation only
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


device = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize reranker globally
reranker = None
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")

# Streamlit page config and CSS
st.set_page_config(page_title="DeepGraph RAG-Pro", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
for key, default in {
    "messages": [],
    "retrieval_pipeline": None,
    "rag_enabled": False,
    "documents_loaded": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar: Document management and settings
with st.sidebar:
    st.header("OpenAI API Key")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI Api Key", placeholder="OPENAI API KEY"
    )
    st.markdown("---")
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT/JSON)", type=["pdf", "docx", "txt", "json"], accept_multiple_files=True
    )
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.uploaded_files_info = [
                {"name": f.name, "type": f.type} for f in uploaded_files
            ]
            process_documents(uploaded_files, reranker)  # Updated call
            st.session_state.documents_loaded = True
        st.success("Documents processed and indexed!")
        st.subheader("Processed Files")
        for info in st.session_state.uploaded_files_info:
            st.write(f"- {info['name']} ({info['type']})")

    st.markdown("---")
    st.header("⚙️ RAG Settings")
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    # st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=False)  # Disabled for ReasonIR
    # st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.use_llm_chunk_validation = st.checkbox("LLM Chunk validation", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 100, 40)

    st.markdown("---")
    if st.checkbox("Show Document Metadata (Debug)"):
        if "retrieval_pipeline" in st.session_state:
            st.write("Document Metadata Samples:")
            for i, doc in enumerate(st.session_state.retrieval_pipeline["texts"][:3]):  # Show first 3
                if hasattr(doc, 'metadata'):
                    st.json(doc.metadata)
                else:
                    st.write(f"Document {i+1}: No metadata")
    st.markdown("---")
    st.header("🎚️ Custom Instructions")
    st.session_state.response_style = st.selectbox(
        "Response Style", ["Professional","Concise","Detailed","Friendly","Academic"], index=0
    )
    st.session_state.user_role = st.text_input(
        "Your Role (for context)", placeholder="e.g., 'medical student', 'business executive'"
    )
    st.session_state.custom_instructions = st.text_area(
        "Additional Instructions", placeholder="Any specific requirements for responses..."
    )
    st.session_state.response_length = st.slider(
        "Response Length", 50, 1000, 800, step=50
    )

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")

    # Footer credit
    st.markdown(
        "<div style='position:absolute;bottom:10px;right:10px;font-size:12px;color:gray;'>"
        "<b>Developed by:</b> Colin 2025</div>", unsafe_allow_html=True
    )
    
# Main chat interface
def main():
    st.title("COLIN DEMO")
    st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, and Chat History")

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Ask about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        is_comparison = "compare" in prompt.lower() or "difference" in prompt.lower()
        
        if is_comparison:
            comparison_prompt = compare_documents(prompt)
            prompt = f"Compare documents focusing on: {comparison_prompt}"

        # Build context if RAG enabled
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            chat_history = "\n".join(m["content"] for m in st.session_state.messages[-5:])

            
            query_variants = reformulate_query_with_llm(prompt, chat_history)
            with st.expander("🔍 Debug: Query Variants", expanded=False):
                if hasattr(st.session_state, "debug_variants"):
                    st.write("Generated query variants:")
                    for i, variant in query_variants:
                        st.write(f"{i}. `{variant}`")
                        
            try:
                # docs = retrieve_documents(prompt, k=st.session_state.max_contexts)
                n_docs = len(st.session_state.uploaded_files_info)  # Number of uploaded documents
                per_doc = int(st.session_state.max_contexts / n_docs)  # Adjust this value for contexts per document
                docs = retrieve_even_contexts(
                    st.session_state.retrieval_pipeline["ensemble"],
                    query_variants,
                    n_docs=n_docs,
                    per_doc=per_doc,
                    fetch_multiplier=3
                )
                context_parts = []
                            
                for i, doc in enumerate(docs):
                    
                    source_name = doc.metadata.get('source_name', f"Document {i+1}")
                    source_file = doc.metadata.get('source_file', 'Unknown file')
                    content = doc.page_content
                    
                    context_parts.append(f"""
                            ### {source_name}
                            **File:** {source_file}  
                            **Content:**  
                            {content}
                            """)
                context = "\n".join(context_parts)
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                context = ""
            
            # Build system prompt
            chat_history = "\n".join(m["content"] for m in st.session_state.messages[-5:])
            system_prompt = build_system_prompt(prompt, chat_history, context)
                    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            if len(st.session_state.messages) > 1:
                for msg in st.session_state.messages[-5:-1]:  # Last 4 messages (excluding current)
                    messages.insert(1, {"role": msg["role"], "content": msg["content"]})

            # Stream response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                ai_client, model_name = get_ai_client(is_async=False)
                
                response = ai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=st.session_state.temperature,
                    stream=True
                )
                try:
                    # Stream the response
                    full_response = ""
                    for chunk in response:
                        # print("Raw chunk:", chunk)  # Debugging: see what the chunk actually looks like
                        
                        delta = chunk.choices[0].delta
                        
                        if delta.content == None:
                            continue
                        
                        full_response += delta.content
                        response_placeholder.markdown(full_response + "▌")
                        
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
        
                    if st.session_state.rag_enabled and context:
                        with st.expander("📚 Source Documents"):
                            st.markdown(context)
                            
                    log_metrics(prompt, docs, full_response, context, reranker)

                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
                    st.session_state.messages.append({"role":"assistant","content":"Sorry, I encountered an error."})

if __name__ == "__main__": main()
