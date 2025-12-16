import streamlit as st
from pathlib import Path
import sys
import traceback

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from src.preprocessor import DocumentPreprocessor
    from src.embedder import EmbeddingGenerator
    from src.search_engine import VectorSearchEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def download_dataset():
    try:
        from download_dataset import download_and_save_dataset
        
        docs_folder = Path("data/docs")
        if docs_folder.exists() and list(docs_folder.glob("*.txt")):
            return True
        

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current: int, total: int):
            progress_bar.progress(current / total)
            status_text.text(f"Downloaded {current}/{total} documents...")
        
        with st.spinner("Downloading dataset (this may take a minute)..."):

            result = download_and_save_dataset(progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if result:
                st.success("✓ Downloaded dataset successfully!")
                return True
            return False
            
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return False

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

@st.cache_resource
def initialize_search_engine():
    try:
        docs_folder = "data/docs"
        if not Path(docs_folder).exists():
            return None
        
        txt_files = list(Path(docs_folder).glob("*.txt"))
        if not txt_files:
            return None
        
        try:
            preprocessor = DocumentPreprocessor(docs_folder)
            documents = preprocessor.load_documents()
            
            if not documents:
                return None
                
            documents_dict = {doc["doc_id"]: doc for doc in documents}
            

            try:
                embedder = EmbeddingGenerator()
            except Exception as e:
                st.error(f"Failed to load embedding model: {e}")
                return None
            
            search_engine = VectorSearchEngine(embedder)
            
            index_file = Path("models/faiss_index.bin")
            mapping_file = Path("models/doc_id_mapping.json")
            
            if index_file.exists() and mapping_file.exists():
                try:
                    search_engine.load_index(documents_dict)
                    return search_engine
                except Exception as e:
                    st.warning(f"Could not load existing index: {e}")
                    st.info("Rebuilding index...")
                    try:
                        embeddings = embedder.embed_batch(documents)
                        search_engine.build_index(documents, embeddings)
                        return search_engine
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")
                        return None
            else:
                with st.spinner("Building index (this may take a few minutes on first run)..."):
                    try:
                        embeddings = embedder.embed_batch(documents)
                        search_engine.build_index(documents, embeddings)
                        return search_engine
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")
                        return None
        
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return None
            
    except Exception as e:
        st.error(f"Error initializing search engine: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

def main():
    st.markdown('<h1 class="main-header">🔍 Semantic Search Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered document search using semantic embeddings</p>', unsafe_allow_html=True)
    
    if st.session_state.search_engine is None:
        with st.spinner("Initializing search engine..."):
            st.session_state.search_engine = initialize_search_engine()
    
    if st.session_state.search_engine is None:
        st.warning("⚠️ Search engine not initialized")
        
        docs_folder = Path("data/docs")
        txt_files = list(docs_folder.glob("*.txt")) if docs_folder.exists() else []
        
        if not txt_files:
            st.info("""
            **No documents found. The app can automatically download a sample dataset for you.**
            
            Click the button below to download the 20 Newsgroups dataset (~11,000 documents).
            This will take a few minutes on first run.
            """)
            
            if st.button("📥 Download Sample Dataset", type="primary"):
                if download_dataset():
                    # Clear the cache so search engine reinitializes
                    initialize_search_engine.clear()
                    st.session_state.search_engine = None
                    st.rerun()
        else:
            st.info("""
            **Documents found but search engine failed to initialize.**
            This might be due to:
            - Model loading issues
            - Index building errors
            - Missing dependencies
            """)
        
        if st.button("🔄 Retry Initialization"):
            st.session_state.search_engine = None
            st.rerun()
        
        st.stop()
    
    search_engine = st.session_state.search_engine
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many search results to display"
        )
        
        st.markdown("---")
        st.subheader("📊 System Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(search_engine.doc_ids))
        with col2:
            cache_stats = search_engine.embedder.cache_manager.stats()
            st.metric("Cached", cache_stats["total_entries"])
        
        st.markdown("---")
        st.subheader("🔄 Rebuild Index")
        st.caption("Clear cache and rebuild with updated chunking")
        
        if st.button("🗑️ Clear Cache & Rebuild", type="secondary", use_container_width=True):
            with st.spinner("Clearing cache and rebuilding index..."):
                try:

                    initialize_search_engine.clear()
                    

                    import os
                    cache_files = [
                        "cache/embeddings_cache.db",
                        "cache/embeddings_cache.db-shm",
                        "cache/embeddings_cache.db-wal",
                        "cache/embeddings_cache.json"
                    ]
                    for cache_file in cache_files:
                        cache_path = Path(cache_file)
                        if cache_path.exists():
                            os.remove(cache_path)
                    

                    index_files = [
                        "models/faiss_index.bin",
                        "models/doc_id_mapping.json"
                    ]
                    for index_file in index_files:
                        index_path = Path(index_file)
                        if index_path.exists():
                            os.remove(index_path)
                    

                    st.session_state.search_engine = None
                    
                    st.success("✅ Cache cleared! Reinitializing search engine...")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        st.markdown("---")
        st.caption("Built with FAISS & Sentence Transformers")
    
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="Enter your search query here...",
                label_visibility="collapsed"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button or (query and query.strip()):
        if query.strip():
            with st.spinner("Searching..."):
                results = search_engine.search(query, top_k=top_k)
            
            if results:
                st.success(f"Found {len(results)} result{'s' if len(results) > 1 else ''}")
                st.markdown("---")
                
                for i, result in enumerate(results, 1):
                    with st.container():
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"### {result['filename']}")
                            st.markdown(f"**ID:** `{result['doc_id']}`")
                        
                        with col2:
                            score_percent = result['score'] * 100
                            st.markdown(f'<div class="score-badge">Score: {score_percent:.1f}%</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Preview:**")
                        preview_text = result['preview'] if len(result['preview']) > 20 else "Preview not available"
                        st.markdown(f'<div class="preview-text">{preview_text}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")
            else:
                st.warning("No results found. Try a different query.")
        else:
            st.info("Please enter a search query.")
    
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #888; padding: 1rem;">'
        'Semantic Search Engine | Powered by FAISS & Sentence Transformers'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
