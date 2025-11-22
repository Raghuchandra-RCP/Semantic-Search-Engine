import streamlit as st
from pathlib import Path
from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import VectorSearchEngine

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .preview-text {
        background-color: white;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None


@st.cache_resource
def initialize_search_engine():
    try:
        docs_folder = "data/docs"
        if not Path(docs_folder).exists():
            st.error(f"Documents folder not found: {docs_folder}")
            st.info("Please run: python download_dataset.py")
            return None
        
        preprocessor = DocumentPreprocessor(docs_folder)
        documents = preprocessor.load_documents()
        documents_dict = {doc["doc_id"]: doc for doc in documents}
        
        embedder = EmbeddingGenerator()
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
                embeddings = embedder.embed_batch(documents)
                search_engine.build_index(documents, embeddings)
        else:
            with st.spinner("Building index (this may take a few minutes on first run)..."):
                embeddings = embedder.embed_batch(documents)
                search_engine.build_index(documents, embeddings)
        
        return search_engine
    except Exception as e:
        st.error(f"Error initializing search engine: {e}")
        import traceback
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
