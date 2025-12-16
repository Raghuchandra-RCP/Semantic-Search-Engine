import sys
from pathlib import Path
from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import VectorSearchEngine
from src.api import create_app
import uvicorn

def setup_search_engine(docs_folder: str = "data/docs"):
    print("=" * 60)
    print("Multi-document Embedding Search Engine")
    print("=" * 60)
    
    print("\n[1/4] Preprocessing documents...")
    preprocessor = DocumentPreprocessor(docs_folder)
    documents = preprocessor.load_documents()
    
    print("\n[2/4] Initializing embedding generator...")
    embedder = EmbeddingGenerator()
    
    print("\n[3/4] Generating embeddings (using cache when available)...")
    embeddings = embedder.embed_batch(documents)
    
    doc_ids_set = {doc["doc_id"] for doc in documents if doc.get("text", "").strip()}
    embedding_doc_ids = set(embeddings.keys())
    missing = doc_ids_set - embedding_doc_ids
    
    if missing:
        print(f"\n⚠ Warning: {len(missing)} documents are missing embeddings!")
        print("This should not happen. Please check the logs above for errors.")
    else:
        print(f"\n✓ All {len(doc_ids_set)} documents have embeddings")
    
    print("\n[4/4] Building vector search index...")
    search_engine = VectorSearchEngine(embedder)
    try:
        search_engine.build_index(documents, embeddings)
    except ValueError as e:
        print(f"\n❌ Error building index: {e}")
        print("\nPlease ensure all documents have valid embeddings.")
        raise
    
    print("\n" + "=" * 60)
    print("Search engine ready!")
    print("=" * 60)
    print(f"Total documents indexed: {len(documents)}")
    cache_stats = embedder.cache_manager.stats()
    print(f"Cache statistics:")
    print(f"  - Total entries: {cache_stats['total_entries']}")
    print(f"  - Valid embeddings: {cache_stats['valid_embeddings']}")
    print(f"  - Cache DB: {cache_stats['cache_db']}")
    print(f"  - Cache size: {cache_stats['total_size_mb']:.2f} MB")
    
    return search_engine

def main():
    docs_folder = "data/docs"
    if not Path(docs_folder).exists():
        print(f"ERROR: Documents folder not found: {docs_folder}")
        print("\nPlease ensure you have:")
        print("1. Downloaded the dataset")
        print("2. Placed .txt files in data/docs/")
        print("\nTo download 20 Newsgroups dataset, run:")
        print("  python download_dataset.py")
        sys.exit(1)
    
    search_engine = setup_search_engine(docs_folder)
    
    app = create_app(search_engine)
    
    print("\n" + "=" * 60)
    print("Starting API server...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
