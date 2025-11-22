# Multi-document Embedding Search Engine with Caching

A lightweight embedding-based search engine that efficiently searches over 100-200 text documents using semantic embeddings, with intelligent caching to avoid recomputing embeddings.

## Features

- ✅ **Efficient Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality embeddings
- ✅ **JSON-based Caching**: Smart caching system that only regenerates embeddings when document content changes
- ✅ **FAISS Vector Search**: Fast similarity search using FAISS IndexFlatIP (inner product for cosine similarity)
- ✅ **FastAPI REST API**: Clean, documented API with `/search` endpoint
- ✅ **Ranking Explanations**: Detailed explanations for why each document was matched
- ✅ **Modular Architecture**: Clean code structure with separate modules for each component

## Project Structure

```
/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Document loading & text cleaning
│   ├── embedder.py          # Embedding generation with caching
│   ├── cache_manager.py     # JSON-based cache management
│   ├── search_engine.py     # FAISS vector search
│   ├── api.py               # FastAPI endpoints
│   └── utils.py             # Helper functions
├── data/
│   └── docs/                # Text documents (.txt files)
├── cache/
│   └── embeddings_cache.json  # JSON cache for embeddings
├── models/
│   └── faiss_index.bin      # Persistent FAISS index
├── requirements.txt
├── main.py                  # Entry point to run API
├── download_dataset.py      # Script to download 20 Newsgroups dataset
└── README.md
```

## Installation

1. **Clone the repository** (or ensure you have the project files)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
```bash
python download_dataset.py
```

This will download the 20 Newsgroups dataset (~11,314 documents) and save them as `.txt` files in `data/docs/`.

## How Caching Works

The caching system uses **JSON** to store embeddings with the following structure:

```json
{
  "doc_001": {
    "embedding": [0.123, -0.456, ...],
    "hash": "sha256_hash_of_document_content",
    "updated_at": "2024-01-01T12:00:00",
    "filename": "doc_001.txt"
  }
}
```

### Cache Logic:

1. **Hash-based Invalidation**: Each document's content is hashed using SHA256
2. **Cache Lookup**: Before generating an embedding, the system checks:
   - If `doc_id` exists in cache
   - If the cached hash matches the current document hash
3. **Cache Hit**: If hash matches → reuse cached embedding (no recomputation)
4. **Cache Miss**: If hash differs or missing → generate new embedding and update cache

### Benefits:

- **No Recomputing**: Documents that haven't changed reuse cached embeddings
- **Automatic Updates**: Changed documents automatically get new embeddings
- **Human-readable**: JSON format allows easy inspection and debugging
- **Persistent**: Cache survives restarts

## How to Run Embedding Generation

Embedding generation happens automatically when you start the API. The system will:

1. Load all `.txt` files from `data/docs/`
2. Check cache for each document
3. Generate embeddings only for new/changed documents
4. Build the FAISS search index

**Manual embedding generation** (if needed):
```python
from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator

# Load documents
preprocessor = DocumentPreprocessor("data/docs")
documents = preprocessor.load_documents()

# Generate embeddings (with caching)
embedder = EmbeddingGenerator()
embeddings = embedder.embed_batch(documents)
```

## How to Start API

1. **Ensure documents are in `data/docs/`**:
   - Run `python download_dataset.py` if you haven't already
   - Or place your own `.txt` files in `data/docs/`

2. **Start the API server**:
```bash
python main.py
```

3. **API will be available at**:
   - Main API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Statistics: http://localhost:8000/stats

## API Usage

### Search Endpoint

**POST** `/search`

**Request Body**:
```json
{
  "query": "quantum physics basics",
  "top_k": 5
}
```

**Response**:
```json
{
  "results": [
    {
      "doc_id": "doc_014",
      "score": 0.88,
      "preview": "Quantum theory is concerned with...",
      "filename": "doc_014_sci.space.txt",
      "explanation": {
        "matched_keywords": ["quantum", "physics"],
        "overlap_ratio": 0.667,
        "document_length": 1234,
        "length_normalization_score": 0.912,
        "similarity_score": 0.8800,
        "explanation_text": "Similarity score: 0.8800. Matched keywords: quantum, physics. Query overlap: 66.7%. Document length: 1234 characters."
      }
    }
  ],
  "query": "quantum physics basics",
  "total_results": 5
}
```

### Example with cURL

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration", "top_k": 3}'
```

### Example with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "machine learning", "top_k": 5}
)

results = response.json()
for result in results["results"]:
    print(f"{result['doc_id']}: {result['score']:.4f}")
    print(f"  {result['explanation']['explanation_text']}\n")
```

## Ranking Explanation

Each search result includes a detailed explanation with:

1. **Similarity Score**: Cosine similarity between query and document embeddings
2. **Matched Keywords**: Words from the query found in the document
3. **Overlap Ratio**: Percentage of query words found in document
4. **Document Length**: Character count of the document
5. **Length Normalization Score**: Normalized score accounting for document length
6. **Explanation Text**: Human-readable summary of why the document matched

## Design Choices

### 1. **JSON Caching** (vs SQLite/Pickle)
- **Why**: Simpler, no database dependency, human-readable, easy to debug
- **Trade-off**: Slightly slower for very large caches, but sufficient for 100-200 documents

### 2. **FAISS IndexFlatIP** (vs Custom Cosine Similarity)
- **Why**: Fast, optimized C++ implementation, supports normalized cosine similarity
- **Implementation**: Normalize embeddings → use inner product = cosine similarity

### 3. **sentence-transformers/all-MiniLM-L6-v2**
- **Why**: Fast, lightweight (384 dimensions), good quality, no API costs
- **Alternative**: Could use OpenAI embeddings, but requires API key and costs

### 4. **Modular Architecture**
- **Why**: Each component in separate file for maintainability and testing
- **Structure**: preprocessor → embedder → search_engine → api

### 5. **Hash-based Cache Invalidation**
- **Why**: Only regenerate when content changes, not on every run
- **Implementation**: SHA256 hash of cleaned document text

## Performance

- **First Run**: Generates embeddings for all documents (~2-5 minutes for 100-200 docs)
- **Subsequent Runs**: Uses cache, builds index in seconds
- **Search Speed**: <100ms per query (FAISS is very fast)
- **Cache Size**: ~150KB per 100 documents (384-dim embeddings)

## Troubleshooting

### No documents found
- Ensure `data/docs/` contains `.txt` files
- Run `python download_dataset.py` to download sample dataset

### Cache not working
- Check `cache/embeddings_cache.json` exists and is readable
- Verify file permissions

### FAISS index errors
- Delete `models/faiss_index.bin` and rebuild
- Ensure embeddings are generated before building index

### Import errors
- Ensure you're in the project root directory
- Install all dependencies: `pip install -r requirements.txt`

## Future Enhancements (Bonus Features)

- [ ] Streamlit UI for interactive searching
- [ ] Query expansion using WordNet or embedding similarity
- [ ] Batch embedding with multiprocessing
- [ ] Evaluation metrics (precision@k, recall@k)
- [ ] Support for other embedding models
- [ ] Docker containerization

## License

This project is created for the AI Engineer Intern Assignment at CodeAtRandom AI.

## Author

Built as part of the AI Engineer Intern Assignment.


# Semantic-Search-Engine
