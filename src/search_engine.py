import json
import numpy as np
import faiss
from typing import List, Dict, Optional
from pathlib import Path
from .embedder import EmbeddingGenerator


class VectorSearchEngine:
    
    def __init__(self, embedder: EmbeddingGenerator, index_file: str = "models/faiss_index.bin",
                 mapping_file: str = "models/doc_id_mapping.json"):
        self.embedder = embedder
        self.index_file = Path(index_file)
        self.mapping_file = Path(mapping_file)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.doc_ids = []
        self.doc_id_to_index = {}
        self.documents = {}
        self.dimension = 384
    
    def build_index(self, documents: List[Dict], embeddings: Dict[str, np.ndarray]):
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        self.documents = {doc["doc_id"]: doc for doc in documents}
        
        doc_ids_set = {doc["doc_id"] for doc in documents}
        embedding_doc_ids = set(embeddings.keys())
        
        missing_embeddings = doc_ids_set - embedding_doc_ids
        if missing_embeddings:
            print(f"⚠ Warning: {len(missing_embeddings)} documents missing embeddings:")
            for doc_id in list(missing_embeddings)[:10]:
                print(f"   - {doc_id}")
            if len(missing_embeddings) > 10:
                print(f"   ... and {len(missing_embeddings) - 10} more")
            raise ValueError(f"{len(missing_embeddings)} documents are missing embeddings. "
                           f"Please regenerate embeddings for all documents.")
        
        invalid_embeddings = []
        for doc_id, embedding in embeddings.items():
            if isinstance(embedding, np.ndarray):
                if embedding.size == 0:
                    invalid_embeddings.append((doc_id, "empty array"))
                elif len(embedding.shape) != 1:
                    invalid_embeddings.append((doc_id, f"wrong shape: {embedding.shape}, expected 1D"))
                elif len(embedding) != self.dimension:
                    invalid_embeddings.append((doc_id, f"wrong dimension: {len(embedding)} != {self.dimension}"))
            elif isinstance(embedding, list):
                if len(embedding) == 0:
                    invalid_embeddings.append((doc_id, "empty list"))
                elif len(embedding) != self.dimension:
                    invalid_embeddings.append((doc_id, f"wrong dimension: {len(embedding)} != {self.dimension}"))
            else:
                invalid_embeddings.append((doc_id, f"invalid type: {type(embedding)}"))
        
        if invalid_embeddings:
            print(f"⚠ Warning: {len(invalid_embeddings)} documents have invalid embeddings:")
            for doc_id, reason in invalid_embeddings[:10]:
                print(f"   - {doc_id}: {reason}")
            raise ValueError(f"{len(invalid_embeddings)} documents have invalid embeddings. "
                           f"Please regenerate embeddings.")
        
        doc_ids_list = []
        embedding_list = []
        
        for idx, doc in enumerate(documents):
            doc_id = doc["doc_id"]
            if doc_id in embeddings:
                embedding = embeddings[doc_id]
                if isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                else:
                    embedding = np.array(embedding, dtype=np.float32)
                
                doc_ids_list.append(doc_id)
                self.doc_id_to_index[doc_id] = len(doc_ids_list) - 1
                embedding_list.append(embedding)
        
        embeddings_array = np.array(embedding_list, dtype=np.float32)
        
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, "
                           f"got {embeddings_array.shape[1]}")
        
        faiss.normalize_L2(embeddings_array)
        
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        
        faiss_ids = np.arange(len(embeddings_array), dtype=np.int64)
        self.index.add_with_ids(embeddings_array, faiss_ids)
        
        self.doc_ids = doc_ids_list
        self.doc_id_to_faiss_id = {doc_id: idx for idx, doc_id in enumerate(doc_ids_list)}
        
        print(f"✓ Built FAISS index with {len(doc_ids_list)} documents")
        print(f"  All {len(documents)} documents have valid embeddings")
        
        self.save_index()
        self.save_mapping()
    
    def add_documents(self, documents: List[Dict], embeddings: Dict[str, np.ndarray]):
        if self.index is None:
            self.build_index(documents, embeddings)
            return
        
        new_doc_ids = []
        new_embeddings = []
        
        for doc in documents:
            doc_id = doc["doc_id"]
            if doc_id in embeddings and doc_id not in self.doc_ids:
                embedding = embeddings[doc_id]
                if isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                else:
                    embedding = np.array(embedding, dtype=np.float32)
                
                if len(embedding) != self.dimension:
                    print(f"⚠ Skipping {doc_id}: dimension mismatch")
                    continue
                
                new_doc_ids.append(doc_id)
                new_embeddings.append(embedding)
                self.documents[doc_id] = doc
        
        if not new_embeddings:
            print("No new documents to add")
            return
        
        new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
        faiss.normalize_L2(new_embeddings_array)
        
        start_id = len(self.doc_ids)
        new_faiss_ids = np.arange(start_id, start_id + len(new_embeddings), dtype=np.int64)
        
        self.index.add_with_ids(new_embeddings_array, new_faiss_ids)
        
        for i, doc_id in enumerate(new_doc_ids):
            faiss_id = start_id + i
            self.doc_ids.append(doc_id)
            self.doc_id_to_index[doc_id] = len(self.doc_ids) - 1
            self.doc_id_to_faiss_id[doc_id] = faiss_id
        
        print(f"✓ Added {len(new_doc_ids)} new documents to index")
        print(f"  Total documents in index: {len(self.doc_ids)}")
        
        self.save_index()
        self.save_mapping()
    
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            print(f"Saved FAISS index to {self.index_file}")
    
    def save_mapping(self):
        mapping_data = {
            "doc_ids": self.doc_ids,
            "doc_id_to_index": self.doc_id_to_index,
            "doc_id_to_faiss_id": getattr(self, 'doc_id_to_faiss_id', {}),
            "dimension": self.dimension,
            "total_documents": len(self.doc_ids)
        }
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        print(f"Saved doc_id mapping to {self.mapping_file}")
    
    def load_index(self, documents: Dict[str, Dict] = None):
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")
        
        self.index = faiss.read_index(str(self.index_file))
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.doc_ids = mapping_data.get("doc_ids", [])
            self.doc_id_to_index = mapping_data.get("doc_id_to_index", {})
            self.doc_id_to_faiss_id = mapping_data.get("doc_id_to_faiss_id", 
                                                         {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)})
            self.dimension = mapping_data.get("dimension", 384)
        
        if documents:
            self.documents = documents
        
        print(f"Loaded FAISS index from {self.index_file}")
        print(f"Loaded mapping for {len(self.doc_ids)} documents from {self.mapping_file}")
    
    def search(self, query: str, top_k: int = 5, expanded_query: Optional[str] = None) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        search_query = expanded_query if expanded_query else query
        
        query_emb = self.embedder.embed_query(search_query)
        query_emb = np.array(query_emb, dtype=np.float32)
        query_embedding = query_emb.reshape(1, -1).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        
        scores, faiss_ids = self.index.search(query_embedding, min(top_k, len(self.doc_ids)))
        
        results = []
        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            if faiss_id >= 0 and faiss_id < len(self.doc_ids):
                doc_id = self.doc_ids[faiss_id]
                doc = self.documents.get(doc_id, {})
                
                doc_text = doc.get("text", "")
                if len(doc_text) > 300:
                    preview = doc_text[:300].strip() + "..."
                elif len(doc_text) > 0:
                    preview = doc_text.strip()
                else:
                    preview = "Preview not available"
                
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "preview": preview,
                    "filename": doc.get("filename", ""),
                    "length": doc.get("length", 0)
                })
        
        return results
