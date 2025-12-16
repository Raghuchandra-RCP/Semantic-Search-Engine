import sqlite3
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import threading

class CacheManager:
    
    def __init__(self, cache_db: str = "cache/embeddings_cache.db"):
        self.cache_db = Path(cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._dimension = 384
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(str(self.cache_db), timeout=30.0) as conn:
            cursor = conn.cursor()
            self._validate_and_clean_cache(conn)
            conn.commit()
    
    def _validate_and_clean_cache(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, embedding, dimension FROM embeddings")
        rows = cursor.fetchall()
        
        if not rows:
            return
        
        invalid_count = 0
        for doc_id, embedding_blob, stored_dim in rows:
            try:
                arr = np.frombuffer(embedding_blob, dtype=np.float32)
                if arr.size > 0:
                    if self._dimension is None or self._dimension == 384:
                        self._dimension = arr.size
                    break
            except:
                pass
        
        for doc_id, embedding_blob, stored_dim in rows:
            try:
                arr = np.frombuffer(embedding_blob, dtype=np.float32)
                if arr.size != stored_dim:
                    cursor.execute("UPDATE embeddings SET dimension=? WHERE doc_id=?", 
                                 (arr.size, doc_id))
                
                if self._dimension and arr.size != self._dimension:
                    print(f"  ⚠ Invalid cache entry for {doc_id}: dimension mismatch "
                          f"({arr.size} != {self._dimension}). Removing...")
                    cursor.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
                    invalid_count += 1
                elif arr.size == 0:
                    print(f"  ⚠ Invalid cache entry for {doc_id}: empty embedding. Removing...")
                    cursor.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
                    invalid_count += 1
            except Exception as e:
                print(f"  ⚠ Invalid cache entry for {doc_id}: {e}. Removing...")
                cursor.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"  ✓ Cleaned {invalid_count} invalid cache entries")
        conn.commit()
    
    def _get_connection(self, retries: int = 3):
        for attempt in range(retries):
            try:
                conn = sqlite3.connect(str(self.cache_db), timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                return conn
            except Exception as e:
                if "database is locked" in str(e).lower() and attempt < retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise
    
    def get(self, doc_id: str, current_hash: str) -> Optional[np.ndarray]:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    result = cursor.fetchone()
                    if result:
                        embedding_blob, dimension = result
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        
                        if embedding.size != dimension:
                            print(f"  ⚠ Cache entry for {doc_id} has dimension mismatch. Regenerating...")
                            return None
                        
                        if self._dimension and embedding.size != self._dimension:
                            print(f"  ⚠ Cache entry for {doc_id} has wrong dimension. Regenerating...")
                            return None
                        
                        self._dimension = dimension
                        return np.array(embedding, dtype=np.float32)
                    return None
            except Exception as e:
                print(f"⚠ Error reading from cache: {e}")
                return None
    
    def set(self, doc_id: str, embedding: List[float] | np.ndarray, doc_hash: str, 
            filename: str = "", save_immediately: bool = False, 
            model_name: str = None, model_version: str = None):
        if isinstance(embedding, list):
            embedding_array = np.array(embedding, dtype=np.float32)
        else:
            embedding_array = np.array(embedding, dtype=np.float32)
        
        if len(embedding_array.shape) != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding_array.shape}")
        
        dimension = len(embedding_array)
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dimension}, got {dimension}")
        
        embedding_blob = embedding_array.tobytes()
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                        doc_id,
                        embedding_blob,
                        doc_hash,
                        filename,
                        datetime.now().isoformat(),
                        dimension,
                        model_name or "sentence-transformers/all-MiniLM-L6-v2",
                        model_version or "default"
                    ))
                    if save_immediately:
                        conn.commit()
            except Exception as e:
                print(f"⚠ Error writing to cache: {e}")
                raise
    
    def save(self):
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.commit()
            except Exception as e:
                print(f"⚠ Error committing cache: {e}")
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        result = {}
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT doc_id, embedding FROM embeddings")
                    
                    for doc_id, embedding_blob in cursor.fetchall():
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        result[doc_id] = np.array(embedding, dtype=np.float32)
            except Exception as e:
                print(f"⚠ Error reading all embeddings: {e}")
        return result
    
    def clear(self):
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM embeddings")
                    conn.commit()
            except Exception as e:
                print(f"⚠ Error clearing cache: {e}")
    
    def stats(self) -> Dict:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM embeddings")
                    count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT SUM(LENGTH(embedding)) FROM embeddings")
                    total_size = cursor.fetchone()[0] or 0
                    
                    db_size = os.path.getsize(self.cache_db) if self.cache_db.exists() else 0
                    
                    return {
                        "total_entries": count,
                        "valid_embeddings": count,
                        "cache_db": str(self.cache_db),
                        "embeddings_size_mb": total_size / (1024 * 1024),
                        "db_size_mb": db_size / (1024 * 1024),
                        "total_size_mb": db_size / (1024 * 1024)
                    }
            except Exception as e:
                print(f"⚠ Error getting cache stats: {e}")
                return {
                    "total_entries": 0,
                    "valid_embeddings": 0,
                    "cache_db": str(self.cache_db),
                    "embeddings_size_mb": 0,
                    "db_size_mb": 0,
                    "total_size_mb": 0
                }
    
    def get_missing_embeddings(self, doc_ids: List[str]) -> List[str]:
        missing = []
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT doc_id FROM embeddings")
                    existing_ids = {row[0] for row in cursor.fetchall()}
                    missing = [doc_id for doc_id in doc_ids if doc_id not in existing_ids]
            except Exception as e:
                print(f"⚠ Error checking missing embeddings: {e}")
                missing = doc_ids
        return missing
    
    def has_embedding(self, doc_id: str) -> bool:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE doc_id = ?", (doc_id,))
                    return cursor.fetchone()[0] > 0
            except Exception as e:
                print(f"⚠ Error checking embedding: {e}")
                return False
