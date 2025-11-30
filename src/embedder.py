import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .cache_manager import CacheManager


class EmbeddingGenerator:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 cache_db: str = "cache/embeddings_cache.db",
                 max_seq_length: int = 512,
                 chunk_overlap: int = 50):
        print(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}") from e
        
        try:
            self.cache_manager = CacheManager(cache_db=cache_db)
        except Exception as e:
            print(f"Warning: Could not initialize cache manager: {e}")
            raise RuntimeError(f"Failed to initialize cache: {e}") from e
        
        self.max_seq_length = max_seq_length
        self.chunk_overlap = chunk_overlap
        
        # Get tokenizer from the model - REQUIRED for token-based chunking
        self.tokenizer = None
        try:
            # SentenceTransformer typically stores tokenizer in the first module
            # Try multiple ways to access it
            if hasattr(self.model, 'tokenizer'):
                self.tokenizer = self.model.tokenizer
            elif hasattr(self.model, '_modules'):
                # Access the first module (usually the transformer)
                modules = self.model._modules
                if '0' in modules:
                    first_module = modules['0']
                    if hasattr(first_module, 'tokenizer'):
                        self.tokenizer = first_module.tokenizer
                    elif hasattr(first_module, 'auto_model'):
                        auto_model = first_module.auto_model
                        if hasattr(auto_model, 'tokenizer'):
                            self.tokenizer = auto_model.tokenizer
                # Try other module keys
                for key in modules:
                    module = modules[key]
                    if hasattr(module, 'tokenizer'):
                        self.tokenizer = module.tokenizer
                        break
            
            # Verify tokenizer is usable
            if self.tokenizer is not None:
                try:
                    # Test tokenizer with a simple string
                    test_tokens = self.tokenizer.encode("test", add_special_tokens=False)
                    if not isinstance(test_tokens, (list, tuple)) or len(test_tokens) == 0:
                        self.tokenizer = None
                except Exception as e:
                    print(f"Error: Tokenizer test failed: {e}")
                    self.tokenizer = None
                    
        except Exception as e:
            print(f"Error: Could not access tokenizer: {e}")
            self.tokenizer = None
        
        # Tokenizer is REQUIRED - raise error if not found
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available from the model. "
                "Token-based chunking requires access to the model's tokenizer. "
                "Please ensure the SentenceTransformer model is properly loaded."
            )
        
        print("✓ Tokenizer loaded successfully for token-based chunking")
        
        print(f"Embedding generator initialized (max_seq_length={max_seq_length}, chunk_overlap={chunk_overlap})")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens using the model's tokenizer.
        REQUIRES tokenizer to be available - no fallbacks.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available. Cannot count tokens without tokenizer. "
                "This should not happen if EmbeddingGenerator was properly initialized."
            )
        
        # Use tokenizer to get actual token count
        # Don't add special tokens for counting (we'll add them during encoding)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def _estimate_tokens(self, text: str) -> int:
        """Legacy method for backward compatibility. Use _count_tokens instead."""
        return self._count_tokens(text)
    
    def _sentence_tokenize(self, text: str) -> List[str]:
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except (ImportError, LookupError):
            import re
            sentences = re.split(r'([.!?]+\s+)', text)
            result = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    result.append(sentences[i] + sentences[i + 1])
                else:
                    result.append(sentences[i])
            if len(sentences) % 2 == 1:
                result.append(sentences[-1])
            return [s.strip() for s in result if s.strip()]
    
    def _chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Chunk text based on token counts using the model's tokenizer.
        Ensures chunks stay under max_seq_length tokens with proper overlap.
        """
        if chunk_size is None:
            chunk_size = self.max_seq_length
        
        # Count total tokens in the text
        total_tokens = self._count_tokens(text)
        
        # If text fits in one chunk, return as-is
        if total_tokens <= chunk_size:
            return [text]
        
        # Split into sentences for better chunking
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return [text]
        
        # Count tokens per sentence
        sentence_tokens = []
        for sentence in sentences:
            token_count = self._count_tokens(sentence)
            sentence_tokens.append((sentence, token_count))
        
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        overlap_tokens = min(self.chunk_overlap, chunk_size // 2)
        
        i = 0
        while i < len(sentence_tokens):
            sentence, sentence_token_count = sentence_tokens[i]
            
            # Handle extremely long sentences that exceed chunk size
            if sentence_token_count > chunk_size:
                # Split long sentence by words and chunk at word boundaries
                words = sentence.split()
                word_chunk = []
                word_chunk_tokens = 0
                
                for word in words:
                    test_chunk = word_chunk + [word]
                    test_chunk_text = " ".join(test_chunk)
                    test_chunk_tokens = self._count_tokens(test_chunk_text)
                    
                    # If adding this word would exceed chunk size, finalize current chunk
                    if test_chunk_tokens > chunk_size and word_chunk:
                        chunk_text = " ".join(word_chunk)
                        chunks.append(chunk_text)
                        
                        # Start new chunk with overlap (last few words)
                        overlap_words = []
                        for w in reversed(word_chunk):
                            test_overlap = overlap_words + [w]
                            test_overlap_text = " ".join(test_overlap)
                            test_overlap_tokens = self._count_tokens(test_overlap_text)
                            if test_overlap_tokens <= overlap_tokens:
                                overlap_words.insert(0, w)
                            else:
                                break
                        
                        word_chunk = overlap_words
                        if word_chunk:
                            word_chunk_tokens = self._count_tokens(" ".join(word_chunk))
                        else:
                            word_chunk_tokens = 0
                    
                    word_chunk.append(word)
                    word_chunk_tokens = self._count_tokens(" ".join(word_chunk))
                
                # Add remaining words as current chunk
                if word_chunk:
                    current_chunk_sentences = [" ".join(word_chunk)]
                    current_token_count = word_chunk_tokens
                
                i += 1
                continue
            
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk_sentences + [sentence]
            test_chunk_text = " ".join(test_chunk)
            test_chunk_tokens = self._count_tokens(test_chunk_text)
            
            if test_chunk_tokens > chunk_size and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                overlap_sentences = []
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent = current_chunk_sentences[j]
                    test_overlap = overlap_sentences + [sent]
                    test_overlap_text = " ".join(test_overlap)
                    test_overlap_tokens = self._count_tokens(test_overlap_text)
                    if test_overlap_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                if current_chunk_sentences:
                    current_token_count = self._count_tokens(" ".join(current_chunk_sentences))
                else:
                    current_token_count = 0
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count = self._count_tokens(" ".join(current_chunk_sentences))
            i += 1
        
        # Add final chunk if it exists
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        if not chunks:
            return [text]
        
        # Verify all chunks are within token limit and split further if needed
        verified_chunks = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            if chunk_tokens <= chunk_size:
                verified_chunks.append(chunk)
            else:
                # Chunk still too large, split further by words
                print(f"    ⚠ Chunk exceeded {chunk_size} tokens ({chunk_tokens}), splitting further...")
                
                # Split into smaller chunks at word boundaries
                words = chunk.split()
                word_chunk = []
                
                for word in words:
                    test_chunk = word_chunk + [word]
                    test_chunk_text = " ".join(test_chunk)
                    test_chunk_tokens = self._count_tokens(test_chunk_text)
                    
                    if test_chunk_tokens > chunk_size and word_chunk:
                        verified_chunks.append(" ".join(word_chunk))
                        word_chunk = [word]
                    else:
                        word_chunk.append(word)
                
                if word_chunk:
                    verified_chunks.append(" ".join(word_chunk))
        
        return verified_chunks if verified_chunks else [text]
    
    def _mean_pool_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("No embeddings to pool")
        
        embeddings_float32 = [np.array(emb, dtype=np.float32) for emb in embeddings]
        embeddings_array = np.array(embeddings_float32, dtype=np.float32)
        mean_embedding = np.mean(embeddings_array, axis=0, dtype=np.float32)
        return np.array(mean_embedding, dtype=np.float32)
    
    def embed_document(self, doc_id: str, text: str, doc_hash: str, filename: str = "", 
                      use_chunking: bool = True) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError(f"Empty text provided for document {doc_id}")
        
        cached_embedding = self.cache_manager.get(doc_id, doc_hash)
        if cached_embedding is not None and len(cached_embedding) > 0:
            return np.array(cached_embedding, dtype=np.float32)
        
        if len(doc_id) < 50:
            print(f"  Generating embedding for: {doc_id}")
        
        try:
            if use_chunking:
                chunks = self._chunk_text(text)
                
                if len(chunks) > 1:
                    if len(doc_id) < 50:
                        total_tokens = self._count_tokens(text)
                        chunk_info = []
                        for i, chunk in enumerate(chunks):
                            chunk_tokens = self._count_tokens(chunk)
                            chunk_info.append(f"chunk{i+1}({chunk_tokens}t)")
                        print(f"    → Split into {len(chunks)} chunks (total: {total_tokens} tokens)")
                        print(f"      Chunks: {', '.join(chunk_info)}")
                    
                    chunk_embeddings = []
                    for i, chunk in enumerate(chunks):
                        if not chunk or not chunk.strip():
                            continue
                        
                        # Verify chunk size before encoding
                        chunk_tokens = self._count_tokens(chunk)
                        if chunk_tokens > self.max_seq_length:
                            print(f"    ⚠ Warning: Chunk {i+1} has {chunk_tokens} tokens (exceeds {self.max_seq_length})")
                        
                        chunk_emb = self.model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
                        chunk_emb = np.array(chunk_emb, dtype=np.float32)
                        chunk_embeddings.append(chunk_emb)
                    
                    if not chunk_embeddings:
                        raise ValueError(f"No valid chunks generated for document {doc_id}")
                    
                    embedding = self._mean_pool_embeddings(chunk_embeddings)
                else:
                    embedding = np.array(
                        self.model.encode(text, convert_to_numpy=True), 
                        dtype=np.float32
                    )
            else:
                embedding = np.array(
                    self.model.encode(text, convert_to_numpy=True), 
                    dtype=np.float32
                )
            
            if embedding is None or len(embedding) == 0:
                raise ValueError(f"Generated empty embedding for document {doc_id}")
            
            embedding = np.array(embedding, dtype=np.float32)
            
            model_name = getattr(self.model, '_model_name', None) or \
                        getattr(self.model, 'model_name', None) or \
                        "sentence-transformers/all-MiniLM-L6-v2"
            
            self.cache_manager.set(
                doc_id, embedding, doc_hash, filename, 
                save_immediately=False,
                model_name=model_name,
                model_version="default"
            )
            
            return embedding
            

        except Exception as e:
            print(f"  ❌ Error generating embedding for {doc_id}: {e}")
            raise
    

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return np.array(embedding, dtype=np.float32)
    

    def embed_batch(self, documents: List[Dict], use_chunking: bool = True, 
                   batch_size: int = 64, show_progress_bar: bool = True) -> Dict[str, np.ndarray]:
        embeddings = {}
        cached_count = 0
        new_count = 0
        hash_mismatch_count = 0
        not_in_cache_count = 0
        missing_embedding_count = 0
        error_count = 0
        
        print(f"Processing {len(documents)} documents for embedding generation...")
        
        documents_to_embed_batch = []
        documents_to_embed_chunked = []
        
        for doc in documents:
            doc_id = doc["doc_id"]
            doc_hash = doc["hash"]
            doc_text = doc.get("text", "")
            doc_filename = doc.get("filename", "")
            
            if not doc_text or not doc_text.strip():
                continue
            
            cached_embedding = self.cache_manager.get(doc_id, doc_hash)
            if cached_embedding is not None:
                embeddings[doc_id] = np.array(cached_embedding, dtype=np.float32)
                cached_count += 1

            else:
                # Use tokenizer-based token counting
                token_count = self._count_tokens(doc_text)
                if use_chunking and token_count > self.max_seq_length:
                    documents_to_embed_chunked.append(doc)
                else:
                    documents_to_embed_batch.append((doc_id, doc_text, doc_hash, doc_filename))
                not_in_cache_count += 1
                
        
        if documents_to_embed_batch:
            print(f"  Batch processing {len(documents_to_embed_batch)} documents...")
            texts = [text for _, text, _, _ in documents_to_embed_batch]
            
            batch_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            
            model_name = getattr(self.model, '_model_name', None) or \
                        getattr(self.model, 'model_name', None) or \
                        "sentence-transformers/all-MiniLM-L6-v2"
            
            for (doc_id, _, doc_hash, doc_filename), embedding in zip(documents_to_embed_batch, batch_embeddings):
                embedding = np.array(embedding, dtype=np.float32)
                embeddings[doc_id] = embedding
                self.cache_manager.set(
                    doc_id, embedding, doc_hash, doc_filename, 
                    save_immediately=False,
                    model_name=model_name,
                    model_version="default"
                )
                new_count += 1
            
            if new_count >= 100:
                print(f"  💾 Periodic save ({new_count} new embeddings so far)...")
                self.cache_manager.save()
        
        if documents_to_embed_chunked:
            print(f"  Processing {len(documents_to_embed_chunked)} documents with chunking...")
            model_name = getattr(self.model, '_model_name', None) or \
                        getattr(self.model, 'model_name', None) or \
                        "sentence-transformers/all-MiniLM-L6-v2"
            
            for doc in documents_to_embed_chunked:
                try:
                    embedding = self.embed_document(
                        doc["doc_id"],
                        doc["text"],
                        doc["hash"],
                        doc.get("filename", ""),
                        use_chunking=True
                    )
                    if embedding is not None and len(embedding) > 0:
                        embeddings[doc["doc_id"]] = np.array(embedding, dtype=np.float32)
                        new_count += 1
                except Exception as e:
                    print(f"⚠ Error processing chunked document {doc['doc_id']}: {e}")
                    error_count += 1
        
        doc_ids_set = {doc["doc_id"] for doc in documents if doc.get("text", "").strip()}
        missing_doc_ids = doc_ids_set - set(embeddings.keys())
        
        if missing_doc_ids:
            print(f"\n⚠ Warning: {len(missing_doc_ids)} documents missing embeddings. Generating now...")
            for doc_id in missing_doc_ids:
                doc = next((d for d in documents if d["doc_id"] == doc_id), None)
                if doc and doc.get("text", "").strip():
                    try:
                        embedding = self.embed_document(
                            doc_id,
                            doc["text"],
                            doc["hash"],
                            doc.get("filename", ""),
                            use_chunking=use_chunking
                        )
                        if embedding and len(embedding) > 0:
                            embeddings[doc_id] = embedding
                            missing_embedding_count += 1
                            new_count += 1
                    except Exception as e:
                        print(f"⚠ Error generating embedding for missing doc {doc_id}: {e}")
                        error_count += 1
        
        if len(embeddings) > 0:
            if new_count > 0 or missing_embedding_count > 0:
                print("\n💾 Saving cache to disk...")
            else:
                print("\n💾 Ensuring cache file exists...")
            self.cache_manager.save()
        else:
            print("\n⚠ Warning: No embeddings to save!")
        
        print(f"\n📊 Embedding Summary:")
        print(f"   ✓ Cached (reused): {cached_count}")
        print(f"   ✗ New (generated): {new_count}")
        if missing_embedding_count > 0:
            print(f"   🔄 Missing (regenerated): {missing_embedding_count}")
        if hash_mismatch_count > 0:
            print(f"   ⚠ Hash mismatches: {hash_mismatch_count} (content changed)")
        if not_in_cache_count > 0:
            print(f"   ℹ Not in cache: {not_in_cache_count} (first time)")
        if error_count > 0:
            print(f"   ❌ Errors: {error_count} (failed to generate)")
        
        total_expected = len([d for d in documents if d.get("text", "").strip()])
        total_embeddings = len(embeddings)
        print(f"\n✅ Final Status:")
        print(f"   Expected documents: {total_expected}")
        print(f"   Documents with embeddings: {total_embeddings}")
        if total_embeddings < total_expected:
            print(f"   ⚠ Missing: {total_expected - total_embeddings} documents")
        else:
            print(f"   ✓ All documents have embeddings!")
        
        return embeddings
