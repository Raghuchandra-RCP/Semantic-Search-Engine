import re
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from .search_engine import VectorSearchEngine


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    
    def __init__(self, **data):
        super().__init__(**data)
        if not (1 <= self.top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")
        if len(self.query) > 10000:
            raise ValueError("Query is too long (max 10000 characters)")
        if not self.query.strip():
            raise ValueError("Query cannot be empty")


class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: Optional[Dict] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


class SearchAPI:
    
    def __init__(self, search_engine: VectorSearchEngine):
        self.app = FastAPI(
            title="Multi-document Embedding Search Engine",
            description="Search engine with caching and ranking explanations",
            version="1.0.0"
        )
        self.search_engine = search_engine
        self._setup_routes()
    
    def _setup_routes(self):
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Multi-document Embedding Search Engine API",
                "endpoints": {
                    "/search": "POST - Search documents",
                    "/health": "GET - Health check",
                    "/stats": "GET - Cache and index statistics"
                }
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @self.app.get("/stats")
        async def stats():
            cache_stats = self.search_engine.embedder.cache_manager.stats()
            return {
                "cache_stats": cache_stats,
                "index_size": len(self.search_engine.doc_ids) if self.search_engine.doc_ids else 0
            }
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search(request: SearchRequest):
            if not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")
            
            if not (1 <= request.top_k <= 100):
                raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
            
            if len(request.query) > 10000:
                raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")
            
            raw_results = self.search_engine.search(request.query, request.top_k)
            
            results = []
            query_words = set(re.findall(r"\w+", request.query.lower()))
            
            for result in raw_results:
                doc_id = result["doc_id"]
                doc = self.search_engine.documents.get(doc_id, {})
                doc_text = doc.get("text", "").lower()
                
                explanation = self._generate_explanation(
                    query_words, doc_text, result["score"], doc.get("length", 0)
                )
                
                results.append(
                    SearchResult(
                        doc_id=result["doc_id"],
                        score=result["score"],
                        preview=result["preview"],
                        explanation=explanation
                    )
                )
            
            return SearchResponse(results=results)
    
    def _generate_explanation(self, query_words: set, doc_text: str, 
                              score: float, doc_length: int) -> Dict:
        doc_words = set(re.findall(r"\w+", doc_text))
        matched_keywords = sorted(list(query_words & doc_words))
        overlap_ratio = len(matched_keywords) / (len(query_words) or 1)
        length_norm = 1.0 / (1.0 + math.log1p(max(doc_length, 1) / 100))
        
        explanation_parts = []
        explanation_parts.append(f"Similarity score: {score:.4f}")
        
        if matched_keywords:
            keywords_str = ", ".join(matched_keywords)
            explanation_parts.append(f"Matched keywords: {keywords_str}")
            explanation_parts.append(f"Query overlap: {overlap_ratio * 100:.1f}%")
        else:
            explanation_parts.append("No exact keyword matches (semantic similarity only)")
        
        explanation_parts.append(f"Document length: {doc_length} characters")
        explanation_parts.append(f"Length normalization: {length_norm:.3f}")
        
        explanation_text = ". ".join(explanation_parts) + "."
        
        return {
            "matched_keywords": matched_keywords,
            "overlap_ratio": round(overlap_ratio, 3),
            "document_length": doc_length,
            "length_normalization_score": round(length_norm, 3),
            "similarity_score": round(score, 4),
            "explanation_text": explanation_text
        }


def create_app(search_engine: VectorSearchEngine) -> FastAPI:
    api = SearchAPI(search_engine)
    return api.app
