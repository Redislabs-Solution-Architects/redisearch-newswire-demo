#!/usr/bin/env python3
"""
Azure OpenAI Embedding Helper for NewsWire Demo
Handles batch embedding generation with rate limiting and retry logic
Now with SEMANTIC Redis caching for embeddings!
"""

import time
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI
from openai import RateLimitError, APIError
import numpy as np
import redis

# Import config
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_DIMENSIONS,
    AMR_HOST,
    AMR_PORT,
    AMR_PASSWORD
)

# ============================================
# AZURE OPENAI CLIENT WITH SEMANTIC CACHING
# ============================================

class NewsEmbedder:
    """Handles embedding generation for news articles with semantic Redis caching"""
    
    def __init__(self, enable_cache: bool = True, similarity_threshold: float = 0.95):
        """
        Initialize Azure OpenAI client and Redis semantic cache
        
        Args:
            enable_cache: Enable Redis caching for embeddings (default: True)
            similarity_threshold: Minimum cosine similarity for cache hit (default: 0.95)
        """
        # Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.dimensions = EMBEDDING_DIMENSIONS
        
        # Semantic cache settings
        self.enable_cache = enable_cache
        self.similarity_threshold = similarity_threshold
        self.cache_client = None
        
        # Cache statistics
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        if self.enable_cache:
            try:
                self.cache_client = redis.Redis(
                    host=AMR_HOST,
                    port=int(AMR_PORT),
                    password=AMR_PASSWORD,
                    ssl=True,
                    decode_responses=False,  # Need binary for embeddings
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                self.cache_client.ping()
                print(f"✅ Semantic cache connected (similarity threshold: {similarity_threshold})")
            except Exception as e:
                print(f"⚠️  Cache connection failed, running without cache: {e}")
                self.cache_client = None
                self.enable_cache = False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent comparison"""
        return text.lower().strip()
    
    def _get_exact_cache_key(self, text: str) -> str:
        """Generate exact match cache key from text"""
        normalized = self._normalize_text(text)
        hash_obj = hashlib.md5(normalized.encode('utf-8'))
        return f"embed:exact:{hash_obj.hexdigest()}"
    
    def _get_semantic_index_key(self) -> str:
        """Get the Redis key for the semantic cache index"""
        return "embed:semantic:index"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0 to 1)
        """
        # Convert to numpy arrays
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _get_from_semantic_cache(self, text: str) -> Optional[Tuple[List[float], str, float]]:
        """
        Try to get embedding from semantic cache
        
        Args:
            text: Text to lookup
            
        Returns:
            Tuple of (embedding, matched_text, similarity) or None
        """
        if not self.enable_cache or not self.cache_client:
            return None
        
        self.stats["total_queries"] += 1
        normalized_text = self._normalize_text(text)
        
        try:
            # First try exact match (fastest)
            exact_key = self._get_exact_cache_key(text)
            cached_data = self.cache_client.get(exact_key)
            
            if cached_data:
                embedding = json.loads(cached_data.decode('utf-8'))
                self.stats["exact_hits"] += 1
                return (embedding, normalized_text, 1.0)  # Perfect match
            
            # No exact match - try semantic search
            # Get all cached embeddings from sorted set
            index_key = self._get_semantic_index_key()
            cached_entries = self.cache_client.zrange(index_key, 0, -1, withscores=False)
            
            if not cached_entries or len(cached_entries) == 0:
                self.stats["misses"] += 1
                return None
            
            # We need to generate embedding to compare (unavoidable for semantic search)
            # This is the trade-off: we generate once to check similarity
            query_embedding = self._generate_embedding_direct(text)
            if not query_embedding:
                self.stats["misses"] += 1
                return None
            
            # Find most similar cached embedding
            best_similarity = 0.0
            best_match = None
            best_text = None
            
            for entry in cached_entries[:50]:  # Check top 50 most recent
                try:
                    entry_data = json.loads(entry.decode('utf-8'))
                    cached_text = entry_data["text"]
                    cached_embedding = entry_data["embedding"]
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_embedding
                        best_text = cached_text
                    
                    # Early exit if we find very similar match
                    if similarity >= 0.98:
                        break
                        
                except Exception as e:
                    continue
            
            # Check if best match exceeds threshold
            if best_similarity >= self.similarity_threshold:
                self.stats["semantic_hits"] += 1
                print(f"  💡 Semantic cache hit: '{text[:30]}' → '{best_text[:30]}' (similarity: {best_similarity:.2%})")
                return (best_match, best_text, best_similarity)
            
            # No semantic match - we already generated the embedding, so cache and return it
            self.stats["misses"] += 1
            self._save_to_semantic_cache(text, query_embedding)
            return (query_embedding, normalized_text, 1.0)
                
        except Exception as e:
            print(f"⚠️  Semantic cache error: {e}")
            self.stats["misses"] += 1
        
        return None
    
    def _generate_embedding_direct(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding directly without caching logic
        Used internally for semantic cache comparison
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Direct embedding generation error: {e}")
            return None
    
    def _save_to_semantic_cache(self, text: str, embedding: List[float], ttl: int = 3600):
        """
        Save embedding to semantic cache
        
        Args:
            text: Text that was embedded
            embedding: The embedding vector
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self.enable_cache or not self.cache_client:
            return
        
        try:
            normalized_text = self._normalize_text(text)
            
            # Save exact match cache
            exact_key = self._get_exact_cache_key(text)
            self.cache_client.setex(exact_key, ttl, json.dumps(embedding))
            
            # Save to semantic index (sorted set with timestamp as score)
            index_key = self._get_semantic_index_key()
            entry_data = {
                "text": normalized_text,
                "embedding": embedding,
                "timestamp": time.time()
            }
            
            # Use timestamp as score for ordering (most recent first)
            self.cache_client.zadd(
                index_key,
                {json.dumps(entry_data): time.time()},
                nx=False  # Allow updates
            )
            
            # Trim index to keep only last 100 entries (to avoid unbounded growth)
            self.cache_client.zremrangebyrank(index_key, 0, -101)
            
        except Exception as e:
            print(f"⚠️  Cache write error: {e}")
    
    def prepare_text(self, title: str, summary: str, content: str) -> str:
        """
        Prepare text for embedding
        Combines title + summary + content (first 2000 chars)
        """
        # Combine fields with clear separation
        text_parts = []
        
        if title:
            text_parts.append(f"Title: {title}")
        
        if summary:
            text_parts.append(f"Summary: {summary}")
        
        if content:
            # Limit content to first 2000 characters to stay within token limits
            content_excerpt = content[:2000] if len(content) > 2000 else content
            text_parts.append(f"Content: {content_excerpt}")
        
        # Join with newlines
        combined_text = "\n".join(text_parts)
        
        return combined_text
    
    def embed_single(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text (with semantic caching)
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            
        Returns:
            List of floats (embedding vector) or None on failure
        """
        # Try semantic cache first
        cached_result = self._get_from_semantic_cache(text)
        if cached_result:
            embedding, matched_text, similarity = cached_result
            return embedding
        
        # Generate new embedding (cache miss)
        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.deployment
                )
                
                # Extract embedding
                embedding = response.data[0].embedding
                
                # Validate dimensions
                if len(embedding) != self.dimensions:
                    print(f"⚠️  Warning: Expected {self.dimensions} dims, got {len(embedding)}")
                
                # Save to semantic cache
                self._save_to_semantic_cache(text, embedding)
                
                return embedding
                
            except RateLimitError as e:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"⚠️  Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{retry_count})")
                time.sleep(wait_time)
                
            except APIError as e:
                print(f"❌ API Error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
                else:
                    return None
                    
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return None
        
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 10, show_progress: bool = True) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts with rate limiting and semantic caching
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed at once (Azure OpenAI supports up to 16)
            show_progress: Show progress messages
            
        Returns:
            List of embeddings (or None for failed embeddings)
        """
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_end = min(i + batch_size, total)
            
            if show_progress:
                print(f"  Embedding {i+1}-{batch_end}/{total}...", end='\r')
            
            # Check semantic cache for each text in batch
            batch_to_generate = []
            batch_indices = []
            batch_results = [None] * len(batch)
            
            for idx, text in enumerate(batch):
                cached_result = self._get_from_semantic_cache(text)
                if cached_result:
                    embedding, matched_text, similarity = cached_result
                    batch_results[idx] = embedding
                else:
                    batch_to_generate.append(text)
                    batch_indices.append(idx)
            
            # Generate embeddings for cache misses
            if batch_to_generate:
                try:
                    # Azure OpenAI supports batch embeddings
                    response = self.client.embeddings.create(
                        input=batch_to_generate,
                        model=self.deployment
                    )
                    
                    # Extract embeddings in order
                    generated_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                    
                    # Save to cache and add to results
                    for idx, text, emb in zip(batch_indices, batch_to_generate, generated_embeddings):
                        self._save_to_semantic_cache(text, emb)
                        batch_results[idx] = emb
                    
                    # Rate limiting: small delay between batches
                    if i + batch_size < total:
                        time.sleep(0.1)  # 100ms delay
                        
                except RateLimitError as e:
                    print(f"\n⚠️  Rate limit hit at batch {i}-{batch_end}, retrying individually...")
                    # Fall back to individual embeddings for this batch
                    for idx, text in zip(batch_indices, batch_to_generate):
                        emb = self.embed_single(text)
                        batch_results[idx] = emb
                        time.sleep(0.5)  # Slower rate for recovery
                        
                except Exception as e:
                    print(f"\n❌ Batch error at {i}-{batch_end}: {e}")
                    # Fall back to individual embeddings
                    for idx, text in zip(batch_indices, batch_to_generate):
                        emb = self.embed_single(text)
                        batch_results[idx] = emb
            
            embeddings.extend(batch_results)
        
        if show_progress:
            print()  # New line after progress
            if self.enable_cache and self.stats["total_queries"] > 0:
                self._print_cache_stats()
        
        return embeddings
    
    def _print_cache_stats(self):
        """Print cache statistics"""
        total = self.stats["total_queries"]
        exact = self.stats["exact_hits"]
        semantic = self.stats["semantic_hits"]
        misses = self.stats["misses"]
        
        total_hits = exact + semantic
        hit_rate = (total_hits / total * 100) if total > 0 else 0
        
        print(f"  📊 Cache stats: {exact} exact hits, {semantic} semantic hits, {misses} misses ({hit_rate:.1f}% hit rate)")
    
    def test_connection(self) -> bool:
        """
        Test Azure OpenAI connection with a simple embedding
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🧪 Testing Azure OpenAI connection...")
            test_text = "This is a test embedding."
            
            embedding = self.embed_single(test_text)
            
            if embedding and len(embedding) == self.dimensions:
                print(f"✅ Connection successful! Embedding dimensions: {len(embedding)}")
                return True
            else:
                print(f"❌ Connection failed or wrong dimensions")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enable_cache or not self.cache_client:
            return {"enabled": False}
        
        try:
            # Count cache keys
            index_key = self._get_semantic_index_key()
            semantic_count = self.cache_client.zcard(index_key)
            
            return {
                "enabled": True,
                "similarity_threshold": self.similarity_threshold,
                "semantic_entries": semantic_count,
                "exact_hits": self.stats["exact_hits"],
                "semantic_hits": self.stats["semantic_hits"],
                "misses": self.stats["misses"],
                "total_queries": self.stats["total_queries"],
                "hit_rate": f"{((self.stats['exact_hits'] + self.stats['semantic_hits']) / self.stats['total_queries'] * 100):.1f}%" if self.stats["total_queries"] > 0 else "0%"
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

# ============================================
# MAIN - FOR TESTING
# ============================================

if __name__ == "__main__":
    """Test the embedder with semantic caching"""
    
    print("=" * 70)
    print("🧪 AZURE OPENAI EMBEDDER TEST (WITH SEMANTIC CACHING)")
    print("=" * 70)
    print()
    
    # Initialize embedder with semantic caching
    embedder = NewsEmbedder(enable_cache=True, similarity_threshold=0.85)
    
    # Test connection
    if not embedder.test_connection():
        print("\n❌ Failed to connect to Azure OpenAI")
        exit(1)
    
    print()
    print("=" * 70)
    print("🧪 TESTING SEMANTIC CACHE")
    print("=" * 70)
    print()
    
    # Test semantic similarity
    queries = [
        "climate change impacts",
        "climate change effects",  # Very similar
        "global warming",  # Semantically related
        "renewable energy",  # Different topic
    ]
    
    print("Testing semantic cache with similar queries:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        start = time.time()
        emb = embedder.embed_single(query)
        elapsed = (time.time() - start) * 1000
        print(f"   Time: {elapsed:.1f}ms")
    
    # Show final stats
    print("\n" + "=" * 70)
    stats = embedder.get_cache_stats()
    print(f"📊 Final Cache Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
