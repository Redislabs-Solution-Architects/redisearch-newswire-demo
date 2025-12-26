#!/usr/bin/env python3
"""
Azure OpenAI Embedding Helper for NewsWire Demo
Handles batch embedding generation with rate limiting and retry logic
"""

import time
from typing import List, Dict, Optional
from openai import AzureOpenAI
from openai import RateLimitError, APIError
import numpy as np

# Import config
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_DIMENSIONS
)

# ============================================
# AZURE OPENAI CLIENT
# ============================================

class NewsEmbedder:
    """Handles embedding generation for news articles"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.dimensions = EMBEDDING_DIMENSIONS
        
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
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            
        Returns:
            List of floats (embedding vector) or None on failure
        """
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
        Generate embeddings for multiple texts with rate limiting
        
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
            
            try:
                # Azure OpenAI supports batch embeddings
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.deployment
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting: small delay between batches
                if i + batch_size < total:
                    time.sleep(0.1)  # 100ms delay
                    
            except RateLimitError as e:
                print(f"\n⚠️  Rate limit hit at batch {i}-{batch_end}, retrying individually...")
                # Fall back to individual embeddings for this batch
                for text in batch:
                    emb = self.embed_single(text)
                    embeddings.append(emb)
                    time.sleep(0.5)  # Slower rate for recovery
                    
            except Exception as e:
                print(f"\n❌ Batch error at {i}-{batch_end}: {e}")
                # Fall back to individual embeddings
                for text in batch:
                    emb = self.embed_single(text)
                    embeddings.append(emb)
        
        if show_progress:
            print()  # New line after progress
        
        return embeddings
    
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

# ============================================
# MAIN - FOR TESTING
# ============================================

if __name__ == "__main__":
    """Test the embedder"""
    
    print("=" * 70)
    print("🧪 AZURE OPENAI EMBEDDER TEST")
    print("=" * 70)
    print()
    
    # Initialize embedder
    embedder = NewsEmbedder()
    
    # Test connection
    if not embedder.test_connection():
        print("\n❌ Failed to connect to Azure OpenAI")
        print("\nPlease check:")
        print("  1. AZURE_OPENAI_ENDPOINT is correct")
        print("  2. AZURE_OPENAI_KEY is valid")
        print("  3. AZURE_OPENAI_DEPLOYMENT matches your deployment name")
        print("  4. Model deployment is active in Azure")
        exit(1)
    
    print()
    print("=" * 70)
    print("🧪 TESTING ARTICLE EMBEDDING")
    print("=" * 70)
    print()
    
    # Test with a sample article
    test_article = {
        "title": "Breaking: AI Revolutionizes Search Technology",
        "summary": "New vector search capabilities enable semantic understanding of queries.",
        "content": "Artificial intelligence is transforming how we search and retrieve information. Vector embeddings allow systems to understand meaning beyond keywords, enabling more intuitive and accurate search results. This technology is being adopted across industries..."
    }
    
    print("📰 Test Article:")
    print(f"   Title: {test_article['title']}")
    print(f"   Summary: {test_article['summary'][:80]}...")
    print()
    
    # Prepare and embed
    text = embedder.prepare_text(
        test_article['title'],
        test_article['summary'],
        test_article['content']
    )
    
    print(f"📝 Prepared text ({len(text)} chars):")
    print(f"   {text[:150]}...")
    print()
    
    print("🔄 Generating embedding...")
    embedding = embedder.embed_single(text)
    
    if embedding:
        print(f"✅ Embedding generated successfully!")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
        print()
        
        # Test batch embedding
        print("=" * 70)
        print("🧪 TESTING BATCH EMBEDDING")
        print("=" * 70)
        print()
        
        test_texts = [
            "First article about technology",
            "Second article about finance",
            "Third article about sports"
        ]
        
        print(f"📦 Embedding batch of {len(test_texts)} articles...")
        batch_embeddings = embedder.embed_batch(test_texts, batch_size=3)
        
        print(f"✅ Batch embedding complete!")
        print(f"   Generated {len(batch_embeddings)} embeddings")
        print()
        
    else:
        print("❌ Failed to generate embedding")
        exit(1)
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Next step: Run load_data_with_vectors.py to embed and load your articles")
    print()