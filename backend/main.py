from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
import os
from pathlib import Path
import hashlib

# Import your existing redis service
from services.redis_search import RedisSearchService
from embedder import NewsEmbedder

# Initialize
app = FastAPI(title="NewsWire Search API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis service
redis_service = RedisSearchService()

# Initialize embedder for vector search
try:
    embedder = NewsEmbedder(enable_cache=True, similarity_threshold=0.90)
    vector_search_enabled = True
    print("✅ Vector search enabled")
except Exception as e:
    embedder = None
    vector_search_enabled = False
    print(f"⚠️  Vector search disabled: {e}")


# Helper functions for exact Query Caching
def get_query_cache_key(
    search_type: str,
    query: str,
    category: str,
    source: str,
    sort: str,
    offset: int,
    limit: int,
) -> str:
    """
    Generate consistent cache key for query results

    Args:
        search_type: "text", "vector", or "hybrid"
        query: Search query text
        category: Category filter
        source: Source filter
        sort: Sort parameter
        offset: Pagination offset
        limit: Results limit

    Returns:
        Cache key string
    """
    # Normalize parameters
    normalized_query = query.lower().strip()
    normalized_category = category.lower().strip() if category else "all"
    normalized_source = source.lower().strip() if source else "all"
    normalized_sort = sort.lower().strip()

    # Create cache key components
    key_parts = [
        search_type,
        normalized_query,
        normalized_category,
        normalized_source,
        normalized_sort,
        str(offset),
        str(limit),
    ]

    # Join with separators
    key_string = ":".join(key_parts)

    # Use MD5 hash to keep keys short
    key_hash = hashlib.md5(key_string.encode("utf-8")).hexdigest()

    return f"query_cache:{search_type}:{key_hash}"


def get_cached_result(cache_key: str):
    """
    Try to get cached query result from Redis

    Args:
        cache_key: The cache key

    Returns:
        Cached result dict or None
    """
    try:
        cached_value = redis_service.client.get(cache_key)
        if cached_value:
            # Parse JSON
            import json

            if isinstance(cached_value, bytes):
                cached_value = cached_value.decode("utf-8")
            return json.loads(cached_value)
    except Exception as e:
        print(f"⚠️  Cache read error: {e}")

    return None


def save_cached_result(cache_key: str, result: dict, ttl: int = 60):
    """Save query result to Redis cache"""
    try:
        import json

        # Serialize result to JSON
        cached_value = json.dumps(result)

        # Save with TTL (use SET with EX instead of SETEX if decode_responses=True)
        redis_service.client.set(cache_key, cached_value, ex=ttl)

    except Exception as e:
        print(f"⚠️  Cache write error: {e}")


# ============================================
# API Endpoints
# ============================================


@app.get("/api/debug-cache")
def debug_cache():
    """Debug cache functionality"""
    import time

    try:
        # Test 1: Can we access redis_service.client?
        can_ping = redis_service.client.ping()

        # Test 2: Try to set a simple value
        test_key = "debug:test:123"
        redis_service.client.set(test_key, "test_value", ex=60)

        # Test 3: Try to get it back
        retrieved = redis_service.client.get(test_key)

        # Test 4: Try cache helper functions
        cache_key = get_query_cache_key("text", "test", "All", "All", "relevance", 0, 5)
        test_data = {"test": "data", "timestamp": time.time()}
        save_cached_result(cache_key, test_data, ttl=60)
        retrieved_cached = get_cached_result(cache_key)

        return {
            "redis_ping": can_ping,
            "direct_set_get": retrieved,
            "cache_key": cache_key,
            "cache_saved": test_data,
            "cache_retrieved": retrieved_cached,
            "cache_working": retrieved_cached is not None,
        }
    except Exception as e:
        return {"error": str(e), "traceback": __import__("traceback").format_exc()}


@app.get("/api/search")
def search(
    q: str = "",
    category: str = "All",
    source: str = "All",
    sort: str = "relevance",
    fuzzy: bool = False,
    offset: int = 0,
    limit: int = 10,
):
    """Search articles with support for multiple categories and sources."""
    import time

    start_total = time.time()

    print(f"🔍 DEBUG: Starting search for q='{q}'")
    # Generate cache key
    cache_key = get_query_cache_key(
        search_type="text",
        query=q,
        category=category,
        source=source,
        sort=sort,
        offset=offset,
        limit=limit,
    )
    print(f"🔍 DEBUG: Cache key = {cache_key}")
    # Try to get from cache
    cached_result = get_cached_result(cache_key)
    print(f"🔍 DEBUG: Cached result = {cached_result is not None}")
    if cached_result:
        cache_time = (time.time() - start_total) * 1000
        print(f"⚡ [TEXT SEARCH - CACHE HIT] q='{q[:30]}' | Time: {cache_time:.1f}ms")

        # Add current cache retrieval time
        cached_result["latency_ms"] = cache_time

        return cached_result

    has_query = q and len(q.strip()) >= 2

    # Handle multiple categories (comma-separated)
    categories = None
    if category and category != "All":
        cat_list = [c.strip() for c in category.split(",") if c.strip()]
        if len(cat_list) == 1:
            categories = cat_list[0]
        elif len(cat_list) > 1:
            categories = cat_list

    # Handle multiple sources (comma-separated)
    source_param = None
    if source and source != "All":
        src_list = [s.strip() for s in source.split(",") if s.strip()]
        if len(src_list) == 1:
            source_param = src_list[0]
        elif len(src_list) > 1:
            source_param = src_list

    # Time the Redis search
    start_redis = time.time()
    response = redis_service.search(
        query=q if has_query else "",
        category=categories,
        source=source_param,
        author=None,
        use_fuzzy=fuzzy,
        sort_by=sort,
        offset=offset,
        limit=limit,
        highlight=has_query,
    )
    redis_time = (time.time() - start_redis) * 1000

    # Strip "article:" prefix from result IDs for frontend
    start_processing = time.time()
    if "results" in response:
        for result in response["results"]:
            if "id" in result and result["id"].startswith("article:"):
                result["id"] = result["id"].replace("article:", "")

            # Content is now included in RETURN fields
            if "content" not in result or not result.get("content"):
                result["content"] = ""

    processing_time = (time.time() - start_processing) * 1000
    total_time = (time.time() - start_total) * 1000

    # Add latency to response
    response["latency_ms"] = total_time
    response["latency_ms"] = total_time
    response["latency_breakdown"] = {
        "text_search_ms": round(max(redis_time - 0.5, 0.1), 2),
    }

    # Log timing breakdown
    num_results = len(response.get("results", []))
    print(
        f"⏱️  [TEXT SEARCH] q='{q[:30]}' | Redis: {redis_time:.1f}ms | Processing: {processing_time:.1f}ms | Total: {total_time:.1f}ms | Results: {num_results}"
    )

    # Save to cache (without latency_ms)
    cache_response = response.copy()
    if "latency_ms" in cache_response:
        del cache_response["latency_ms"]
    save_cached_result(cache_key, cache_response, ttl=60)

    return response


@app.get("/api/categories")
def get_categories():
    """Get all categories with counts."""
    try:
        # Get unique category values
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "category"
        )

        categories = []
        for cat in tag_values:
            if isinstance(cat, bytes):
                cat = cat.decode()

            # Skip "Other" category
            if cat.lower() == "other":
                continue

            try:
                count_result = redis_service.client.execute_command(
                    "FT.SEARCH",
                    redis_service.index_name,
                    f"@category:{{{cat}}}",
                    "LIMIT",
                    "0",
                    "0",
                )
                count = count_result[0] if count_result else 0
            except:
                count = 0

            if count > 0:
                categories.append({"name": cat.title(), "count": count})

        # Sort by count descending
        categories.sort(key=lambda x: x["count"], reverse=True)

        return {"categories": categories}
    except Exception as e:
        print(f"ERROR in get_categories: {e}")
        return {"categories": [], "error": str(e)}


@app.get("/api/sources")
def get_sources():
    """Get all sources with counts."""
    try:
        # Get unique source values using FT.TAGVALS
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "source"
        )

        sources = []
        for src in tag_values:
            if isinstance(src, bytes):
                src = src.decode()

            # Get count for this source
            escaped_src = src.replace("-", "\\-").replace(" ", "\\ ")
            count_result = redis_service.client.execute_command(
                "FT.SEARCH",
                redis_service.index_name,
                f"@source:{{{escaped_src}}}",
                "LIMIT",
                "0",
                "0",
            )
            count = count_result[0] if count_result else 0
            sources.append({"name": src, "count": count})

        # Sort by count descending
        sources.sort(key=lambda x: x["count"], reverse=True)

        return {"sources": sources[:20]}  # Top 20
    except Exception as e:
        print(f"ERROR in get_sources: {e}")
        return {"sources": [], "error": str(e)}


@app.get("/api/category-counts")
def get_category_counts(q: str = "", source: str = "All"):
    """Get category counts based on current filters."""
    try:
        # Get all categories first
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "category"
        )

        counts = {}
        for cat in tag_values:
            if isinstance(cat, bytes):
                cat = cat.decode()

            # Build query with filters
            query_parts = []

            if q and len(q.strip()) >= 2:
                query_parts.append(q.strip())

            if source and source != "All":
                # Handle multiple sources
                src_list = [s.strip() for s in source.split(",") if s.strip()]
                if len(src_list) == 1:
                    escaped_src = src_list[0].replace("-", "\\-").replace(" ", "\\ ")
                    query_parts.append(f"@source:{{{escaped_src}}}")
                elif len(src_list) > 1:
                    source_queries = []
                    for s in src_list:
                        escaped_s = s.replace("-", "\\-").replace(" ", "\\ ")
                        source_queries.append(f"@source:{{{escaped_s}}}")
                    query_parts.append(f"({' | '.join(source_queries)})")

            # Add category filter
            query_parts.append(f"@category:{{{cat}}}")

            search_query = " ".join(query_parts)

            count_result = redis_service.client.execute_command(
                "FT.SEARCH", redis_service.index_name, search_query, "LIMIT", "0", "0"
            )
            count = count_result[0] if count_result else 0
            counts[cat.title()] = count

        return {"counts": counts}
    except Exception as e:
        print(f"ERROR in get_category_counts: {e}")
        return {"counts": {}, "error": str(e)}


@app.get("/api/article/{doc_id:path}")
def get_article(doc_id: str):
    """Get full article by ID."""
    # Add article: prefix if not present
    if not doc_id.startswith("article:"):
        doc_id = f"article:{doc_id}"
    doc = redis_service.get_document(doc_id)

    if not doc:
        return {"error": f"Article not found: {doc_id}"}

    return {"article": doc}


@app.get("/api/spellcheck")
def spellcheck(q: str):
    """Get spelling suggestions."""
    suggestions = redis_service.spell_check(q)
    return {"suggestions": suggestions}


@app.get("/api/stats")
def get_stats():
    """Get index statistics."""
    stats = redis_service.get_index_stats()
    return stats


@app.get("/api/homepage")
def get_homepage():
    """Get homepage data: top stories + category sections."""
    try:
        # Get top categories using FT.TAGVALS
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "category"
        )

        # Get counts and sort
        cat_with_counts = []
        for cat in tag_values:
            if isinstance(cat, bytes):
                cat = cat.decode()
            count_result = redis_service.client.execute_command(
                "FT.SEARCH",
                redis_service.index_name,
                f"@category:{{{cat}}}",
                "LIMIT",
                "0",
                "0",
            )
            count = count_result[0] if count_result else 0
            cat_with_counts.append((cat, count))

        # Sort by count and get top 5
        cat_with_counts.sort(key=lambda x: x[1], reverse=True)
        top_categories = [c[0] for c in cat_with_counts[:5]]

        # Get top stories (most recent)
        top_stories = redis_service.search(
            query="",
            category=None,
            source=None,
            author=None,
            use_fuzzy=False,
            sort_by="date_desc",
            offset=0,
            limit=3,
            highlight=False,
        )

        # Get articles for each category
        category_sections = []
        for cat in top_categories:
            cat_results = redis_service.search(
                query="",
                category=cat.title(),
                source=None,
                author=None,
                use_fuzzy=False,
                sort_by="date_desc",
                offset=0,
                limit=3,
                highlight=False,
            )
            category_sections.append(
                {"category": cat.title(), "articles": cat_results.get("results", [])}
            )

        return {
            "top_stories": top_stories.get("results", []),
            "category_sections": category_sections,
        }
    except Exception as e:
        print(f"ERROR in get_homepage: {e}")
        return {"error": str(e), "top_stories": [], "category_sections": []}


@app.get("/api/trending")
def get_trending():
    """Get trending topics based on tags."""
    try:
        # Get unique tags using FT.TAGVALS
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "tags"
        )

        trending = []
        for tag in tag_values[:50]:
            if isinstance(tag, bytes):
                tag = tag.decode()

            if not tag or len(tag) < 2:
                continue

            escaped_tag = tag.replace("-", "\\-").replace(" ", "\\ ")
            try:
                count_result = redis_service.client.execute_command(
                    "FT.SEARCH",
                    redis_service.index_name,
                    f"@tags:{{{escaped_tag}}}",
                    "LIMIT",
                    "0",
                    "0",
                )
                count = count_result[0] if count_result else 0
                if count > 0:
                    trending.append({"tag": tag, "count": count})
            except:
                continue

        trending.sort(key=lambda x: x["count"], reverse=True)
        return {"trending": trending[:15]}
    except Exception as e:
        print(f"ERROR in get_trending: {e}")
        return {"trending": [], "error": str(e)}


@app.get("/api/autocomplete")
def autocomplete(q: str, limit: int = 7):
    """
    Unified autocomplete for all search modes.
    Returns curated suggestions from dictionary + spell check.
    """
    try:
        if not q or len(q) < 2:
            return {"suggestions": [], "spell_suggestion": None}

        # Get suggestions from curated dictionary
        suggestions_raw = redis_service.autocomplete(q, limit=limit * 2, fuzzy=False)

        # Parse and format suggestions
        suggestions = []
        seen = set()

        for sugg in suggestions_raw[:limit]:
            if isinstance(sugg, bytes):
                sugg = sugg.decode("utf-8")

            if sugg and sugg.lower() not in seen:
                # Determine suggestion type
                sugg_type = "article" if len(sugg) > 40 else "topic"

                suggestions.append({"title": sugg, "type": sugg_type})
                seen.add(sugg.lower())

        # Add spell correction if needed
        spell_suggestion_text = None
        if len(suggestions) < 3:  # If few results, try spell check
            spell_suggestions = redis_service.spell_check(q)
            if spell_suggestions:
                corrected = q
                for orig, sugg in spell_suggestions.items():
                    corrected = corrected.replace(orig, sugg)
                if corrected != q and corrected.lower() not in seen:
                    spell_suggestion_text = corrected

        return {
            "suggestions": suggestions,
            "spell_suggestion": spell_suggestion_text,
            "query": q,
        }

    except Exception as e:
        print(f"Autocomplete error: {e}")
        import traceback

        traceback.print_exc()
        return {"suggestions": [], "spell_suggestion": None}


# ============================================
# Vector Search Endpoints
# ============================================


@app.get("/api/search/vector")
def vector_search(
    q: str = "",
    category: str = "All",
    source: str = "All",
    offset: int = 0,
    limit: int = 10,
):
    """Pure vector search - semantic similarity only."""
    import time

    start_total = time.time()

    if not vector_search_enabled:
        return {"error": "Vector search not enabled", "results": [], "total": 0}

    if not q or len(q.strip()) < 2:
        return {"error": "Query too short", "results": [], "total": 0}

    try:
        # Generate query embedding
        start_embed = time.time()
        query_embedding = embedder.embed_single(q.strip())
        embed_time = (time.time() - start_embed) * 1000

        if not query_embedding:
            return {"error": "Failed to generate embedding", "results": [], "total": 0}

        # Build vector search query
        vector_query = f"*=>[KNN {limit + offset} @content_vector $vec AS score]"

        # Add filters
        filter_parts = []
        if category and category != "All":
            filter_parts.append(f"@category:{{{category}}}")
        if source and source != "All":
            escaped_src = source.replace("-", "\\-").replace(" ", "\\ ")
            filter_parts.append(f"@source:{{{escaped_src}}}")

        if filter_parts:
            vector_query = f"({' '.join(filter_parts)})=>[KNN {limit + offset} @content_vector $vec AS score]"

        # Execute search
        import numpy as np

        start_redis = time.time()
        result = redis_service.client.execute_command(
            "FT.SEARCH",
            redis_service.index_name,
            vector_query,
            "PARAMS",
            "2",
            "vec",
            np.array(query_embedding, dtype=np.float32).tobytes(),
            "SORTBY",
            "score",
            "LIMIT",
            str(offset),
            str(limit),
            "RETURN",
            "8",
            "$.title",
            "$.author",
            "$.category",
            "$.source",
            "$.published_at",
            "$.content",
            "$.word_count",
            "score",
            "DIALECT",
            "2",
        )
        redis_time = (time.time() - start_redis) * 1000

        # Parse results
        start_parse = time.time()
        total = result[0]
        results = []

        for i in range(1, len(result), 2):
            doc_id = (
                result[i].decode("utf-8") if isinstance(result[i], bytes) else result[i]
            )
            fields = result[i + 1]

            doc = {"id": doc_id.replace("article:", "")}
            for j in range(0, len(fields), 2):
                key = (
                    fields[j].decode("utf-8")
                    if isinstance(fields[j], bytes)
                    else fields[j]
                )
                value = (
                    fields[j + 1].decode("utf-8")
                    if isinstance(fields[j + 1], bytes)
                    else fields[j + 1]
                )

                # Remove $.  prefix from field names
                key = key.replace("$.", "") if key.startswith("$.") else key

                if key == "score":
                    doc["similarity_score"] = float(value)
                else:
                    doc[key] = value

            results.append(doc)

        parse_time = (time.time() - start_parse) * 1000
        total_time = (time.time() - start_total) * 1000

        # Log timing breakdown
        print(
            f"⏱️  [VECTOR SEARCH] q='{q[:30]}' | latency_breakdown: embedding_ms={embed_time:.2f}, redis_vector_search_ms={redis_time:.2f}, parse_ms={parse_time:.2f} | Total: {total_time:.1f}ms | Results: {len(results)}"
        )

        return {
            "results": results,
            "total": total,
            "query": q,
            "search_type": "vector",
            "latency_ms": total_time,
            "latency_breakdown": {
                "embedding_ms": round(embed_time, 2),
                "redis_vector_search_ms": round(redis_time, 2),
                "parse_ms": round(parse_time, 2),
            },
        }

    except Exception as e:
        print(f"Vector search error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "results": [], "total": 0}


@app.get("/api/search/hybrid")
def hybrid_search(
    q: str = "",
    category: str = "All",
    source: str = "All",
    offset: int = 0,
    limit: int = 10,
    vector_weight: float = 0.5,
):
    """Hybrid search - combines text search (BM25) with vector similarity."""
    import time

    start_total = time.time()

    if not vector_search_enabled:
        return {"error": "Vector search not enabled", "results": [], "total": 0}

    if not q or len(q.strip()) < 2:
        return {"error": "Query too short", "results": [], "total": 0}

    try:
        from concurrent.futures import ThreadPoolExecutor
        
        # Run text and vector searches in PARALLEL
        start_parallel = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches simultaneously
            text_future = executor.submit(
                search, q=q, category=category, source=source, 
                offset=0, limit=10, fuzzy=False
            )
            vector_future = executor.submit(
                vector_search, q=q, category=category, source=source, 
                offset=0, limit=10
            )
            
            # Wait for both to complete
            text_results = text_future.result()
            vector_results = vector_future.result()
        
        parallel_time = (time.time() - start_parallel) * 1000
        
        # Get individual times from results (if available)
        text_time = text_results.get("latency_ms", 0)
        vector_time = vector_results.get("latency_ms", 0)


        # Merge using Reciprocal Rank Fusion (RRF)
        start_merge = time.time()
        k = 60  # RRF constant
        scores = {}

        # Score text results
        for rank, doc in enumerate(text_results.get("results", []), 1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - vector_weight) / (k + rank)

        # Score vector results
        for rank, doc in enumerate(vector_results.get("results", []), 1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank)

        # Sort by combined score
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Fetch full documents
        results = []
        all_docs = {
            doc["id"]: doc
            for doc in text_results.get("results", [])
            + vector_results.get("results", [])
        }

        for doc_id in ranked_ids[offset : offset + limit]:
            if doc_id in all_docs:
                doc = all_docs[doc_id].copy()
                doc["hybrid_score"] = scores[doc_id]
                results.append(doc)

        merge_time = (time.time() - start_merge) * 1000
        total_time = (time.time() - start_total) * 1000

        # Log timing breakdown
        print(
            f"⏱️  [HYBRID SEARCH] q='{q[:30]}' | Parallel: {parallel_time:.1f}ms (Text: {text_time:.1f}ms, Vector: {vector_time:.1f}ms) | Merge: {merge_time:.1f}ms | Results: {len(results)}"
        )

        # Extract Redis-only times from both searches
        text_breakdown = text_results.get("latency_breakdown", {})
        text_search_ms = text_breakdown.get("text_search_ms", text_time)
        
        vector_breakdown = vector_results.get("latency_breakdown", {})
        redis_vector_ms = vector_breakdown.get("redis_vector_search_ms", 0)

        return {
            "results": results,
            "total": len(ranked_ids),
            "query": q,
            "search_type": "hybrid",
            "vector_weight": vector_weight,
            "latency_ms": total_time,
            "latency_breakdown": {
                "parallel_ms": round(parallel_time, 2),
                "text_search_ms": round(text_search_ms, 2),
                "redis_vector_search_ms": round(redis_vector_ms, 2),
            },
        }


    except Exception as e:
        print(f"Hybrid search error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "results": [], "total": 0}


@app.get("/api/similar/{doc_id}")
def find_similar(doc_id: str, limit: int = 5):
    """Find articles similar to a given article using vector similarity."""
    if not vector_search_enabled:
        return {"error": "Vector search not enabled", "similar": []}

    try:
        # Clean the doc_id
        clean_id = doc_id.strip("/")
        article_key = f"article:{clean_id}"

        print(f"Looking for article: {article_key}")  # Debug

        article_data = redis_service.client.execute_command(
            "JSON.GET", article_key, "$"
        )
        if not article_data:
            return {"error": f"Article not found: {doc_id}", "similar": []}

        import json

        # Handle both string and list responses
        if isinstance(article_data, bytes):
            article_data = article_data.decode("utf-8")
        if isinstance(article_data, str):
            parsed = json.loads(article_data)
            article = parsed[0] if isinstance(parsed, list) else parsed
        else:
            article = (
                article_data[0] if isinstance(article_data, list) else article_data
            )

        if "content_vector" not in article:
            return {"error": "Article has no vector embedding", "similar": []}

        vector = article["content_vector"]

        # Search for similar articles (excluding this one)
        import numpy as np

        result = redis_service.client.execute_command(
            "FT.SEARCH",
            redis_service.index_name,
            f"*=>[KNN {limit + 1} @content_vector $vec AS score]",
            "PARAMS",
            "2",
            "vec",
            np.array(vector, dtype=np.float32).tobytes(),
            "SORTBY",
            "score",
            "LIMIT",
            "0",
            str(limit + 1),
            "RETURN",
            "6",
            "$.title",
            "$.author",
            "$.category",
            "$.source",
            "$.published_at",
            "score",
            "DIALECT",
            "2",
        )

        # Parse results and exclude the source article
        similar = []
        for i in range(1, len(result), 2):
            found_id = (
                result[i].decode("utf-8") if isinstance(result[i], bytes) else result[i]
            )

            # Skip the source article itself
            if found_id == article_key:
                continue

            fields = result[i + 1]
            doc = {"id": found_id.replace("article:", "")}

            for j in range(0, len(fields), 2):
                key = (
                    fields[j].decode("utf-8")
                    if isinstance(fields[j], bytes)
                    else fields[j]
                )
                value = (
                    fields[j + 1].decode("utf-8")
                    if isinstance(fields[j + 1], bytes)
                    else fields[j + 1]
                )

                # Remove $. prefix from field names
                key = key.replace("$.", "") if key.startswith("$.") else key

                if key == "score":
                    doc["similarity_score"] = float(value)
                else:
                    doc[key] = value

            similar.append(doc)

            if len(similar) >= limit:
                break

        return {"article_id": doc_id, "similar": similar}

    except Exception as e:
        print(f"Find similar error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "similar": []}


@app.get("/api/search/capabilities")
def get_search_capabilities():
    """Get available search capabilities."""
    return {
        "text_search": True,
        "vector_search": vector_search_enabled,
        "hybrid_search": vector_search_enabled,
        "filters": ["category", "source"],
        "sorting": ["relevance", "date_desc", "date_asc"],
    }


# ============================================
# Serve Frontend
# ============================================

# Serve static files
frontend_path = Path(__file__).parent.parent / "frontend"


@app.get("/")
def serve_frontend():
    return FileResponse(frontend_path / "index.html")


@app.get("/{path:path}")
def serve_static(path: str):
    file_path = frontend_path / path
    if file_path.exists():
        return FileResponse(file_path)
    return FileResponse(frontend_path / "index.html")


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
