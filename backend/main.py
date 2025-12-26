from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
import os
from pathlib import Path

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
    embedder = NewsEmbedder()
    vector_search_enabled = True
    print("✅ Vector search enabled")
except Exception as e:
    embedder = None
    vector_search_enabled = False
    print(f"⚠️  Vector search disabled: {e}")

# ============================================
# API Endpoints
# ============================================


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
    has_query = q and len(q.strip()) >= 2

    # Handle multiple categories (comma-separated) - use lowercase for Redis
    categories = None
    if category and category != "All":
        cat_list = [c.strip() for c in category.split(",") if c.strip()]  # Remove .lower()
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

    # Strip "article:" prefix from result IDs for frontend
    if "results" in response:
        for result in response["results"]:
            if "id" in result and result["id"].startswith("article:"):
                result["id"] = result["id"].replace("article:", "")
            
            # Fetch content for preview if not present (text search doesn't return it)
            if "content" not in result or not result.get("content"):
                try:
                    doc_id = f"article:{result['id']}"
                    doc = redis_service.get_document(doc_id)
                    if doc and "content" in doc:
                        result["content"] = doc["content"][:300]  # First 300 chars for preview
                except Exception as e:
                    print(f"Warning: Could not fetch content for {result['id']}: {e}")
                    result["content"] = ""

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
                sugg = sugg.decode('utf-8')
            
            if sugg and sugg.lower() not in seen:
                # Determine suggestion type
                sugg_type = "article" if len(sugg) > 40 else "topic"
                
                suggestions.append({
                    "title": sugg,
                    "type": sugg_type
                })
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
            "query": q
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
    if not vector_search_enabled:
        return {"error": "Vector search not enabled", "results": [], "total": 0}

    if not q or len(q.strip()) < 2:
        return {"error": "Query too short", "results": [], "total": 0}

    try:
        # Generate query embedding
        query_embedding = embedder.embed_single(q.strip())
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

        # Parse results
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

        return {"results": results, "total": total, "query": q, "search_type": "vector"}

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
    if not vector_search_enabled:
        return {"error": "Vector search not enabled", "results": [], "total": 0}

    if not q or len(q.strip()) < 2:
        return {"error": "Query too short", "results": [], "total": 0}

    try:
        # Get text search results
        text_results = search(
            q=q, category=category, source=source, offset=0, limit=50, fuzzy=False
        )

        # Get vector search results
        vector_results = vector_search(
            q=q, category=category, source=source, offset=0, limit=50
        )

        # Merge using Reciprocal Rank Fusion (RRF)
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

        return {
            "results": results,
            "total": len(ranked_ids),
            "query": q,
            "search_type": "hybrid",
            "vector_weight": vector_weight,
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
