from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
import os
from pathlib import Path

# Import your existing redis service
from services.redis_search import RedisSearchService

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

    return response


@app.get("/api/categories")
def get_categories():
    """Get all categories with counts."""
    try:
        # Get unique category values using FT.TAGVALS
        tag_values = redis_service.client.execute_command(
            "FT.TAGVALS", redis_service.index_name, "category"
        )

        categories = []
        for cat in tag_values:
            if isinstance(cat, bytes):
                cat = cat.decode()

            # Get count for this category
            count_result = redis_service.client.execute_command(
                "FT.SEARCH",
                redis_service.index_name,
                f"@category:{{{cat}}}",
                "LIMIT",
                "0",
                "0",
            )
            count = count_result[0] if count_result else 0
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
def autocomplete(q: str, limit: int = 5):
    """Get autocomplete suggestions: matching titles + spell suggestions."""
    try:
        # Get matching article titles
        results = redis_service.search(
            query=q,
            category=None,
            source=None,
            author=None,
            use_fuzzy=False,
            sort_by="relevance",
            offset=0,
            limit=limit,
            highlight=False,
        )

        articles = []
        for article in results.get("results", []):
            articles.append(
                {
                    "title": article.get("title", ""),
                    "category": article.get("category", ""),
                    "source": article.get("source", ""),
                    "date": article.get("published_at", ""),
                    "id": article.get("id", ""),
                }
            )

        # Get spell suggestions
        spell_suggestions = redis_service.spell_check(q)
        suggestion_text = None
        if spell_suggestions:
            corrected = q
            for orig, sugg in spell_suggestions.items():
                corrected = corrected.replace(orig, sugg)
            if corrected != q:
                suggestion_text = corrected

        return {"articles": articles, "spell_suggestion": suggestion_text}
    except Exception as e:
        return {"articles": [], "spell_suggestion": None, "error": str(e)}


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