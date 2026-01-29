import redis
import time
import socket
from config import AMR_HOST, AMR_PORT, AMR_PASSWORD


class RedisSearchService:
    def __init__(self):
        from redis.connection import ConnectionPool, SSLConnection

        # Create connection pool with aggressive keepalive
        self.pool = ConnectionPool(
            connection_class=SSLConnection,
            host=AMR_HOST,
            port=int(AMR_PORT),
            password=AMR_PASSWORD,
            decode_responses=True,
            max_connections=100,
            socket_keepalive=True,
            socket_keepalive_options={
                socket.TCP_KEEPIDLE: 10,
                socket.TCP_KEEPINTVL: 5,
                socket.TCP_KEEPCNT: 3
            },
            health_check_interval=5,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_read_size=65536,
            client_name="fastapi-vector-search"
        )

        self.client = redis.Redis(connection_pool=self.pool)
        self.index_name = "newswire_idx"

    def get_categories(self):
        """Fetch unique categories from AMR."""
        try:
            result = self.client.execute_command(
                "FT.AGGREGATE",
                self.index_name,
                "*",
                "GROUPBY",
                "1",
                "@category",
                "REDUCE",
                "COUNT",
                "0",
                "AS",
                "count",
                "SORTBY",
                "2",
                "@count",
                "DESC",
                "LIMIT",
                "0",
                "50",
            )
            categories = ["All"]
            for item in result[1:]:
                if isinstance(item, list) and len(item) >= 2:
                    categories.append(item[1])
            return categories
        except Exception as e:
            print(f"Error fetching categories: {e}")
            return ["All"]

    def get_sources(self):
        """Fetch unique sources from AMR."""
        try:
            result = self.client.execute_command(
                "FT.AGGREGATE",
                self.index_name,
                "*",
                "GROUPBY",
                "1",
                "@source",
                "REDUCE",
                "COUNT",
                "0",
                "AS",
                "count",
                "SORTBY",
                "2",
                "@count",
                "DESC",
                "LIMIT",
                "0",
                "50",
            )
            sources = ["All"]
            for item in result[1:]:
                if isinstance(item, list) and len(item) >= 2:
                    sources.append(item[1])
            return sources
        except Exception as e:
            print(f"Error fetching sources: {e}")
            return ["All"]

    def get_authors(self):
        """Fetch top authors from AMR."""
        try:
            result = self.client.execute_command(
                "FT.AGGREGATE",
                self.index_name,
                "*",
                "GROUPBY",
                "1",
                "@author",
                "REDUCE",
                "COUNT",
                "0",
                "AS",
                "count",
                "SORTBY",
                "2",
                "@count",
                "DESC",
                "LIMIT",
                "0",
                "50",
            )
            authors = ["All"]
            for item in result[1:]:
                if isinstance(item, list) and len(item) >= 2:
                    author = item[1]
                    if author and author != "Unknown":
                        authors.append(author)
            return authors
        except Exception as e:
            print(f"Error fetching authors: {e}")
            return ["All"]

    def get_index_stats(self):
        """Get index statistics for display."""
        try:
            info = self.client.execute_command("FT.INFO", self.index_name)
            info_dict = dict(zip(info[::2], info[1::2]))
            return {
                "num_docs": info_dict.get("num_docs", 0),
                "num_terms": info_dict.get("num_terms", 0),
                "index_size_mb": info_dict.get("inverted_sz_mb", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_category_counts(self, query="", source=None, author=None):
        """Get article counts by category for current search context."""
        try:
            query_parts = []

            if query and query.strip() and len(query.strip()) >= 2:
                query_parts.append(query.strip())

            if source and source != "All":
                escaped = source.replace("-", "\\-").replace(" ", "\\ ")
                query_parts.append(f"@source:{{{escaped}}}")

            if author and author != "All":
                escaped = author.replace("-", "\\-").replace(" ", "\\ ")
                query_parts.append(f"@author:{{{escaped}}}")

            search_query = " ".join(query_parts) if query_parts else "*"

            result = self.client.execute_command(
                "FT.AGGREGATE",
                self.index_name,
                search_query,
                "GROUPBY",
                "1",
                "@category",
                "REDUCE",
                "COUNT",
                "0",
                "AS",
                "count",
                "SORTBY",
                "2",
                "@count",
                "DESC",
                "LIMIT",
                "0",
                "15",
            )

            categories = []
            for item in result[1:]:
                if isinstance(item, list) and len(item) >= 4:
                    categories.append({"name": item[1], "count": int(item[3])})

            return categories
        except Exception as e:
            print(f"Error fetching category counts: {e}")
            return []

    def spell_check(self, query):
        """Get spelling suggestions for query."""
        if not query or len(query.strip()) < 2:
            return None

        try:
            result = self.client.execute_command(
                "FT.SPELLCHECK", self.index_name, query
            )

            suggestions = {}
            for term_result in result:
                if isinstance(term_result, list) and len(term_result) >= 3:
                    original_term = term_result[1]
                    term_suggestions = term_result[2]
                    if term_suggestions:
                        best = term_suggestions[0]
                        if isinstance(best, list) and len(best) >= 2:
                            suggestions[original_term] = best[1]

            return suggestions if suggestions else None
        except Exception as e:
            print(f"Spell check error: {e}")
            return None

    def get_document(self, doc_id):
        """Get full document by ID."""
        try:
            doc = self.client.json().get(doc_id)
            return doc
        except Exception as e:
            print(f"Error fetching document: {e}")
            return None

    def _escape_tag_value(self, value):
        """Escape special characters in TAG field values."""
        if not value:
            return value
        # Escape all special characters for TAG fields
        value = value.replace("\\", "\\\\")  # Escape backslash first
        value = value.replace(",", "\\,")
        value = value.replace(".", "\\.")
        value = value.replace("<", "\\<")
        value = value.replace(">", "\\>")
        value = value.replace("{", "\\{")
        value = value.replace("}", "\\}")
        value = value.replace("[", "\\[")
        value = value.replace("]", "\\]")
        value = value.replace('"', '\\"')
        value = value.replace("'", "\\'")
        value = value.replace(":", "\\:")
        value = value.replace(";", "\\;")
        value = value.replace("!", "\\!")
        value = value.replace("@", "\\@")
        value = value.replace("#", "\\#")
        value = value.replace("$", "\\$")
        value = value.replace("%", "\\%")
        value = value.replace("^", "\\^")
        value = value.replace("&", "\\&")
        value = value.replace("*", "\\*")
        value = value.replace("(", "\\(")
        value = value.replace(")", "\\)")
        value = value.replace("-", "\\-")
        value = value.replace("+", "\\+")
        value = value.replace("=", "\\=")
        value = value.replace("~", "\\~")
        value = value.replace(" ", "\\ ")
        return value

    def _build_tag_filter(self, field_name, values):
        """
        Build a TAG filter for single value or list of values.

        Args:
            field_name: The field name (e.g., "category", "source")
            values: Single string or list of strings

        Returns:
            Query string like "@category:{value}" or "(@source:{a} | @source:{b})"
        """
        if not values:
            return None

        # Handle single value (string)
        if isinstance(values, str):
            if values == "All":
                return None
            escaped = self._escape_tag_value(values)
            return f"@{field_name}:{{{escaped}}}"

        # Handle list of values
        if isinstance(values, list):
            if len(values) == 0:
                return None
            if len(values) == 1:
                escaped = self._escape_tag_value(values[0])
                return f"@{field_name}:{{{escaped}}}"

            # Multiple values - use OR with pipe inside the tag braces
            escaped_values = [self._escape_tag_value(v) for v in values]
            filters = [
                f"(@{field_name}:{{{val}}})" for val in escaped_values
            ]  # Note: added () around each
            return " | ".join(filters)

        return None

    def search(
        self,
        query="",
        category=None,
        source=None,
        author=None,
        use_fuzzy=False,
        sort_by="relevance",
        offset=0,
        limit=10,
        highlight=True,
    ):
        """
        Execute search and return results with timing.

        Args:
            query: Search query text
            category: Single category string or list of categories
            source: Single source string or list of sources
            author: Single author string or list of authors
            use_fuzzy: Enable fuzzy matching
            sort_by: "relevance", "date_desc", or "date_asc"
            offset: Pagination offset
            limit: Number of results
            highlight: Enable result highlighting
        """
        start = time.perf_counter()

        query_parts = []
        filter_parts = []

        # Handle text query
        if query and query.strip():
            if use_fuzzy:
                words = query.strip().split()
                fuzzy_terms = [f"%%{word}%%" for word in words if word]
                query_parts.append(f"@title|summary:({' '.join(fuzzy_terms)})")
            else:
                query_parts.append(f"@title|summary:({query.strip()})")


        # Handle category filter (string or list)
        category_filter = self._build_tag_filter("category", category)
        if category_filter:
            filter_parts.append(category_filter)

        # Handle source filter (string or list)
        source_filter = self._build_tag_filter("source", source)
        if source_filter:
            filter_parts.append(source_filter)

        # Handle author filter (string or list)
        author_filter = self._build_tag_filter("author", author)
        if author_filter:
            filter_parts.append(author_filter)

        # Build final query with proper grouping
        if query_parts and filter_parts:
            # Text query + Filters: (query) AND (filters)
            text_part = f"({' '.join(query_parts)})"
            
            # Wrap multiple filters in parentheses for correct precedence
            if len(filter_parts) > 1:
                filter_part = f"({' '.join(filter_parts)})"
            else:
                filter_part = filter_parts[0]
            
            search_query = f"{text_part} {filter_part}"
        elif filter_parts:
            # Only filters
            search_query = " ".join(filter_parts)
        elif query_parts:
            # Only text query
            search_query = " ".join(query_parts)
        else:
            # No query or filters
            search_query = "*"

        cmd_args = [self.index_name, search_query]

        if highlight and query and query.strip():
            cmd_args.extend(
                [
                    "RETURN",
                    "8",
                    "title",
                    "summary",
                    "content",
                    "author",
                    "category",
                    "published_at",
                    "source",
                    "word_count",
                    "HIGHLIGHT",
                    "FIELDS",
                    "1",
                    "title",
                    "TAGS",
                    "<mark>",
                    "</mark>",
                    "SUMMARIZE",
                    "FIELDS",
                    "1",
                    "summary",
                    "FRAGS",
                    "1",
                    "LEN",
                    "50",
                ]
            )
        else:
            cmd_args.extend(
                [
                    "RETURN",
                    "8",
                    "title",
                    "summary",
                    "content",
                    "author",
                    "category",
                    "published_at",
                    "source",
                    "word_count",
                ]
            )

        if sort_by == "date_desc":
            cmd_args.extend(["SORTBY", "published_ts", "DESC"])
        elif sort_by == "date_asc":
            cmd_args.extend(["SORTBY", "published_ts", "ASC"])

        cmd_args.extend(["LIMIT", offset, limit])

        try:
            raw_results = self.client.execute_command("FT.SEARCH", *cmd_args)
            latency_ms = (time.perf_counter() - start) * 1000

            results = self._parse_results(raw_results)

            command = f'FT.SEARCH {self.index_name} "{search_query}"'
            if sort_by != "relevance":
                sort_field = {
                    "date_desc": "published_ts DESC",
                    "date_asc": "published_ts ASC",
                }.get(sort_by, "")
                command += f" SORTBY {sort_field}"
            command += f" LIMIT {offset} {limit}"

            return {
                "results": results,
                "latency_ms": latency_ms,
                "command": command,
                "total": raw_results[0] if raw_results else 0,
                "offset": offset,
                "limit": limit,
            }

        except Exception as e:
            print(f"Search error: {e}")
            return {
                "results": [],
                "latency_ms": 0,
                "command": f'FT.SEARCH {self.index_name} "{search_query}" LIMIT {offset} {limit}',
                "error": str(e),
                "total": 0,
                "offset": offset,
                "limit": limit,
            }

    def autocomplete(
        self, prefix, limit=5, fuzzy=False
    ):  # Changed fuzzy default to False
        """
        Get autocomplete suggestions using RediSearch's native FT.SUGGET.
        Much faster than full-text search for prefix matching.
        """
        if not prefix or len(prefix) < 2:
            return []

        try:
            # Don't use FUZZY - it doesn't work well for autocomplete
            # Use prefix matching only
            cmd = ["FT.SUGGET", "newswire_suggest", prefix, "MAX", str(limit)]

            suggestions = self.client.execute_command(*cmd)

            # Handle None or empty response
            if not suggestions:
                return []

            # Decode bytes if necessary
            return [s.decode() if isinstance(s, bytes) else s for s in suggestions]
        except Exception as e:
            print(f"Autocomplete error: {e}")
            return []

    def _parse_results(self, raw):
        """Parse FT.SEARCH response into list of dicts."""
        if not raw or raw[0] == 0:
            return []

        results = []
        i = 1
        while i < len(raw):
            doc_key = raw[i]
            fields = raw[i + 1] if i + 1 < len(raw) else []

            doc = {"id": doc_key}
            for j in range(0, len(fields), 2):
                if j + 1 < len(fields):
                    doc[fields[j]] = fields[j + 1]

            results.append(doc)
            i += 2

        return results
