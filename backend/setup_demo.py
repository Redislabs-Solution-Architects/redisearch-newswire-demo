#!/usr/bin/env python3
"""
Complete NewsWire Vector Search Demo Setup
Single script to create index and load data with embeddings

Usage:
    python setup_demo.py              # Load 100 documents (default)
    python setup_demo.py --docs 1000  # Load 1000 documents
    python setup_demo.py --help       # Show help
"""

import sys
import time
import argparse
import pandas as pd
import redis
import json
from pathlib import Path
from redisvl.index import SearchIndex
from typing import Dict, Any

# Import our modules
from config import AMR_HOST, AMR_PORT, AMR_PASSWORD, EMBEDDING_DIMENSIONS
from embedder import NewsEmbedder

# ============================================
# CONFIGURATION
# ============================================

INDEX_NAME = "newswire_idx"
SCHEMA_PATH = "redisvl_schema.yaml"
DATA_FILE = "data/sample.parquet"
BATCH_SIZE = 50  # Documents per batch
EMBEDDING_BATCH_SIZE = 10  # Embeddings per API call
CONNECTION_TIMEOUT = 30
SOCKET_TIMEOUT = 30

# ============================================
# REDIS CONNECTION
# ============================================
SOCKET_TIMEOUT = 30

# Category cleanup mapping
CATEGORY_CLEANUP = {
    "big story 10": "Featured",
    "bonds news": "Business",
    "business news": "Business",
    "davos": "World",
    "financials": "Business",
    "funds news": "Business",
    "health": "Health",
    "identity": "Lifestyle",
    "noisey": "Entertainment",
    "politics": "Politics",
    "regulatory news - americas": "Business",
    "sports": "Sports",
    "tech by vice": "Technology",
    "technology news": "Technology",
    "u.s.": "Politics",
    "world news": "World",
    "none": "Other",
    "comics!": "Other",
    "games": "Other",
    "sex": "Other",
    "food by vice": "Lifestyle",
    "the vice guide to the 2016 election": "Politics",
}


def clean_category(raw_cat):
    """Clean and normalize category names"""
    if not raw_cat or pd.isna(raw_cat):
        return "Other"
    return CATEGORY_CLEANUP.get(str(raw_cat).lower(), str(raw_cat).title())


def create_redis_connection():
    """Create Redis connection"""
    try:
        r = redis.Redis(
            host=AMR_HOST,
            port=AMR_PORT,
            password=AMR_PASSWORD,
            ssl=True,
            decode_responses=False,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=CONNECTION_TIMEOUT,
            socket_keepalive=True,
            retry_on_timeout=True,
        )
        r.ping()
        return r
    except Exception as e:
        raise Exception(f"Failed to connect to Redis: {e}")


# ============================================
# STEP 1: VALIDATION
# ============================================


def validate_environment():
    """Validate all prerequisites"""
    print("=" * 70)
    print("🔍 STEP 1: VALIDATING ENVIRONMENT")
    print("=" * 70)
    print()

    # Check schema file
    print("📄 Checking schema file...")
    schema_file = Path(SCHEMA_PATH)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    print(f"✅ Found: {SCHEMA_PATH}")
    print()

    # Check data file
    print("📂 Checking data file...")
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    print(f"✅ Found: {DATA_FILE}")
    print()

    # Test Redis connection
    print("🔌 Testing Redis connection...")
    try:
        redis_client = create_redis_connection()
        print(f"✅ Connected to Redis at {AMR_HOST}:{AMR_PORT}")
    except Exception as e:
        raise Exception(f"Redis connection failed: {e}")
    print()

    # Test Azure OpenAI
    print("🧪 Testing Azure OpenAI connection...")
    embedder = NewsEmbedder()
    if not embedder.test_connection():
        raise Exception("Azure OpenAI connection failed")
    print()

    print("✅ All validations passed!")
    print()

    return redis_client, embedder, str(schema_file)


# ============================================
# STEP 2: CREATE INDEX
# ============================================


def create_index(redis_client, schema_path):
    """Create RediSearch index with vector field using raw commands"""
    print("=" * 70)
    print("🗃️  STEP 2: CREATING SEARCH INDEX")
    print("=" * 70)
    print()

    try:
        # Check if index exists and drop it
        print(f"🔍 Checking for existing index '{INDEX_NAME}'...")
        try:
            redis_client.execute_command("FT.INFO", INDEX_NAME)
            print(f"⚠️  Index '{INDEX_NAME}' exists, dropping...")
            redis_client.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
            print(f"✅ Dropped existing index")
        except Exception:
            print(f"✅ No existing index found")

        # ADD THIS: Drop suggestion dictionary if exists
        print(f"🔍 Clearing suggestion dictionary...")
        try:
            redis_client.delete("newswire_suggest")
            print(f"✅ Cleared existing suggestions")
        except Exception:
            print(f"✅ No existing suggestions found")

        print()
        print(f"🗃️  Creating new index '{INDEX_NAME}'...")

        # Create index with raw FT.CREATE command
        redis_client.execute_command(
            "FT.CREATE",
            INDEX_NAME,
            "ON",
            "JSON",
            "PREFIX",
            "1",
            "article:",
            "SCHEMA",
            "$.title",
            "AS",
            "title",
            "TEXT",
            "WEIGHT",
            "5.0",
            "$.content",
            "AS",
            "content",
            "TEXT",
            "WEIGHT",
            "1.0",
            "$.summary",
            "AS",
            "summary",
            "TEXT",
            "WEIGHT",
            "2.0",
            "$.author",
            "AS",
            "author",
            "TEXT",
            "WEIGHT",
            "1.0",
            "$.published_at",
            "AS",
            "published_at",
            "TEXT",
            "SORTABLE",
            "$.category",
            "AS",
            "category",
            "TAG",
            "SEPARATOR",
            ",",
            "$.source",
            "AS",
            "source",
            "TAG",
            "SEPARATOR",
            ",",
            "$.tags[*]",
            "AS",
            "tags",
            "TAG",
            "SEPARATOR",
            ",",
            "$.published_ts",
            "AS",
            "published_ts",
            "NUMERIC",
            "SORTABLE",
            "$.word_count",
            "AS",
            "word_count",
            "NUMERIC",
            "SORTABLE",
            "$.content_vector",
            "AS",
            "content_vector",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(EMBEDDING_DIMENSIONS),
            "DISTANCE_METRIC",
            "COSINE",
        )

        print("✅ Index created successfully!")
        print()
        print("📋 Index Schema:")
        print("   TEXT fields: title, content, summary, author, published_at")
        print("   TAG fields: category, source, tags")
        print("   NUMERIC fields: published_ts, word_count")
        print(f"   VECTOR field: content_vector ({EMBEDDING_DIMENSIONS} dims, HNSW)")
        print()

        # Wait for index to be ready
        import time

        time.sleep(2)

        # Verify
        info = redis_client.execute_command("FT.INFO", INDEX_NAME)
        print("✅ Index verification successful!")
        print()

        return True

    except Exception as e:
        raise Exception(f"Failed to create index: {e}")


# ============================================
# STEP 3: LOAD DATA WITH VECTORS
# ============================================


def transform_row(row: pd.Series, embedding: list) -> Dict[str, Any]:
    """Transform parquet row to Redis JSON document with vector"""
    import datetime

    # Parse date
    date_obj = pd.to_datetime(row.get("date"))

    doc = {
        "id": f"nw_{row.get('idx', 0):07d}",
        "title": str(row.get("title", "")),
        "content": str(row.get("article", "")),
        "summary": "",  # Not in parquet
        "author": str(row.get("author", "")),
        "category": clean_category(row.get("section", "")),  # CLEANED!
        "tags": [],
        "source": str(row.get("publication", "")),
        "published_at": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
        "published_ts": int(date_obj.timestamp()),
        "word_count": len(str(row.get("article", "")).split()),
        "content_vector": embedding,
    }
    return doc


def load_data_with_vectors(redis_client, embedder, target_docs):
    """Load data from parquet with vector embeddings"""
    print("=" * 70)
    print("📦 STEP 3: LOADING DATA WITH VECTORS")
    print("=" * 70)
    print()

    # Read parquet
    print("📖 Reading parquet file...")
    try:
        df = pd.read_parquet(DATA_FILE)
        print(f"✅ Loaded {len(df):,} rows from parquet")

        # Limit to target docs
        if len(df) > target_docs:
            df = df.head(target_docs)
            print(f"⚠️  Limited to first {target_docs:,} documents")
    except Exception as e:
        raise Exception(f"Failed to read parquet: {e}")

    print()
    total_rows = len(df)
    loaded_count = 0
    skipped_count = 0
    start_time = time.time()

    print(f"🔄 Processing {total_rows:,} documents in batches of {BATCH_SIZE}...")
    print()

    # Process in batches
    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[batch_start:batch_end]

        print(f"📦 Batch {batch_start + 1}-{batch_end}/{total_rows}")

        # Prepare texts for embedding
        texts = []
        for idx, row in batch_df.iterrows():
            text = embedder.prepare_text(
                row.get("title", ""), row.get("summary", ""), row.get("content", "")
            )
            texts.append(text)

        # Generate embeddings
        print(f"   🔄 Generating {len(texts)} embeddings...")
        embeddings = embedder.embed_batch(
            texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress=False
        )

        # Load documents
        print(f"   💾 Loading to Redis...")
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            try:
                embedding = embeddings[i]

                if embedding is None:
                    print(f"   ⚠️  Skipping {row['id']} - embedding failed")
                    skipped_count += 1
                    continue

                # Transform to document
                doc = transform_row(row, embedding)

                # Create Redis key
                redis_key = f"article:{doc['id']}"

                # Store as JSON
                redis_client.execute_command(
                    "JSON.SET", redis_key, "$", json.dumps(doc)
                )

                loaded_count += 1
                # Create curated autocomplete suggestions
                try:
                    title = doc.get("title", "").strip()
                    category = doc.get("category", "").strip()
                    content = doc.get("content", "").strip()

                    if title and len(title) > 3:
                        # 1. Add full article title (score = 1)
                        redis_client.execute_command(
                            "FT.SUGADD", "newswire_suggest", title, "1"
                        )

                    # 2. Add category as searchable term (score = 5, higher priority)
                    if category and category.lower() not in ["other", "none"]:
                        try:
                            redis_client.execute_command(
                                "FT.SUGADD", "newswire_suggest", category, "5"
                            )
                        except:
                            pass

                    # 3. Extract and add key terms (topics, entities)
                    # Combine title + first 200 chars of content for context
                    text_sample = f"{title} {content[:200]}"

                    # Common stop words to exclude
                    stop_words = {
                        "the",
                        "and",
                        "for",
                        "are",
                        "but",
                        "not",
                        "you",
                        "all",
                        "can",
                        "her",
                        "was",
                        "one",
                        "our",
                        "out",
                        "day",
                        "get",
                        "has",
                        "him",
                        "his",
                        "how",
                        "its",
                        "may",
                        "new",
                        "now",
                        "old",
                        "see",
                        "two",
                        "who",
                        "boy",
                        "did",
                        "let",
                        "put",
                        "say",
                        "she",
                        "too",
                        "use",
                        "with",
                        "this",
                        "that",
                        "from",
                        "have",
                        "been",
                        "will",
                        "what",
                        "when",
                        "make",
                        "like",
                        "time",
                        "just",
                        "know",
                        "take",
                        "into",
                        "year",
                        "your",
                        "some",
                        "could",
                        "them",
                        "than",
                        "then",
                        "look",
                        "only",
                        "come",
                        "over",
                        "also",
                        "back",
                        "after",
                        "work",
                        "first",
                        "well",
                        "even",
                        "want",
                        "because",
                        "these",
                        "give",
                        "most",
                    }

                    # Extract significant words (4+ chars, not stop words)
                    words = text_sample.lower().split()
                    word_freq = {}

                    for word in words:
                        # Clean word (remove punctuation)
                        clean_word = "".join(c for c in word if c.isalnum())

                        if (
                            len(clean_word) >= 4
                            and clean_word not in stop_words
                            and not clean_word.isdigit()
                        ):
                            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

                    # Add top 5 most frequent words (score = 2)
                    top_words = sorted(
                        word_freq.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    for word, freq in top_words:
                        try:
                            # Score = 2 + frequency bonus
                            score = min(2 + freq, 10)  # Cap at 10
                            redis_client.execute_command(
                                "FT.SUGADD", "newswire_suggest", word, str(score)
                            )
                        except:
                            pass

                except Exception as e:
                    # Don't fail on suggestion errors
                    pass

            except Exception as e:
                print(f"   ❌ Error loading {row.get('id', 'unknown')}: {e}")
                skipped_count += 1

        # Progress
        elapsed = time.time() - start_time
        rate = loaded_count / elapsed if elapsed > 0 else 0
        print(
            f"   ✅ Progress: {loaded_count:,}/{total_rows} | Rate: {rate:.1f} docs/sec"
        )
        print()

    # Summary
    elapsed_time = time.time() - start_time

    print("=" * 70)
    print("📊 LOADING SUMMARY")
    print("=" * 70)
    print(f"Total rows:       {total_rows:,}")
    print(f"Loaded:           {loaded_count:,}")
    print(f"Skipped:          {skipped_count:,}")
    print(f"Time:             {elapsed_time:.1f} seconds")
    print(f"Average rate:     {loaded_count/elapsed_time:.1f} docs/sec")
    print()

    first_id = f"nw_{df.iloc[0]['idx']:07d}"
    last_id = f"nw_{df.iloc[-1]['idx']:07d}"
    return loaded_count, first_id, last_id


# ============================================
# STEP 4: VERIFICATION
# ============================================


def verify_setup(redis_client, loaded_count, first_id, last_id):
    """Verify the setup is working"""
    print("=" * 70)
    print("✅ STEP 4: VERIFICATION")
    print("=" * 70)
    print()

    if loaded_count == 0:
        print("❌ No documents were loaded!")
        return False

    print(f"✅ Successfully loaded {loaded_count:,} documents")
    print(f"📝 Document ID range: {first_id} to {last_id}")
    print()

    # Verify sample document
    sample_key = f"article:{first_id}"
    try:
        sample_doc = json.loads(
            redis_client.execute_command("JSON.GET", sample_key, "$")
        )
        if sample_doc and len(sample_doc) > 0:
            doc_data = sample_doc[0]
            if "content_vector" in doc_data:
                vector_len = len(doc_data["content_vector"])
                print(
                    f"✅ Vector verification: Sample document has {vector_len}-dim vector"
                )
                print(
                    f"   Sample vector: [{doc_data['content_vector'][0]:.4f}, {doc_data['content_vector'][1]:.4f}, ...]"
                )
            else:
                print("⚠️  Warning: Sample document missing vector field")
                return False
    except Exception as e:
        print(f"⚠️  Could not verify vector: {e}")
        return False

    print()
    print("=" * 70)
    print("🎉 DEMO SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run: python main.py")
    print("  2. Open: http://localhost:8000")
    print("  3. Try vector search queries!")
    print()

    return True


# ============================================
# MAIN ENTRY POINT
# ============================================


def main():
    """Main setup function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Setup NewsWire Vector Search Demo")
    parser.add_argument(
        "--docs",
        type=int,
        default=100,
        help="Number of documents to load (default: 100)",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("🚀 NEWSWIRE VECTOR SEARCH DEMO SETUP")
    print("=" * 70)
    print()
    print(f"Target documents: {args.docs:,}")
    print()

    try:
        # Step 1: Validate
        redis_client, embedder, schema_path = validate_environment()

        # Step 2: Create Index
        index = create_index(redis_client, schema_path)

        # Step 3: Load Data with Vectors
        loaded_count, first_id, last_id = load_data_with_vectors(
            redis_client, embedder, args.docs
        )

        # Step 4: Verify
        success = verify_setup(redis_client, loaded_count, first_id, last_id)

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
