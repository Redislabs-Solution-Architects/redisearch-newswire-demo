#!/usr/bin/env python3
"""
Simplified data loader for demo purposes
Loads a single parquet file into Redis for testing
"""

import pandas as pd
import redis
from redis.exceptions import ConnectionError, TimeoutError, ResponseError
from pathlib import Path
from datetime import datetime
import sys
import time
from collections import Counter

# Import config
from config import AMR_HOST, AMR_PORT, AMR_PASSWORD

# ============================================
# CONFIGURATION
# ============================================

PARQUET_FILE = Path(__file__).parent / "data" / "sample.parquet"
INDEX_NAME = "newswire_idx"
TARGET_DOCS = 10_000  # Configurable limit for demo

# Redis connection settings
CONNECTION_TIMEOUT = 30
SOCKET_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# ============================================
# CATEGORY & SOURCE MAPPINGS
# ============================================

SECTION_TO_CATEGORY = {
    'politics': 'Politics', 'Politics': 'Politics', 'opinions': 'Politics', 'opinion': 'Politics',
    'Supreme Court': 'Politics', 'Policy': 'Politics', 'immigration': 'Politics',
    'election 2020': 'Politics', '2020 elections': 'Politics', 'Elections': 'Politics',
    'Super Tuesday': 'Politics', 'Brexit': 'Politics', 'Vote Now': 'Politics',
    'nyregion': 'Politics', 'upshot': 'Politics',
    
    'Business News': 'Business', 'Market News': 'Business', 'Financials': 'Business',
    'Bonds News': 'Business', 'Consumer Goods and Retail': 'Business', 'Commodities': 'Business',
    'Company News': 'Business', 'Deals': 'Business', 'Energy': 'Business', 'business': 'Business',
    'Funds News': 'Business', 'Cyclical Consumer Goods': 'Business', 'Basic Materials': 'Business',
    'Industrials': 'Business', 'Hot Stocks': 'Business', 'Credit RSS': 'Business',
    'Wealth': 'Business', 'Markets': 'Business', 'Money': 'Business', 'investing': 'Business',
    'Retail': 'Business', 'Economy': 'Business', 'Finance': 'Business', 'Business': 'Business',
    'finance-and-economics': 'Business', 'economy': 'Business', 'Earnings': 'Business',
    
    'World News': 'World', 'us': 'World', 'asia': 'World', 'middleeast': 'World',
    'africa': 'World', 'Asia': 'World', 'Japan': 'World', 'americas': 'World',
    'europe': 'World', 'China Economy': 'World', 'china': 'World', 'britain': 'World',
    'middle-east-and-africa': 'World', 'india': 'World', 'australia': 'World', 'uk': 'World',
    'World Politics': 'World', 'Asia Politics': 'World', 'Europe Politics': 'World',
    'Europe News': 'World', 'World Economy': 'World', 'Asia Economy': 'World',
    'Europe Economy': 'World', 'world': 'World', 'U.S.': 'World', 'international': 'World',
    
    'Tech by VICE': 'Technology', 'Technology News': 'Technology', 'Tech': 'Technology',
    'tech': 'Technology', 'Communications Equipment': 'Technology', 'Software': 'Technology',
    'Semiconductors': 'Technology', 'Computer Hardware': 'Technology', 'Mobile': 'Technology',
    'Apps': 'Technology', 'Internet News': 'Technology', 'Cybersecurity': 'Technology',
    'Internet': 'Technology', 'Social Media': 'Technology', 'technology': 'Technology',
    'science': 'Technology', 'Science': 'Technology', 'Science News': 'Technology',
    'science-and-technology': 'Technology', 'Technology': 'Technology',
    
    'entertainment': 'Entertainment', 'Entertainment': 'Entertainment', 'Music by VICE': 'Entertainment',
    'Noisey': 'Entertainment', 'Entertainment News': 'Entertainment', 'arts': 'Entertainment',
    'music': 'Entertainment', 'movies': 'Entertainment', 'tv': 'Entertainment',
    'theater': 'Entertainment', 'celebrities': 'Entertainment', 'tv-shows': 'Entertainment',
    'Television': 'Entertainment', 'Music News': 'Entertainment', 'Film News': 'Entertainment',
    'culture': 'Entertainment', 'Television News': 'Entertainment', 'Arts': 'Entertainment',
    'books': 'Entertainment', 'books-and-arts': 'Entertainment', 'celebrity': 'Entertainment',
    
    'Sports News': 'Sports', 'Sports': 'Sports', 'US College Basketball': 'Sports',
    'US NBA': 'Sports', 'US NHL': 'Sports', 'US MLB': 'Sports', 'US NFL': 'Sports',
    'Olympics News': 'Sports', 'sport': 'Sports', 'sports': 'Sports',
    'US College Football': 'Sports', 'Olympics Rio': 'Sports', 'RIO 2016': 'Sports',
    
    'Healthcare': 'Health', 'health': 'Health', 'Health News': 'Health', 'Health': 'Health',
    'Health and Science': 'Health', 'Drugs': 'Health', 'Health ': 'Health',
    'healthcare': 'Health', 'Health Insurance': 'Health', 'Hospitals': 'Health',
    'Cancer': 'Health', 'Biotechnology': 'Health', 'Biotech and Pharma': 'Health',
    'Pharmaceuticals': 'Health', 'well': 'Health',
    
    'Food by VICE ': 'Lifestyle', 'Identity': 'Lifestyle', 'Travel': 'Lifestyle',
    'Lifestyle': 'Lifestyle', 'dining': 'Lifestyle', 'fashion': 'Lifestyle',
    'realestate': 'Lifestyle', 'travel': 'Lifestyle', 'Food & Beverage': 'Lifestyle',
    'Restaurants': 'Lifestyle', 'Food': 'Lifestyle', 'parenting': 'Lifestyle',
    'parents': 'Lifestyle', 'living': 'Lifestyle', 'style': 'Lifestyle',
    'pets': 'Lifestyle', 'royals': 'Lifestyle', 't-magazine': 'Lifestyle',
    
    'Editorial': 'Opinion', 'Commentary': 'Opinion', 'Voices': 'Opinion',
    'perspectives': 'Opinion', 'ideas': 'Opinion', 'Comment': 'Opinion',
    'letters': 'Opinion', 'Opinion': 'Opinion', 'leaders': 'Opinion',
}

PUBLICATION_TO_SOURCE = {
    'Reuters': 'Reuters', 'CNN': 'CNN', 'TMZ': 'TMZ', 'Vice': 'VICE', 'Vice News': 'VICE News',
    'Washington Post': 'The Washington Post', 'Mashable': 'Mashable', 'Vox': 'Vox',
    'CNBC': 'CNBC', 'Business Insider': 'Business Insider', 'The New York Times': 'The New York Times',
    'Refinery 29': 'Refinery29', 'The Hill': 'The Hill', 'Hyperallergic': 'Hyperallergic',
    'The Verge': 'The Verge', 'TechCrunch': 'TechCrunch', 'Politico': 'Politico',
    'Economist': 'The Economist', 'Buzzfeed News': 'BuzzFeed News', 'Wired': 'Wired',
    'People': 'People', 'New Republic': 'New Republic', 'Gizmodo': 'Gizmodo'
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def map_section_to_category(section):
    """Map section to one of 10 categories"""
    if pd.isna(section) or not section or str(section).strip() == 'None':
        return 'General'
    return SECTION_TO_CATEGORY.get(str(section).strip(), 'General')

def normalize_source(publication):
    """Normalize publication name"""
    if pd.isna(publication) or not publication:
        return 'Unknown'
    return PUBLICATION_TO_SOURCE.get(str(publication).strip(), str(publication).strip())

def generate_summary(content, max_chars=250):
    """Generate summary from content"""
    if not content or pd.isna(content) or str(content).strip() == 'None':
        return "No summary available."
    content_str = str(content).strip()
    if len(content_str) <= max_chars:
        return content_str
    return content_str[:max_chars].rsplit(' ', 1)[0] + "..."

def transform_row(row, doc_id):
    """Transform parquet row to our schema"""
    date_str = str(row['date'])
    try:
        published_at = date_str.split(' ')[0] if ' ' in date_str else date_str
        dt = datetime.strptime(published_at, '%Y-%m-%d')
        published_ts = int(dt.timestamp())
    except:
        published_at = date_str
        published_ts = 0
    
    content = str(row['article']) if not pd.isna(row['article']) and str(row['article']) != 'None' else ''
    word_count = len(content.split()) if content else 0
    author = str(row['author']) if not pd.isna(row['author']) and str(row['author']) != 'None' else 'Unknown'
    category = map_section_to_category(row['section'])
    section_str = str(row['section']) if not pd.isna(row['section']) and str(row['section']) != 'None' else None
    tags = [section_str] if section_str else []
    
    return {
        "id": f"nw_{doc_id:07d}",
        "title": str(row['title']).strip(),
        "content": content,
        "summary": generate_summary(content),
        "author": author,
        "category": category,
        "tags": tags,
        "source": normalize_source(row['publication']),
        "published_at": published_at,
        "published_ts": published_ts,
        "word_count": word_count
    }

def is_valid_row(row):
    """Check if row should be loaded"""
    if pd.isna(row['title']) or not str(row['title']).strip():
        return False, "empty_title"
    article = str(row['article']) if not pd.isna(row['article']) else ''
    if not article or article == 'None' or len(article.strip()) == 0:
        return False, "empty_article"
    if len(article.split()) < 10:
        return False, "too_short"
    return True, None

# ============================================
# REDIS CONNECTION
# ============================================

def create_redis_connection():
    """Create Redis connection with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
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
                health_check_interval=30,
                retry_on_timeout=True
            )
            r.ping()
            print(f"✅ Connected to Redis at {AMR_HOST}:{AMR_PORT}")
            return r
        except (ConnectionError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                print(f"⚠️  Connection attempt {attempt} failed: {e}")
                print(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"❌ Failed to connect to Redis after {MAX_RETRIES} attempts")

def load_document_with_retry(redis_client, doc_id, document):
    """Load a document to Redis with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            redis_client.json().set(f"article:{doc_id}", "$", document)
            return True
        except (ConnectionError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                print(f"    ⚠️  Failed to load {doc_id} (attempt {attempt}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ❌ Failed to load {doc_id} after {MAX_RETRIES} attempts")
                return False
        except ResponseError as e:
            print(f"    ❌ Redis error loading {doc_id}: {e}")
            return False
        except Exception as e:
            print(f"    ❌ Unexpected error loading {doc_id}: {e}")
            return False
    return False

# ============================================
# MAIN LOADING LOGIC
# ============================================

def load_sample_data():
    """Load sample parquet file into Redis"""
    
    print("=" * 70)
    print("🚀 SAMPLE DATA LOADER FOR NEWSWIRE DEMO")
    print("=" * 70)
    print()
    
    # Check if parquet file exists
    if not PARQUET_FILE.exists():
        print(f"❌ Error: Parquet file not found at {PARQUET_FILE}")
        print()
        print("Please ensure you have:")
        print("  1. Created the 'data' folder: backend/data/")
        print("  2. Downloaded the sample parquet file")
        print("  3. Placed it as: backend/data/sample.parquet")
        print()
        sys.exit(1)
    
    print(f"📂 Loading from: {PARQUET_FILE}")
    print(f"🎯 Target documents: {TARGET_DOCS:,}")
    print()
    
    # Connect to Redis
    print("🔌 Connecting to Redis...")
    try:
        redis_client = create_redis_connection()
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print()
        print("Please check:")
        print("  1. .env file exists with correct credentials")
        print("  2. AMR_HOST, AMR_PORT, AMR_PASSWORD are set")
        print("  3. Redis instance is running and accessible")
        sys.exit(1)
    
    print()
    
    # Read parquet file
    print("📖 Reading parquet file...")
    try:
        df = pd.read_parquet(PARQUET_FILE)
        print(f"✅ Loaded {len(df):,} rows from parquet")
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        sys.exit(1)
    
    print()
    print("🔄 Processing and loading documents...")
    print()
    
    # Process rows
    stats = {
        "total_rows": len(df),
        "loaded": 0,
        "skipped": 0,
        "skip_reasons": Counter()
    }
    
    seen_keys = set()
    current_id = 0
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        # Check if we've reached target
        if stats["loaded"] >= TARGET_DOCS:
            print(f"\n🎯 Target of {TARGET_DOCS:,} documents reached!")
            break
        
        # Validate row
        is_valid, skip_reason = is_valid_row(row)
        if not is_valid:
            stats["skipped"] += 1
            stats["skip_reasons"][skip_reason] += 1
            continue
        
        # Check for duplicates
        dup_key = f"{row['title']}_{row['date']}"
        if dup_key in seen_keys:
            stats["skipped"] += 1
            stats["skip_reasons"]["duplicate"] += 1
            continue
        seen_keys.add(dup_key)
        
        # Transform and load
        current_id += 1
        doc = transform_row(row, current_id)
        
        if load_document_with_retry(redis_client, doc["id"], doc):
            stats["loaded"] += 1
            
            # Progress indicator
            if stats["loaded"] % 500 == 0:
                elapsed = time.time() - start_time
                rate = stats["loaded"] / elapsed
                print(f"   ✓ Loaded: {stats['loaded']:,} docs ({rate:.0f} docs/sec)")
        else:
            stats["skipped"] += 1
            stats["skip_reasons"]["redis_error"] += 1
    
    elapsed = time.time() - start_time
    
    # Final summary
    print()
    print("=" * 70)
    print("📊 LOADING SUMMARY")
    print("=" * 70)
    print(f"Total rows processed:  {stats['total_rows']:,}")
    print(f"Documents loaded:      {stats['loaded']:,}")
    print(f"Documents skipped:     {stats['skipped']:,}")
    print(f"Time taken:            {elapsed:.1f} seconds")
    print(f"Average rate:          {stats['loaded']/elapsed:.0f} docs/sec")
    print()
    
    if stats["skip_reasons"]:
        print("Skip reasons:")
        for reason, count in stats["skip_reasons"].most_common():
            print(f"  • {reason}: {count:,}")
        print()
    
    print(f"✅ Successfully loaded {stats['loaded']:,} documents to Redis!")
    print(f"📝 Document ID range: nw_0000001 to nw_{current_id:07d}")
    print()
    print("🎉 Ready to start the demo!")
    print("   Run: python main.py")
    print("=" * 70)

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        load_sample_data()
    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)