#!/usr/bin/env python3
"""
Create RediSearch index for NewsWire demo
Drops existing index if present and creates a new one with proper schema
"""

import redis
from redis.commands.search.field import TextField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import sys

# Import config
from config import AMR_HOST, AMR_PORT, AMR_PASSWORD

# ============================================
# CONFIGURATION
# ============================================

INDEX_NAME = "newswire_idx"
CONNECTION_TIMEOUT = 30
SOCKET_TIMEOUT = 30

# ============================================
# REDIS CONNECTION
# ============================================

def create_redis_connection():
    """Create Redis connection"""
    try:
        r = redis.Redis(
            host=AMR_HOST,
            port=AMR_PORT,
            password=AMR_PASSWORD,
            ssl=True,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=CONNECTION_TIMEOUT,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        r.ping()
        print(f"✅ Connected to Redis at {AMR_HOST}:{AMR_PORT}")
        return r
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print()
        print("Please check:")
        print("  1. .env file exists with correct credentials")
        print("  2. AMR_HOST, AMR_PORT, AMR_PASSWORD are set")
        print("  3. Redis instance is running and accessible")
        print("  4. RediSearch module is enabled on your Redis instance")
        sys.exit(1)

# ============================================
# INDEX CREATION
# ============================================

def create_index():
    """Create RediSearch index with proper schema"""
    
    print("=" * 70)
    print("🔧 REDISEARCH INDEX CREATOR")
    print("=" * 70)
    print()
    
    # Connect to Redis
    print("🔌 Connecting to Redis...")
    redis_client = create_redis_connection()
    print()
    
    # Check if index exists and drop it
    print(f"🔍 Checking for existing index '{INDEX_NAME}'...")
    try:
        # Try to get index info
        redis_client.execute_command("FT.INFO", INDEX_NAME)
        print(f"⚠️  Index '{INDEX_NAME}' already exists")
        print(f"🗑️  Dropping existing index...")
        redis_client.execute_command("FT.DROPINDEX", INDEX_NAME)
        print(f"✅ Dropped existing index")
    except redis.exceptions.ResponseError as e:
        if "Unknown index name" in str(e):
            print(f"✅ No existing index found (this is fine)")
        else:
            print(f"⚠️  Warning: {e}")
    
    print()
    print(f"🏗️  Creating new index '{INDEX_NAME}'...")
    print()
    
    # Define schema
    schema = (
        TextField("$.title", as_name="title", weight=5.0, sortable=False),
        TextField("$.content", as_name="content", weight=1.0, sortable=False),
        TextField("$.summary", as_name="summary", weight=2.0, sortable=False),
        TextField("$.author", as_name="author", weight=1.0, sortable=False),
        TagField("$.category", as_name="category", separator=","),
        TagField("$.source", as_name="source", separator=","),
        TagField("$.tags[*]", as_name="tags", separator=","),
        TextField("$.published_at", as_name="published_at", sortable=True),
        NumericField("$.published_ts", as_name="published_ts", sortable=True),
        NumericField("$.word_count", as_name="word_count", sortable=True),
    )
    
    # Create index definition
    definition = IndexDefinition(
        prefix=["article:"],
        index_type=IndexType.JSON
    )
    
    try:
        # Create the index
        redis_client.ft(INDEX_NAME).create_index(
            schema,
            definition=definition
        )
        
        print("✅ Index created successfully!")
        print()
        print("📋 Index Schema:")
        print("   TEXT fields:")
        print("     • title (weight: 5.0)")
        print("     • content (weight: 1.0)")
        print("     • summary (weight: 2.0)")
        print("     • author (weight: 1.0)")
        print("     • published_at (sortable)")
        print()
        print("   TAG fields:")
        print("     • category")
        print("     • source")
        print("     • tags")
        print()
        print("   NUMERIC fields:")
        print("     • published_ts (sortable)")
        print("     • word_count (sortable)")
        print()
        
        # Verify index was created
        info = redis_client.execute_command("FT.INFO", INDEX_NAME)
        print("✅ Index verification successful!")
        print()
        
        # Display some index info
        info_dict = dict(zip(info[::2], info[1::2]))
        print(f"📊 Index Details:")
        print(f"   • Index name: {INDEX_NAME}")
        print(f"   • Index type: JSON")
        print(f"   • Key prefix: article:")
        print(f"   • Number of fields: {info_dict.get('num_docs', 0)}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        print()
        print("Common issues:")
        print("  1. RediSearch module not enabled on Redis instance")
        print("  2. Insufficient permissions")
        print("  3. Invalid schema definition")
        sys.exit(1)
    
    print("=" * 70)
    print("🎉 INDEX CREATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run: python load_sample_data.py")
    print("  2. Run: python main.py")
    print("  3. Open: http://localhost:8000")
    print()

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        create_index()
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