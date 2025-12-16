import os
from dotenv import load_dotenv

load_dotenv()

# Azure Managed Redis
AMR_HOST = os.getenv("AMR_HOST", "your-redis.redis.cache.windows.net")
AMR_PORT = int(os.getenv("AMR_PORT", 6380))
AMR_PASSWORD = os.getenv("AMR_PASSWORD", "")

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://your-search.search.windows.net")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "newswire")

# Data path
DATA_PATH = os.path.expanduser("~/Documents/redis/comparison/rediSearch/newswire_100k.jsonl")