import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Existing Redis config
AMR_HOST = os.getenv("AMR_HOST")
AMR_PORT = int(os.getenv("AMR_PORT", "10000"))
AMR_PASSWORD = os.getenv("AMR_PASSWORD")

# NEW - Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Validate critical config
if not all([AMR_HOST, AMR_PORT, AMR_PASSWORD]):
    raise ValueError("Redis configuration incomplete. Check .env file")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY]):
    raise ValueError("Azure OpenAI configuration incomplete. Check .env file")
