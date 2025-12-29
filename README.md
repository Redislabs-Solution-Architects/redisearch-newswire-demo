This demo uses relative API paths (/api) in the frontend so the UI and backend can be deployed together on the same host (VM, container, etc.).

✅ Best practice:
For realistic performance testing and demos, run the backend in the same region as your Redis instance.

## Prerequisites

### 1. Azure Managed Redis Instance

You need to create an Azure Managed Redis instance with the following specifications:

- **SKU:** Balanced B1 
- **Features Required:** RediSearch, RedisJson module enabled
- **Region:** Any (recommend same region as your location for better latency)

> 💡 **Tip:** For testing with 100 documents, B0/B1 provides comfortable headroom and good performance.

### 2. Python Environment

- Python 3.8 or higher
- pip package manager

### 3. Sample Data

- The repository includes a small sample parquet file (`sample.parquet`) with 100 documents
- You may substitute your own parquet file for larger datasets (10k+)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone -b branch-1 --single-branch https://github.com/Redislabs-Solution-Architects/redisearch-newswire-demo
```

### 2. Set up virtual environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Create environment file

Copy the template and add your credentials:
```bash
cp .env.template .env
```

Edit `.env` with your actual Azure Managed Redis credentials:
```env
AMR_HOST=your-redis-instance.westus2.redis.azure.net
AMR_PORT=10000
AMR_PASSWORD=your-redis-password-here
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-openai-instance-endpoint
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=your-openai-deployment
AZURE_OPENAI_API_VERSION=your-azure-version
EMBEDDING_DIMENSIONS=1536
```

> 🔑 **Where to find credentials:**
> 1. Go to Azure Portal
> 2. Navigate to your Redis Cache instance
> 3. Go to "Access keys" section
> 4. Copy the hostname, port, and primary access key

### 4. Install dependencies

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev -y
python3.11 --version
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Load sample data & Create the RediSearch index

```bash
python backend/setup_demo.py --docs 100

```
This loads documents from your parquet file into Redis. The small sample takes ~30 seconds.
This creates the `newswire_idx` index with the proper schema for searching news articles.

### 7. Run & start API server
```bash
python backend/main.py
```

### 8. Open in browser

On the VM:
```bash
http://127.0.0.1:8000
```

From your laptop:
```bash
http://<VM_PUBLIC_IP>:8000
```

Make sure port 8000 is allowed in your Azure VM network rules.


### 🧹 Cleanup

To delete all indexed data:
```bash
FLUSHDB
```
