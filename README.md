# RediSearch NewsWire Demo
<img width="1502" height="742" alt="image" src="https://github.com/user-attachments/assets/58567b35-f134-427c-bba6-a28c9f0da774" />

**⚠️ Important:**
This demo uses relative API paths (/api) in the frontend so the UI and backend can be deployed together on the same host (VM, container, etc.).

**Latency Expectations (Important)**
| Deployment                                     | Typical Latency |
| ---------------------------------------------- | --------------- |
| **Backend on Azure VM (same region as Redis)** | **~1.5–2 ms**   |
| **Backend on local laptop → Azure Redis**      | **~35–70 ms**   |

**Why this happens**

When running on an Azure VM in the same region as Redis:

- Traffic stays inside Azure’s backbone network
- Very low RTT and consistent performance

When running from a local machine:
- Requests traverse the public internet
- TLS + WAN latency dominates total response time

✅ Best practice:
For realistic performance testing and demos, run the backend in the same region as your Redis instance.

## Prerequisites

### 1. Azure Managed Redis Instance

You need to create an Azure Managed Redis instance with the following specifications:

- **SKU:** Balanced B0/B1 
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
git clone https://github.com/yourusername/redisearch-newswire-demo.git
cd redisearch-newswire-demo
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
AMR_HOST=your-redis-instance.redis.cache.windows.net
AMR_PORT=10000
AMR_PASSWORD=your-redis-password-here
```

> 🔑 **Where to find credentials:**
> 1. Go to Azure Portal
> 2. Navigate to your Redis Cache instance
> 3. Go to "Access keys" section
> 4. Copy the hostname, port, and primary access key

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create the RediSearch index

```bash
cd backend
python create_index.py
```

This creates the `newswire_idx` index with the proper schema for searching news articles.

### 6. Load sample data

```bash
python load_sample_data.py
```

This loads documents from your parquet file into Redis. The small sample takes ~30 seconds.

### 7. Start API server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
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
redis-cli FT.DROPINDEX newswire_idx DD
```
