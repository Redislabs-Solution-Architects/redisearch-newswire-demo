# redisearch-newswire-demo
A news search demo powered by Azure Managed Redis (RediSearch) with full-text search, filtering, and autocomplete

Setup Instructions

1. Create virtual environment and activate

Mac/Linux:

bash

python3 -m venv venv
source venv/bin/activate

Windows:

bash

python -m venv venv
venv\Scripts\activate

2. Update .env file in backend folder

Navigate to the backend folder and create/update the .env file with your Azure Managed Redis credentials:

bash

cd backend

Create a .env file with the following content:

env

AMR_HOST=your-redis-instance.redis.cache.windows.net
AMR_PORT=10000
AMR_PASSWORD=your-redis-password-here

Replace the values with your actual Azure Managed Redis instance details.

3. Install dependencies

bash

pip install -r requirements.txt

4. Create the RediSearch index

bash

python create_index.py

This will create the newswire_idx index with the proper schema for searching news articles.

5. Load sample data

bash

python load_sample_data.py

This will load 10,000 documents from your parquet file into Redis. The process takes approximately 2-3 minutes.

6. Run the demo

bash

python main.py

The API server will start on http://localhost:8000

7. Open your browser

Navigate to:

http://localhost:8000

