# DocMind_KB

# 🧠 Lightweight Local Knowledge Base

A **100% local knowledge base** that lets you upload CSV or PDF files and ask questions in natural language. Your documents are processed into searchable text chunks, stored in PostgreSQL, and indexed by MindsDB for semantic search. All AI models run locally—no API keys, no cloud, just fast, private answers on your machine.

## Demo 
https://github.com/user-attachments/assets/d3af4fd6-63b3-4b18-b8bc-5db5661f74ac


## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Adii0906/DocMind_KB.git
cd DocMind_KB
```

### 2. Install Docker
Download and install Docker from: **https://www.docker.com/products/docker-desktop/**

### 3. Pull & Run MindsDB
```bash
# Pull MindsDB image
docker pull mindsdb/mindsdb

# Run MindsDB container
docker run -p 47334:47334 -p 5436:5432 mindsdb/mindsdb
```
**MindsDB GUI will be available at**: `http://127.0.0.1:47334/`

### 4. Install PostgreSQL
Download and install PostgreSQL from: **https://www.postgresql.org/download/**
- Ensure it's running on port 5432
- Remember your username/password for setup

### 5. Connect PostgreSQL to MindsDB
1. Go to MindsDB GUI (`http://127.0.0.1:47334/`)
2. On the right side → **Learning Hub** → Search **"postgres"**
3. Run this SQL command:

```sql
CREATE DATABASE postgresql_conn 
WITH ENGINE = 'postgres', 
PARAMETERS = {
    "host": "host.docker.internal",  -- Use this for Docker
    "port": 5432,
    "database": "your_database",       -- Your database name created on postgres
    "user": "postgres",
    "schema": "public",
    "password": "password_you_setuped_during_installation"               -- Your password
};
```

### 6. Set Up Python Environment

#### a. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### b. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 7. Run the Application
```bash
streamlit run main.py
```

## 🔍 How It Works

**Simple Explanation:**
1. **Upload Documents** → Your files (CSV/PDF) get converted to text chunks
2. **Store in Database** → Text chunks saved in PostgreSQL (your local database)
3. **Create AI Index** → MindsDB reads your data and creates "smart embeddings" (mathematical understanding of meaning)
4. **Ask Questions** → When you ask something, the system finds relevant chunks based on meaning (not just keywords)
5. **Generate Answers** → Local AI model combines relevant information to create natural language answers

**Why This Works:**
- **PostgreSQL** = Your filing cabinet (stores the actual documents)
- **MindsDB** = Your smart librarian (understands what documents mean and finds relevant ones)
- **Local AI** = Your assistant (reads relevant documents and answers your questions)

### Simple Architecture
```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Your      │───▶│  PostgreSQL  │◄──▶│    MindsDB      │
│  Documents  │    │  (Raw Data)  │    │  (AI Layer)     │
└─────────────┘    └──────────────┘    └─────────────────┘
       │                                         │
       ▼                                         ▼
┌─────────────┐                        ┌─────────────────┐
│   Local     │                        │   Semantic      │
│   Models    │◄──────────────────────▶│   Search &      │
│ (Embeddings)  │                        │   Answers       │
└─────────────┘                        └─────────────────┘
```

**Data Flow:**
1. **Upload** → Files processed into text chunks
2. **Store** → Text saved in PostgreSQL 
3. **Index** → MindsDB creates embeddings for semantic search
4. **Query** → Ask questions, get intelligent answers

## 🤖 AI Models (340MB Total)
- **Embeddings**: `multi-qa-MiniLM-L6-cos-v1` (QA-optimized, 80MB) – Finds relevant information for your questions
- **Question Answering**: `distilbert-base-cased-distilled-squad` (260MB) – Extracts precise answers from your documents
- **Lightweight & Fast** – Runs on CPU, no GPU required

## 📋 Usage Steps

1. **Initialize Models** → Click "🔧 Initialize Lightweight Models"
2. **Upload File** → Drop CSV or PDF file
3. **Auto-Create KB** → Click "🚀 Auto-Create Knowledge Base" 
4. **Ask Questions** → Type natural language questions
5. **Get Answers** → Receive contextual responses with sources

## ⚙️ Configuration

Update database settings in `main.py`:
```python
DB_CONFIG = {
    "database": "database_name",    # Your database name
    "user": "postgres",
    "password": "your_pass",           # Your password  
    "host": "127.0.0.1",
    "port": 5432
}
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| MindsDB not connecting | Check docker is running: `docker ps` |
| PostgreSQL connection failed | Verify PostgreSQL is running on port 5432 |
| Models not loading | Check internet for first download (~360MB) |
| GUI not accessible | Ensure port 47334 is available |

## ✨ Features
- 🔒 **100% Local** - No external APIs
- 📁 **Multi-format** - CSV & PDF support  
- 🧠 **Smart Search** - Semantic understanding
- 🚀 **Auto-setup** - One-click knowledge base creation
- 💬 **Natural Q&A** - Ask questions in plain English

---
**Tech Stack**: Streamlit • PostgreSQL • MindsDB • Transformers • Docker • Python
