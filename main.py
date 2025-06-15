# enhanced_main.py - Updated with KB management UI and user greeting
import streamlit as st
import pandas as pd
import psycopg2
import requests
import fitz
import io
import re
import time
import numpy as np
from datetime import datetime

# Import our enhanced models
from models import (
    setup_enhanced_models, 
    query_enhanced_semantic_search, 
    generate_enhanced_answer,
    enhanced_models,
    get_model_info
)


#Database configuration
DB_CONFIG = {
    "database": "your_database_name",
    "user": "postgres",
    "password": "your_password",
    "host": "127.0.0.1",
    "port": 5432
}

MINDSDB_URL = "http://127.0.0.1:47334"

def sanitize_table_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

def get_all_knowledge_bases():
    """Get all knowledge bases from PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Get all tables that are not system tables
        cur.execute("""
            SELECT table_name, 
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size,
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = t.table_name AND column_name = 'content') as has_content
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            AND table_name NOT LIKE 'pg_%'
            AND table_name NOT LIKE 'sql_%'
            ORDER BY table_name;
        """)
        
        tables = cur.fetchall()
        knowledge_bases = []
        
        for table_name, size, has_content in tables:
            if has_content > 0:  # Only include tables with content column (our KB tables)
                # Get record count and creation info
                cur.execute(f'SELECT COUNT(*), MIN(id), MAX(id) FROM "{table_name}";')
                count_result = cur.fetchone()
                record_count = count_result[0] if count_result else 0
                
                # Get sample roles
                cur.execute(f'SELECT DISTINCT role FROM "{table_name}" LIMIT 5;')
                roles = [row[0] for row in cur.fetchall()]
                
                knowledge_bases.append({
                    'name': table_name,
                    'size': size,
                    'records': record_count,
                    'roles': roles,
                    'kb_name': f"kb_{table_name}"
                })
        
        conn.close()
        return knowledge_bases
        
    except Exception as e:
        st.error(f"Error fetching knowledge bases: {e}")
        return []

def delete_knowledge_base(table_name, kb_name):
    """Delete both PostgreSQL table and MindsDB KB"""
    try:
        # Delete PostgreSQL table
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        conn.commit()
        conn.close()
        
        # Delete MindsDB KB
        try:
            drop_sql = f"DROP KNOWLEDGE_BASE IF EXISTS {kb_name};"
            response = requests.post(f"{MINDSDB_URL}/api/sql/query", json={"query": drop_sql})
        except:
            pass  # MindsDB KB might not exist
        
        return True
    except Exception as e:
        st.error(f"Error deleting knowledge base: {e}")
        return False

def get_kb_preview(table_name, limit=3):
    """Get preview of KB content"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f'SELECT content, role FROM "{table_name}" LIMIT %s;', (limit,))
        results = cur.fetchall()
        conn.close()
        return [{"content": row[0][:200] + "..." if len(row[0]) > 200 else row[0], "role": row[1]} for row in results]
    except:
        return []

def create_postgres_datasource(base_url, db_config):
    """Create PostgreSQL datasource in MindsDB (auto-creation)"""
    try:
        response = requests.get(f"{base_url}/api/databases")
        if response.status_code == 200:
            databases = response.json()
            if any(db.get('name') == 'my_postgres_ds' for db in databases):
                return True

        datasource_payload = {
            "name": "my_postgres_ds",
            "engine": "postgres",
            "connection_data": {
                "host": db_config["host"],
                "port": db_config["port"],
                "user": db_config["user"],
                "password": db_config["password"],
                "database": db_config["database"]
            }
        }

        response = requests.post(f"{base_url}/api/databases", json=datasource_payload)
        return response.status_code in [200, 201]
    except Exception as e:
        st.error(f"Datasource error: {e}")
        return False

def create_enhanced_kb(base_url, kb_name, table_name):
    """Create KB with enhanced embeddings"""
    try:
        # Drop existing KB
        drop_sql = f"DROP KNOWLEDGE_BASE IF EXISTS {kb_name};"
        requests.post(f"{base_url}/api/sql/query", json={"query": drop_sql})
        
        # Create KB with better embeddings
        kb_sql = f"""
        CREATE KNOWLEDGE_BASE {kb_name}
        FROM my_postgres_ds (SELECT id, content, role FROM "{table_name}")
        USING
            embedding_model = {{
                "provider": "huggingface",
                "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
            }},
            content_columns = ['content'],
            metadata_columns = ['role'],
            id_column = 'id';
        """
        
        response = requests.post(f"{base_url}/api/sql/query", json={"query": kb_sql})
        
        if response.status_code in [200, 201]:
            result = response.json()
            return 'error' not in result
        return False
            
    except Exception as e:
        st.error(f"KB creation error: {e}")
        return False

def store_data_in_postgres(df, table_name):
    """Store DataFrame in PostgreSQL (auto-creation)"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        cur.execute(f'''
            CREATE TABLE "{table_name}" (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        for _, row in df.iterrows():
            cur.execute(
                f'INSERT INTO "{table_name}" (content, role) VALUES (%s, %s);',
                (row['content'], row['role'])
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

def get_data_from_postgres(table_name):
    """Get all data from PostgreSQL table"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f'SELECT content, role FROM "{table_name}";')
        results = cur.fetchall()
        conn.close()
        return [{"content": row[0], "role": row[1]} for row in results]
    except:
        return []

def enhanced_semantic_query(table_name, question, role_filter=None):
    """Query using enhanced local models"""
    # Get data from PostgreSQL
    data = get_data_from_postgres(table_name)
    
    if not data:
        return []
    
    # Filter by role if specified
    if role_filter and role_filter != "general":
        data = [item for item in data if item["role"] == role_filter]
    
    # Extract texts for semantic search
    texts = [item["content"] for item in data]
    
    # Perform enhanced semantic search
    results = query_enhanced_semantic_search(question, texts, top_k=5)
    
    return results

# Streamlit UI Configuration
st.set_page_config(
    page_title="🧠 Enhanced Local KB", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# User Greeting
current_time = datetime.now().hour
if current_time < 12:
    greeting = "Good Morning! 🌅"
elif current_time < 17:
    greeting = "Good Afternoon! ☀️"
else:
    greeting = "Good Evening! 🌙"

st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>🚀 Enhanced Local Knowledge Base</h1>
        <h3 style='color: white; margin: 5px 0;'>{greeting} Welcome to your AI-powered KB!</h3>
        <p style='color: white; margin: 0; opacity: 0.9;'>Better QA Models • Enhanced Semantic Search • 340MB Total</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for KB Management
st.sidebar.markdown("## 📚 Knowledge Base Management")

# Get all knowledge bases
knowledge_bases = get_all_knowledge_bases()

if knowledge_bases:
    st.sidebar.success(f"📊 Found {len(knowledge_bases)} Knowledge Base(s)")
    
    # KB Selection dropdown
    kb_options = ["Select a Knowledge Base..."] + [kb['name'] for kb in knowledge_bases]
    selected_kb = st.sidebar.selectbox("Choose KB to work with:", kb_options)
    
    # Show KB details if one is selected
    if selected_kb != "Select a Knowledge Base...":
        selected_kb_info = next((kb for kb in knowledge_bases if kb['name'] == selected_kb), None)
        if selected_kb_info:
            st.sidebar.markdown("### 📋 KB Details")
            st.sidebar.info(f"""
            **Name:** {selected_kb_info['name']}  
            **Records:** {selected_kb_info['records']}  
            **Size:** {selected_kb_info['size']}  
            **Roles:** {', '.join(selected_kb_info['roles'][:3])}
            """)
            
            # Preview button
            if st.sidebar.button("👀 Preview Content"):
                st.session_state.show_preview = selected_kb
            
            # Delete button with confirmation
            st.sidebar.markdown("---")
            if st.sidebar.button("🗑️ Delete This KB", type="secondary"):
                st.session_state.confirm_delete = selected_kb
            
            # Confirmation dialog
            if st.session_state.get('confirm_delete') == selected_kb:
                st.sidebar.warning("⚠️ Are you sure?")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✅ Yes, Delete"):
                        if delete_knowledge_base(selected_kb, f"kb_{selected_kb}"):
                            st.sidebar.success("✅ KB Deleted!")
                            st.session_state.confirm_delete = None
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Delete Failed")
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_delete = None
                        st.rerun()
else:
    st.sidebar.info("📝 No Knowledge Bases found. Create one below!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Model status and initialization
    st.markdown("### 🔧 Model Management")
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        if st.button("🚀 Initialize Enhanced Models", type="primary"):
            with st.spinner("Loading enhanced models (QA-optimized, ~340MB)..."):
                if setup_enhanced_models():
                    st.success("✅ Enhanced models ready!")
                    # Show model info
                    info = get_model_info()
                    st.info(f"📊 Models: {info['total_size']}")
                else:
                    st.error("❌ Failed to load models")
    
    with model_col2:
        # Enhanced model status
        if enhanced_models.embedding_model and enhanced_models.qa_model:
            st.success("🟢 Enhanced Models Ready")
            st.caption("QA-optimized embeddings + BERT QA")
        elif enhanced_models.embedding_model:
            st.warning("🟡 Embeddings Only")
        else:
            st.info("🔵 Models Not Loaded")

with col2:
    # Connection status
    st.markdown("### 🔗 Connection Status")
    try:
        response = requests.get(f"{MINDSDB_URL}/api/status", timeout=5)
        if response.status_code == 200:
            st.success("🟢 MindsDB Connected")
        else:
            st.error("🔴 MindsDB Connection Issue")
    except:
        st.error("🔴 MindsDB Disconnected")

# Show KB Preview if requested
if st.session_state.get('show_preview'):
    st.markdown("---")
    st.markdown(f"### 👀 Preview: {st.session_state.show_preview}")
    preview_data = get_kb_preview(st.session_state.show_preview)
    if preview_data:
        for i, item in enumerate(preview_data, 1):
            with st.expander(f"Record {i} - Role: {item['role']}"):
                st.text_area("Content:", item['content'], height=100, key=f"preview_{i}")
    else:
        st.info("No preview data available")
    
    if st.button("Close Preview"):
        st.session_state.show_preview = None
        st.rerun()

st.markdown("---")

# File upload section
st.markdown("### 📁 Create New Knowledge Base")
with st.expander("Upload CSV or PDF", expanded=not knowledge_bases):
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "pdf"])
    selected_role = st.text_input("Metadata role:", value="general")

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    content_sentences = []

    # Process file
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        for _, row in df.iterrows():
            content_sentences.append(", ".join([f"{col}: {val}" for col, val in zip(df.columns, row)]))

    elif file_type == "pdf":
        pdf_text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()
        content_sentences = [line.strip() for line in pdf_text.split("\n") 
                           if line.strip() and len(line.strip()) > 10]
        st.text_area("PDF Preview", pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text, height=200)

    table_name = sanitize_table_name(uploaded_file.name.split(".")[0])
    kb_name = f"kb_{table_name}"

    # Auto-create everything
    if st.button("🚀 Auto-Create Enhanced Knowledge Base", type="primary"):
        if not content_sentences:
            st.error("No content found")
        else:
            with st.spinner("Creating enhanced knowledge base..."):
                # Step 1: Store in PostgreSQL
                df_to_store = pd.DataFrame({
                    "content": content_sentences,
                    "role": [selected_role] * len(content_sentences)
                })
                
                if store_data_in_postgres(df_to_store, table_name):
                    st.success("✅ Data stored in PostgreSQL")
                    
                    # Step 2: Create datasource (auto)
                    if create_postgres_datasource(MINDSDB_URL, DB_CONFIG):
                        st.success("✅ PostgreSQL datasource auto-created")
                        
                        # Step 3: Create enhanced KB (auto)
                        if create_enhanced_kb(MINDSDB_URL, kb_name, table_name):
                            st.success("✅ Enhanced Knowledge Base created with QA-optimized embeddings")
                        else:
                            st.warning("⚠️ MindsDB KB creation failed, using local search only")
                    
                    st.success("🎉 Enhanced Knowledge Base ready for queries!")
                    st.balloons()
                    
                    # Refresh the page to show new KB
                    time.sleep(2)
                    st.rerun()

# Enhanced Query section
st.markdown("---")
st.markdown("### 💬 Ask Questions (Enhanced QA)")

# Only show query section if there are KBs or one is selected
if knowledge_bases:
    # Question input with examples
    question_examples = [
        "What is the best pizza mentioned?",
        "Tell me about the highest rated item",
        "What are the key features discussed?",
        "Summarize the main points"
    ]
    
    # KB selection for querying
    if len(knowledge_bases) > 1:
        query_kb_options = [kb['name'] for kb in knowledge_bases]
        selected_query_kb = st.selectbox("Select KB to query:", query_kb_options, key="query_kb_select")
    else:
        selected_query_kb = knowledge_bases[0]['name']
        st.info(f"Querying KB: **{selected_query_kb}**")
    
    question = st.text_input("Your question:", placeholder="Type your question here...")

    if st.button("🔍 Enhanced Search & Answer", type="primary") and question:
        with st.spinner("Searching with enhanced QA models..."):
            
            # Use enhanced semantic search
            if enhanced_models.embedding_model:
                results = enhanced_semantic_query(selected_query_kb, question, selected_role)
                
                if results:
                    # Get top contexts
                    contexts = [r['text'] for r in results[:3]]
                    combined_context = "\n\n".join(contexts)
                    
                    # Generate enhanced answer
                    answer = generate_enhanced_answer(combined_context, question)
                    
                    st.markdown("### 🎯 Enhanced QA Answer")
                    st.success(answer)
                    
                    # Show quality metrics
                    avg_score = np.mean([r['score'] for r in results])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Search Quality", f"{avg_score:.3f}")
                    with col2:
                        st.metric("Sources Found", len(results))
                    with col3:
                        st.metric("KB Used", selected_query_kb)
                    
                    # Show sources with better formatting
                    with st.expander("📖 Sources & Relevance Scores"):
                        for i, result in enumerate(results, 1):
                            confidence = "🟢 High" if result['score'] > 0.7 else "🟡 Medium" if result['score'] > 0.4 else "🔴 Low"
                            st.markdown(f"**Source {i}** - Score: {result['score']:.3f} ({confidence})")
                            st.text_area("", result['text'], height=80, key=f"enhanced_source_{i}")
                else:
                    st.warning("No relevant content found. Try rephrasing your question.")
                    
                    # Suggest improvements
                    st.info("""
                    **Tips for better results:**
                    - Use specific keywords from your document
                    - Ask direct questions (What, Where, How, etc.)
                    - Try shorter, focused questions
                    """)
            else:
                st.error("Enhanced models not loaded. Please initialize them first.")
else:
    st.info("👆 Create a Knowledge Base first to start asking questions!")

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    # Model information
    with st.expander("🔧 Enhanced Model Details"):
        if enhanced_models.embedding_model or enhanced_models.qa_model:
            info = get_model_info()
            st.markdown(f"""
    **Enhanced Model Architecture:**

    - **Embedding Model**: {info['embedding_model']}
    - **QA Model**: {info['qa_model']}
    - **Total Size**: {info['total_size']}

    **Key Improvements:**
    - ✅ QA-optimized embeddings (better question understanding)
    - ✅ BERT-based question answering (vs basic text generation)
    - ✅ Better context extraction and answer synthesis
    - ✅ Improved relevance scoring
    - ✅ Fallback mechanisms for robustness
            """)
        else:
            st.info("Initialize models to see detailed information")

with col2:
    # Performance tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
    **Getting Better Answers:**

    1. **Question Types That Work Best:**
       - "What is the rating of [item]?"
       - "Which [item] has the highest [metric]?"
       - "Tell me about [specific topic]"
       - "How does [item] compare?"

    2. **Search Quality:**
       - Scores > 0.7: Highly relevant
       - Scores 0.4-0.7: Moderately relevant  
       - Scores < 0.4: Less relevant
        """)

with col3:
    # Usage statistics
    with st.expander("📊 Usage Statistics"):
        total_kbs = len(knowledge_bases)
        total_records = sum(kb['records'] for kb in knowledge_bases)
        st.metric("Total KBs", total_kbs)
        st.metric("Total Records", total_records)
        if enhanced_models.embedding_model:
            st.success("🟢 Models Loaded")
        else:
            st.info("🔵 Models Not Loaded")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p>Enhanced Local KB • QA-Optimized • 340MB Total • No API Keys Required</p>
    <p>Made with ❤️ for efficient knowledge management</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = None
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = None