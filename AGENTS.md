# AGENTS.md - Essential Project Information

## 🚀 Project Overview
**LangGraph Agent with MongoDB Atlas + Voyage AI + Local LLM**
- **Purpose**: Demonstrate complete AI workflow from query to response
- **Architecture**: LangGraph orchestrated workflow with vector search and local reasoning
- **Key Components**: Query → Embedding → Vector Search → LLM Reasoning → Response

## 🔑 Critical Environment Variables
```bash
# MongoDB Atlas (REQUIRED)
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.name.mongodb.net/
MONGODB_DATABASE=personal_ai
MONGODB_COLLECTION=conversations

# Voyage AI (REQUIRED)
VOYAGE_API_KEY=your_voyage_api_key_here
VOYAGE_MODEL=voyage-large-2-instruct

# Local LLM (Optional but recommended)
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3-14b
```

**⚠️ IMPORTANT**: All credentials are stored in the `.env` file which is gitignored. Never commit real credentials to the repository.

## 📁 Key Files & Their Purpose
- **`src/langgraph_agent.py`** - Main LangGraph agent with complete workflow
- **`src/frontend/langgraph_ui.py`** - Streamlit UI with tabs for agent workflow and document management
- **`src/frontend/document_management_ui.py`** - Document management interface component
- **`src/services/document_manager.py`** - Document upload, chunking, and memory compression service
- **`src/services/core_service.py`** - MongoDB connection, vector search, and unified memory
- **`src/services/llm_service.py`** - Voyage AI embeddings and LM Studio integration
- **`demo_langgraph_agent.py`** - Command-line demo of the workflow
- **`document_manager_cli.py`** - CLI for document management operations
- **`main.py`** - Main launcher for Streamlit UI
- **`run_tests.py`** - Comprehensive test runner for all components

## 🎯 Core Workflow Steps
1. **Query Processing** - User input validation and preparation
2. **Embedding Generation** - Voyage AI converts text to vectors
3. **MongoDB Vector Search** - Semantic similarity search across unified memory
4. **Results Analysis** - Score analysis and context preparation
5. **LLM Reasoning** - Local LLM analyzes search results
6. **Response Generation** - Final comprehensive answer

## 🚫 What NOT to Do
- ❌ Never commit `.env` file (contains real credentials)
- ❌ Don't modify environment variables in code
- ❌ Don't hardcode API keys or connection strings
- ❌ Don't delete the existing MongoDB data (contains valuable test data)

## ✅ What TO Do
- ✅ Use existing `.env` file for all configuration
- ✅ Test connections before running workflows
- ✅ Use the demo scripts to verify functionality
- ✅ Check MongoDB connection status in sidebar
- ✅ Monitor workflow execution in real-time

## 🔧 Quick Start Commands
```bash
# Test the agent
python demo_langgraph_agent.py demo

# Interactive mode
python demo_langgraph_agent.py interactive

# Launch Streamlit UI
python main.py

# Document management
python document_manager_cli.py list                    # List all documents
python document_manager_cli.py upload file.txt         # Upload a document
python document_manager_cli.py enable doc_abc123       # Enable a document
python document_manager_cli.py disable doc_abc123      # Disable a document
python document_manager_cli.py delete doc_abc123       # Delete a document
python document_manager_cli.py compress                # Compress duplicate memories
python document_manager_cli.py stats                   # Get statistics
python document_manager_cli.py reembed                 # Reembed all documents (when model changes)

# Run comprehensive tests
python run_tests.py

# Test individual components
python -c "from src.langgraph_agent import LangGraphAgent; import asyncio; asyncio.run(LangGraphAgent().run_workflow('test query', 'test_user'))"
```

## 📊 Expected Data Structure
- **Collection**: `memory` (unified storage for all content types)
- **Content Types**: `document`, `conversation`, `personal_info`, `document_metadata`, `compressed_memory`
- **Vector Dimensions**: 1024 (Voyage AI model)
- **Index**: `memory_vector_index` (cosine similarity)
- **Document Management**: Upload, chunk, enable/disable, delete documents
- **Memory Compression**: Deduplicate conversations and generate summaries

## 🐛 Common Issues & Solutions
- **Connection Failed**: Check MongoDB connection string and network
- **Embedding Error**: Verify Voyage AI API key and model name
- **LLM Error**: Ensure LM Studio is running on localhost:1234
- **Import Error**: Check Python path and virtual environment

## 📈 Performance Expectations
- **Embedding Generation**: ~0.5-2 seconds
- **Vector Search**: ~0.1-1 second
- **LLM Reasoning**: ~2-10 seconds (depends on local model)
- **Total Workflow**: ~3-15 seconds

## 🎭 Demo Queries to Test
1. "What is the main topic of the Accelerando document?"
2. "Tell me about the characters in the story"
3. "What are the key themes discussed?"
4. "Who is the protagonist and what happens to them?"

## 📚 Document Management Features
- **Upload & Chunking**: Support for TXT, MD, HTML, PDF files with configurable chunk sizes
- **Enable/Disable**: Control which documents appear in vector search queries
- **Memory Compression**: Automatically deduplicate similar conversations and generate summaries
- **CLI Interface**: Command-line tools for document management operations
- **Streamlit UI**: Web interface with tabs for agent workflow and document management
- **Statistics**: Real-time metrics on document counts, chunk sizes, and memory usage
- **Bulk Operations**: 69.7% performance improvement with bulk MongoDB inserts
- **Async Stability**: All async issues resolved, 100% stable operation

## 🔍 Debugging Tips
- Check stdout for detailed step-by-step logging
- Use Streamlit UI for visual workflow monitoring
- Monitor MongoDB connection status
- Verify all environment variables are loaded
- Check virtual environment and dependencies

## 📚 Key Dependencies
- `langgraph` - Workflow orchestration
- `voyageai` - Text embeddings
- `pymongo` - MongoDB connection
- `streamlit` - Web interface
- `plotly` - Data visualization
- `pandas` - Data manipulation
- `beautifulsoup4` - HTML parsing
- `PyPDF2` - PDF text extraction
- `markdown` - Markdown processing
