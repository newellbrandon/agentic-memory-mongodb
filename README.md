# 🤖 Agentic Memory with MongoDB Demo

A LangGraph agent demonstration showcasing **MongoDB Vector Search** with persistent AI memory, document processing, and intelligent reasoning.

![LangGraph Agentic Memory Workflow UI](screenshot.png)

## 🚀 Quick Start

### Setup

```bash
# Clone and setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your credentials
```

### Environment Variables

```bash
# Required
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.name.mongodb.net/
MONGODB_DATABASE=personal_ai
MONGODB_COLLECTION=conversations
VOYAGE_API_KEY=your_voyage_api_key_here
VOYAGE_MODEL=voyage-large-2-instruct  # Default model, configurable

# Optional
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3-14b  # Default model, configurable
```

### Launch

```bash
# Quick demo
python demo_langgraph_agent.py demo

# Interactive mode
python demo_langgraph_agent.py interactive

# Web UI
python main.py

# Document management
python document_manager_cli.py list
python document_manager_cli.py upload file.txt
python document_manager_cli.py compress
```

## ✨ Features

- **🔄 LangGraph Workflow**: 9-step orchestrated workflow with real-time monitoring
- **🔍 Vector Search**: MongoDB Atlas semantic search across unified memory
- **📄 Document Intelligence**: Upload PDF, MD, DOCX, HTML with AI chunking
- **🧠 Persistent Memory**: Remembers conversations and user preferences across sessions
- **⚡ Local LLM**: LM Studio integration for private reasoning
- **📊 Analytics**: Real-time workflow metrics and performance tracking

## 🏗️ Architecture

```text
User Query → LangGraph Agent → Embedding → Vector Search → LLM Reasoning → Response
     ↓              ↓              ↓            ↓              ↓            ↓
Chat Input → Workflow Orchestrator → Voyage AI → MongoDB Atlas → Local LLM → Answer
```

## 🎯 Usage

### Streamlit UI

- **Agent Workflow Tab**: Chat with AI and monitor execution
- **Document Management Tab**: Upload and manage documents
- **Workflow Visualization**: Real-time step-by-step monitoring

## 🔧 Technical Stack

- **LangGraph**: Workflow orchestration with state management
- **Voyage AI**: Text embeddings (configurable via VOYAGE_MODEL, 1024 dimensions)
- **MongoDB Atlas**: Vector search with HNSW indexing (memory_vector_index)
- **LM Studio**: Local LLM inference (configurable via LM_STUDIO_MODEL)
- **Streamlit**: Web interface

### MongoDB Configuration

```javascript
{
  "vectorSize": 1024,
  "metric": "cosine",
  "type": "vector"
}
```

## 📁 Project Structure

```text
agentic-mongodb/
├── src/
│   ├── langgraph_agent.py         # Main agent
│   ├── services/                  # Core services
│   ├── workflow/                  # LangGraph orchestration
│   └── frontend/                  # Streamlit UI
├── demo_langgraph_agent.py        # CLI demo
├── document_manager_cli.py        # Document management
├── main.py                        # Main launcher for Streamlit UI
└── run_tests.py                   # Test runner
```

## 📊 Performance

- **Embedding Generation**: ~0.5-2 seconds
- **Vector Search**: ~0.1-1 second  
- **LLM Reasoning**: ~2-10 seconds
- **Total Workflow**: ~3-15 seconds

## 🐛 Troubleshooting

**Connection Issues**: Check MongoDB connection string and network
**Embedding Errors**: Verify Voyage AI API key and model name
**LLM Errors**: Ensure LM Studio is running on localhost:1234
**Import Errors**: Check Python path and virtual environment


## 📚 Key Dependencies

- `langgraph` - Workflow orchestration
- `voyageai` - Text embeddings
- `pymongo` - MongoDB connection
- `streamlit` - Web interface
- `plotly` - Data visualization

**🚀 Quick Start**: `python demo_langgraph_agent.py demo`

**📖 Documentation**: See `AGENTS.md` for detailed project information
