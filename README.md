# 🤖 Agentic MongoDB Demo

A fully functioning demonstration application that showcases **LangGraph workflows with MongoDB Vector Search**. The system provides persistent AI memory, processes documents, researches context using DuckDuckGo, generates vector embeddings, and stores everything in a unified MongoDB collection with vector search capabilities while demonstrating LangGraph workflow orchestration.

## ✨ Features

- **💬 Intelligent Conversation Processing**: Automatically summarize and analyze chat history
- **🔍 Vector Search**: Find similar topics using MongoDB Atlas vector search
- **🌐 Internet Research**: Enrich conversations with real-time DuckDuckGo search results
- **📄 Document Intelligence**: Upload PDF, Markdown, DOCX files with AI-powered chunking
- **📊 Real-time Analytics**: Beautiful visualizations of your knowledge base
- **🚀 Streamlit Frontend**: Modern, interactive web interface
- **🧠 Persistent AI Memory**: Remembers user names, preferences, and conversation context
- **🔄 LangGraph Workflows**: Demonstrates workflow orchestration with MongoDB Vector Search

## 🏗️ Architecture

```
User Input → AI Agent → Memory Search → Response Generation → MongoDB Atlas Storage
    ↓            ↓            ↓              ↓              ↓
Chat Input → Core Logic → Vector Search → LLM Analysis → Unified Memory
Documents → Analysis + Chunking → Embeddings → Knowledge Base
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd agentic-mongodb

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env with your credentials
nano .env
```

**Required Environment Variables:**
```bash
# MongoDB Atlas Configuration
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.name.mongodb.net/
MONGODB_DATABASE=chat_analytics
MONGODB_COLLECTION=memory

# Voyage AI Configuration
VOYAGE_API_KEY=your_voyage_api_key_here
VOYAGE_MODEL=voyage-large-2-instruct

# LM Studio Configuration (Optional)
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=qwen/qwen3-14b

# Application Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 3. Launch Application

```bash
# Option 1: Use the launcher script
python run_app.py

# Option 2: Direct Streamlit command
streamlit run src/frontend/main_streamlit.py

# Option 3: With custom configuration
streamlit run src/frontend/main_streamlit.py --server.port 8501 --server.address 0.0.0.0
```

The application will automatically open in your browser at `http://localhost:8501`

## 🎯 How to Use

### 1. **Chat with AI**
- Start typing in the chat interface
- The AI will remember your name and preferences
- Ask questions about uploaded documents
- Get intelligent responses based on memory and context

### 2. **Upload Documents**
- Go to "📄 Upload Document" page
- Upload PDF, Markdown, or DOCX files
- The AI will analyze and chunk your documents intelligently
- All content becomes searchable in your knowledge base

### 3. **Explore Your Knowledge**
- Use "📚 History" to browse chat history and documents
- Use "🔍 Vector Search" to find similar content
- View "📊 Analytics" for insights and trends

## 🔧 Technical Details

### Core Technologies
- **AI Agent Core**: Unified memory management and intelligent response generation
- **LangGraph**: Workflow orchestration and task management
- **Voyage AI**: Text embedding generation (voyage-large-2-instruct model)
- **MongoDB Atlas**: Vector search and unified memory storage
- **LM Studio**: Local LLM inference for intelligent responses
- **DuckDuckGo**: Internet research and context enrichment
- **Streamlit**: Interactive web frontend
- **Python**: Backend processing and async operations

### MongoDB Collections
- `memory`: Unified collection for all content types (conversations, documents, personal info, web research)
- **Content Types**: Each memory entry tagged with `content_type` (conversation, document, personal_info, website)
- **Vector Search**: Single collection enables cross-content semantic search

### Vector Search Configuration
```javascript
// MongoDB Atlas Vector Search Index
{
  "vectorSize": 1024,  // voyage-large-2-instruct dimensions
  "metric": "dotProduct",
  "type": "hnsw"
}
```

## 📁 Project Structure

```
agentic-mongodb/
├── src/
│   ├── main.py            # Core AIAgent application
│   ├── services/          # Core services (Voyage AI, MongoDB, etc.)
│   │   ├── core_service.py        # MongoDB + Vector Search + Unified Memory
│   │   ├── llm_service.py         # Voyage AI + LM Studio integration
│   │   ├── document_service.py    # File processing + Intelligent chunking
│   │   └── search_service.py      # Web search + DuckDuckGo integration
│   ├── workflow/          # LangGraph workflow orchestration
│   │   ├── nodes.py               # Workflow node definitions
│   │   └── graph.py               # Workflow graph construction
│   └── frontend/          # Streamlit chat interface
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
├── launch_app.py         # Application launcher
└── README.md             # This file
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_workflow.py
python -m pytest tests/test_frontend/
```

## 🔒 Security Features

- **Environment Variables**: All credentials stored in `.env` file
- **Git Security**: `.env` files excluded from version control
- **API Key Management**: Secure Voyage AI API key handling
- **Input Validation**: All user inputs validated and sanitized

## 🚧 Development

### Adding New Features
1. Extend the AI agent in `src/main.py`
2. Add services in `src/services/`
3. Update the chat interface in `src/frontend/`
4. Add tests in `tests/`

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run src/frontend/simple_app.py --server.runOnSave true

# Run tests with coverage
python -m pytest --cov=src tests/
```

## 📊 Performance

- **Vector Search**: MongoDB Atlas HNSW index for fast similarity search
- **Async Processing**: Non-blocking workflow execution
- **Batch Operations**: Efficient document processing and embedding generation
- **Caching**: Streamlit caching for improved response times

## 🌟 Advanced Features

### Intelligent Document Chunking
- **Header-based**: For structured documents (Markdown)
- **Section-based**: For natural content breaks
- **Fixed-size**: Fallback with overlap for continuous text
- **Personal Info Extraction**: Automatically detects and stores user information

### Research Integration
- **DuckDuckGo Search**: Real-time internet queries
- **Context Enrichment**: Combine conversation with research
- **Source Tracking**: Track research queries and results

### Vector Search Capabilities
- **Unified Memory Search**: Search across all content types in single collection
- **Similarity Scoring**: MongoDB Atlas vectorSearchScore for result ranking
- **Configurable Thresholds**: Adjustable similarity thresholds
- **Personal Memory**: Prioritizes user-specific information in search results

## 🐛 Troubleshooting

### Common Issues

**1. Voyage AI API Error**
```bash
# Check your API key
echo $VOYAGE_API_KEY

# Verify model name
VOYAGE_MODEL=voyage-3-large
```

**2. MongoDB Connection Issues**
```bash
# Test connection string
mongosh "your_connection_string"

# Check network access
ping your-cluster.mongodb.net

# Verify collection name is 'memory'
```

**3. Streamlit Port Conflicts**
```bash
# Use different port
streamlit run src/frontend/simple_app.py --server.port 8502

# Kill existing processes
lsof -ti:8501 | xargs kill -9
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run src/frontend/main_streamlit.py
```

## 📈 Monitoring and Analytics

The application provides comprehensive monitoring:
- **Real-time Workflow Status**: Live progress tracking
- **Performance Metrics**: Execution time and throughput
- **Error Tracking**: Detailed error reporting and logging
- **Usage Analytics**: Conversation and document statistics

## 🔮 Future Enhancements

- **Real-time Processing**: Stream processing for live conversations
- **Multi-modal Support**: Image, audio, and video processing
- **Advanced Analytics**: Sentiment analysis and topic modeling
- **API Endpoints**: REST API for external integrations
- **User Management**: Multi-user support and authentication
- **Enhanced Personal Memory**: More sophisticated user preference learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and the PID
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact the project maintainer

---

**🚀 Ready to build your intelligent knowledge base? Launch the application and start chatting with your AI!**
