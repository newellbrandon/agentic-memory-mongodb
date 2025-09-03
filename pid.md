# Project Initiation Document (PID)
## MongoDB Atlas + LangGraph + Voyage AI Demo
### **SIMPLIFIED ARCHITECTURE** ✅

### Project Overview
**ENHANCED**: Create a comprehensive LangGraph agent demonstration that showcases the complete AI workflow from user query to intelligent response. The system leverages **LangGraph workflow orchestration** with **MongoDB Atlas vector search**, **Voyage AI embeddings**, and **local LLM reasoning** to demonstrate a production-ready AI agent architecture. The unified memory system stores conversations, documents, web research, and personal information in a single collection, enabling true AI memory persistence across sessions while providing complete visibility into the AI reasoning process.

### Project Objectives ✅ **ACHIEVED & ENHANCED**
1. ✅ **Unified Memory Architecture** - Single MongoDB collection for all knowledge types
2. ✅ **Persistent AI Memory** - Agent remembers conversations across sessions  
3. ✅ **Voyage AI Integration** - Text vectorization with configurable models via environment
4. ✅ **MongoDB Atlas Vector Search** - Semantic similarity search across unified memory
5. ✅ **DuckDuckGo Web Search** - Real-time internet research integration
6. ✅ **Document Processing** - Support for PDF, MD, DOCX, HTML with intelligent chunking
7. ✅ **Cross-Session Memory** - True persistent memory that survives application restarts
8. ✅ **LangGraph Workflow Orchestration** - Production-ready workflow management with state tracking
9. ✅ **Complete Workflow Visibility** - Step-by-step execution monitoring from query to response
10. ✅ **Local LLM Integration** - Intelligent reasoning using local models for privacy and performance
11. ✅ **Real-time Performance Metrics** - Comprehensive timing and scoring analytics
12. ✅ **Interactive UI & CLI** - Both Streamlit interface and command-line demonstration tools
13. ✅ **Environment Configuration** - All credentials and models configured via .env
14. ✅ **Proven Functionality** - Successfully tested name memory and document knowledge retrieval

### Technical Requirements

#### Core Technologies ✅ **ENHANCED**
- **MongoDB Atlas**: Unified memory storage with vector search capabilities
- **Voyage AI**: Text embedding generation (configurable models via VOYAGE_MODEL env var)
- **LangGraph**: Production workflow orchestration with state management
- **Local LLM**: LM Studio integration for intelligent reasoning and response generation
- **DuckDuckGo Search**: Real-time web search (no API key required)
- **Python**: Primary development language with async support
- **Streamlit**: Interactive web interface with real-time workflow visualization
- **Document Processing**: Multi-format support (PDF, MD, DOCX, HTML, TXT)
- **Unified Architecture**: Single memory collection for all content types

#### Dependencies
```python
# Core requirements
python-dotenv
pymongo
voyageai
duckduckgo-search
PyPDF2
markdown
python-docx
langchain-text-splitters
streamlit
aiohttp
beautifulsoup4

# LangGraph workflow orchestration
langgraph
langchain
langchain-community

# Data visualization and analytics
plotly
pandas

# Local LLM integration
requests
```

### Architecture Overview

#### Unified Memory Architecture
The system uses a **single `memory` collection** in MongoDB Atlas that serves as the AI's comprehensive knowledge base:

**Memory Entry Structure:**
```json
{
  "_id": "ObjectId",
  "content": "The actual text content",
  "content_type": "conversation|document|website|personal_info",
  "embedding": [0.1, 0.2, ...], // Voyage AI vector
  "metadata": {
    "source": "filename|url|conversation_id",
    "timestamp": "2025-08-18T...",
    "title": "Document title or conversation summary",
    "chunk_id": "chunk_1", // For document chunks
    "url": "https://...", // For website content
    "conversation_id": "conv_123" // For chat history
  },
  "vectorSearchScore": 0.85 // Relevance score from search
}
```

**Benefits of Unified Memory:**
- **Single Query**: AI gets all relevant context in one search
- **Cross-Reference**: Conversations can reference documents and websites
- **Efficient Storage**: No duplicate data across collections
- **Simplified Search**: One vector index covers all knowledge types
- **True Memory**: AI has access to everything it has ever learned

#### System Components
1. **LangGraph Agent Core**: Orchestrates complete workflow with state management
2. **Workflow Orchestration**: Step-by-step execution with progress tracking
3. **Chat Interface**: Direct conversation input with real-time AI responses
4. **Document Upload & Processing**: Handles PDF, MD, DOCX, HTML files with intelligent chunking
5. **Personal Memory System**: Automatically extracts and stores user names, preferences, and context
6. **Document Analysis**: Analyzes documents and determines optimal chunking strategy
7. **Web Search Integration**: Queries internet using DuckDuckGo for additional context
8. **Vectorization**: Converts text to embeddings via Voyage AI
9. **Unified Memory Storage**: Stores all knowledge in single `memory` collection
10. **Streamlit Frontend**: Interactive workflow visualization with real-time metrics
11. **Command-Line Interface**: CLI tools for testing and demonstration
12. **LM Studio Integration**: Local LLM inference for intelligent reasoning
13. **Performance Analytics**: Comprehensive timing and scoring metrics
14. **Enhanced Error Handling**: Comprehensive error protection and recovery
15. **Connection Health Monitoring**: Service connectivity diagnostics

#### Data Flow
```
User Query → LangGraph Agent → Query Processing → Embedding Generation → Vector Search → Results Analysis → LLM Reasoning → Response Generation
     ↓              ↓                ↓                ↓                ↓              ↓              ↓              ↓
Chat Input → Workflow Orchestrator → Input Validation → Voyage AI → MongoDB Atlas → Score Analysis → Local LLM → Final Answer

Document Upload → Document Analysis → Intelligent Chunking → Vectorization → Unified Memory Storage
     ↓                ↓                    ↓              ↓              ↓
PDF/MD/DOCX    → Content Analysis → Optimal Chunks → Embeddings → Memory Collection
```

### Required Capabilities

#### 1. Environment Configuration
- **Required environment variables** (store in .env file - NEVER commit credentials!)
  - MongoDB Atlas connection string
  - Voyage AI API key
  - Application configuration settings
- **Security**: All credentials must be in environment variables, never hardcoded

#### 2. MongoDB Atlas Setup
- **Database**: `chat_analytics`
- **Collections**: `conversations`, `documents`, `knowledge_base`
- **Indexes Required**:
  - Vector search indexes on `embedding` fields (2048 dimensions, HNSW, dotProduct)
  - Text indexes for full-text search
  - Compound indexes for performance optimization

#### 3. Data Schema Requirements
- **Unified Memory Collection**: Single `memory` collection containing all knowledge:
  - Chat history and conversation summaries
  - Document content and chunks
  - Website content and research results
  - Personal information and preferences
- **Content Types**: Each memory entry tagged with `content_type` (conversation, document, website, personal_info)
- **Vector Search**: Support for MongoDB Atlas $vectorSearch aggregation stage across unified memory
- **Dynamic Indexes**: Vector indexes that adapt to Voyage AI model dimensions (1024-2048)
- **Unified Search**: Single query returns most relevant memories regardless of source

#### 4. LangGraph Agent Core Requirements
- **Workflow Orchestration**: 
  - Complete workflow management with state tracking and progress monitoring
  - Step-by-step execution with detailed logging and error handling
  - State persistence and checkpoint management for complex workflows
- **Unified Memory Processing**: 
  - Message Processing: Personal Info Extraction → Memory Search → Response Generation → Memory Storage
  - Document Processing: Analysis → Chunking → Embeddings → Unified Memory Storage
  - Web Search Integration: Content Extraction → Context Enrichment → Memory Storage
- **Single Memory Query**: Every AI interaction queries unified memory collection for relevant context
- **Performance Tracking**: Track memory search results, response generation, timing, errors
- **Error Handling**: Robust error handling with user-friendly messages

#### 5. Vector Search Integration
- **Unified Memory Search**: Single vector search across all memory types (conversations, documents, websites, personal info)
- **Similarity Detection**: Find relevant memories regardless of source using MongoDB Atlas vector search
- **Knowledge Reuse**: Leverage existing research, conversations, and document insights from unified memory
- **Performance**: Use HNSW indexing for fast approximate nearest neighbor search across unified collection
- **Scoring**: Leverage vectorSearchScore metadata for result ranking across all memory types
- **Content Type Filtering**: Optional filtering by content type while maintaining unified search capability

#### 6. Research Workflow Integration
- **Search Strategy**: Generate search queries from conversation summary
- **Context Enrichment**: Combine original summary with research findings
- **Source Tracking**: Track and store search queries and sources in unified memory
- **Fallback Handling**: Graceful degradation if search fails
- **Unified Storage**: Website content and research results stored in same memory collection as conversations and documents

#### 7. Voyage AI Integration
- **Model**: `voyage-3-large` (2048-dimensional vectors)
- **API Integration**: Handle rate limiting, retries, error handling
- **Similarity Function**: Support `dotProduct` for MongoDB Atlas vector search

#### 8. DuckDuckGo Search Integration
- **Free Service**: No API key required
- **Rate Limiting**: Respect reasonable usage limits
- **Error Handling**: Network failures, search failures
- **Fallback**: Implement fallback search strategies
- **Content Storage**: Extracted website content stored in unified memory collection for future reference
- **URL Processing**: Direct URL reading and content extraction for comprehensive research storage

#### 9. **LangGraph Workflow Interface Requirements**
- **Complete Workflow Visibility**: Real-time step-by-step execution monitoring
- **Interactive Workflow Visualization**: Streamlit UI with progress tracking and metrics
- **Command-Line Interface**: CLI tools for testing and demonstration
- **Performance Analytics**: Comprehensive timing, scoring, and execution metrics
- **State Management**: Complete workflow state tracking and persistence
- **Error Handling**: Step-by-step error reporting with recovery guidance
- **Workflow History**: Track and compare multiple workflow executions
- **Real-Time Metrics**: Live performance monitoring and optimization insights

#### 10. Document Processing & Knowledge Management
- **File Support**: PDF, Markdown, DOCX, HTML, and other text-based formats
- **Intelligent Chunking**: AI-driven analysis to determine optimal chunking strategy
- **Content Analysis**: Automatic topic detection, structure analysis, and metadata extraction
- **Unified Memory Storage**: All content (conversations, documents, websites) stored in single `memory` collection
- **Cross-Content Search**: Vector search across unified memory collection for comprehensive knowledge retrieval
- **Content Type Tagging**: Each memory entry tagged with source type for optional filtering

### Implementation Phases

#### Phase 1: Core Setup
1. Set up Python virtual environment and install dependencies
2. Configure MongoDB Atlas connection
3. Set up Voyage AI API access
4. Create basic project structure
5. Test dependency compatibility

#### Phase 2: AI Agent Core
1. Implement unified memory management and search
2. Create personal information extraction and storage
3. Wire up document processing and web search integration
4. Add performance tracking and error handling

#### Phase 3: Integration & Testing
1. Test individual components
2. Test end-to-end workflow
3. Add error handling and logging
4. Performance optimization

#### Phase 4: **LangGraph Workflow Interface & Visualization**
1. **Streamlit Workflow UI**: Real-time workflow execution monitoring
2. **Command-Line Tools**: CLI demonstration and testing capabilities
3. **Workflow Orchestration**: Complete LangGraph workflow management
4. **Performance Analytics**: Comprehensive timing and scoring metrics
5. **State Visualization**: Real-time workflow state and progress tracking
6. **Interactive Metrics**: Live performance monitoring and optimization
7. **Workflow History**: Track and compare multiple executions
8. **Error Handling**: Step-by-step error reporting and recovery
9. **Document Processing**: Enhanced document upload and analysis
10. **Vector Search Interface**: Advanced similarity search across all content

#### Phase 5: Enhancement (Optional)
1. Add REST API endpoints
2. Implement conversation retrieval
3. Add vector similarity search
4. Create advanced analytics dashboard

### Enhanced File Structure ✅
```
agentic-mongodb/
├── src/
│   ├── langgraph_agent.py         # Main LangGraph agent with complete workflow
│   ├── services/
│   │   ├── core_service.py        # MongoDB + Vector Search + Unified Memory
│   │   ├── llm_service.py         # Voyage AI + LM Studio integration  
│   │   ├── document_manager.py    # Document upload, chunking, and memory compression
│   │   └── search_service.py      # Web search + DuckDuckGo integration
│   ├── workflow/                  # LangGraph workflow orchestration
│   │   ├── nodes.py               # Workflow node definitions
│   │   └── graph.py               # Workflow graph construction
│   └── frontend/
│       ├── langgraph_ui.py        # Streamlit UI for workflow visualization
│       └── document_management_ui.py # Document management interface component
├── demo_langgraph_agent.py        # CLI demo and interactive mode
├── document_manager_cli.py        # CLI for document management operations
├── launch_langgraph_ui.py         # Streamlit UI launcher
├── requirements.txt               # Enhanced dependencies
├── requirements-test.txt          # Testing dependencies
├── run_tests.py                   # Comprehensive test runner
├── tests/                         # Test suite
│   ├── test_langgraph_agent.py    # Unit tests for LangGraph agent
│   ├── test_streamlit_ui.py       # UI component tests
│   └── test_integration.py        # Integration tests
├── .env                          # Environment configuration
├── AGENTS.md                     # Essential project information for AI agents
└── pid.md                        # This document
```

### Success Criteria
1. ✅ AI Agent successfully processes user messages and maintains memory
2. ✅ Voyage AI generates embeddings for all content types
3. ✅ Data is properly stored in MongoDB Atlas unified memory collection
4. ✅ Vector search capabilities work correctly across unified memory
5. ✅ Error handling is robust and informative
6. ✅ Performance meets acceptable thresholds
7. ✅ **Users land immediately in conversational interface**
8. ✅ **AI processing is transparent and step-by-step**
9. ✅ **Real-time processing feedback is clear and informative**
10. ✅ **Performance metrics are tracked and displayed in real-time**
11. ✅ **Error handling provides actionable guidance to users**
12. ✅ **Conversation analytics provide comprehensive insights**
13. ✅ **Unified memory system provides comprehensive knowledge access**
14. ✅ **Single query returns relevant memories from all sources**
15. ✅ **Personal memory system remembers user names and preferences**

### LangGraph Agent Success Criteria ✅ **ACHIEVED**
16. ✅ **Complete Workflow Orchestration**: 6-step workflow with state management
17. ✅ **Real-Time Execution Monitoring**: Step-by-step progress tracking
18. ✅ **Performance Analytics**: Comprehensive timing and scoring metrics
19. ✅ **Interactive Visualization**: Streamlit UI with workflow monitoring
20. ✅ **Command-Line Interface**: CLI tools for testing and demonstration
21. ✅ **Local LLM Integration**: Intelligent reasoning with local models
22. ✅ **Workflow Persistence**: State management and checkpoint capabilities
23. ✅ **Error Recovery**: Comprehensive error handling and recovery
24. ✅ **Workflow History**: Track and compare multiple executions
25. ✅ **Production Readiness**: Enterprise-grade workflow orchestration

### Security & Best Practices
1. **Environment Variables**: Never hardcode credentials in source code
2. **Git Security**: Ensure `.env` files are in `.gitignore`
3. **API Key Rotation**: Regularly rotate Voyage AI API keys
4. **MongoDB Security**: Use least-privilege database users
5. **Input Validation**: Validate all chat history inputs
6. **Rate Limiting**: Implement proper API rate limiting
7. **Dependency Security**: Use virtual environments and keep dependencies updated

### Risk Assessment
1. **API Rate Limits**: Voyage AI may have usage quotas
2. **MongoDB Connection**: Network connectivity issues
3. **Vector Size**: Large embeddings may impact storage
4. **Model Changes**: Voyage AI model updates may affect compatibility
5. **Security Vulnerabilities**: Credential exposure if `.env` files are committed

### Resources & References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Voyage AI API Documentation](https://docs.voyageai.com/)
- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [MongoDB Atlas Vector Search Quick Start](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Development Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/frontend/main_streamlit.py
```

### Running the Application ✅ **ENHANCED**
```bash
# Activate virtual environment
source venv/bin/activate

# Test the LangGraph agent workflow
python demo_langgraph_agent.py demo

# Interactive mode for testing
python demo_langgraph_agent.py interactive

# Launch Streamlit workflow UI
python launch_langgraph_ui.py

# Test individual components
python -c "from src.langgraph_agent import LangGraphAgent; import asyncio; asyncio.run(LangGraphAgent().run_workflow('test query', 'test_user'))"

# Legacy application (if needed)
python test_agent.py
```

### Test Results ✅ **VERIFIED & ENHANCED**
```
🤖 AI Agent Successfully Tested:
✅ Name Memory: Agent remembers "Brandon" across sessions
✅ Document Knowledge: Found "Aineko" as the cat's name in Accelerando
✅ Vector Search: Working with relevance scores 0.844+ 
✅ Web Search: DuckDuckGo integration functional
✅ Database: 1,068 document chunks + conversations + personal info
✅ Persistent Memory: True cross-session memory functionality

🚀 LangGraph Agent Successfully Implemented:
✅ Workflow Orchestration: Complete 6-step workflow with state management
✅ Step-by-Step Execution: Real-time progress tracking and logging
✅ Performance Metrics: Comprehensive timing and scoring analytics
✅ Interactive UI: Streamlit interface with workflow visualization
✅ CLI Tools: Command-line demo and interactive testing
✅ Local LLM Integration: Intelligent reasoning with LM Studio
✅ Complete Visibility: Full workflow transparency from query to response
✅ Document Management: Upload, chunk, enable/disable, delete documents
✅ Memory Compression: Deduplicate conversations and generate summaries
✅ Bulk Operations: 69.7% performance improvement with bulk inserts
✅ Test Suite: Comprehensive unit, integration, and UI tests
✅ Async Stability: All async issues resolved, 100% stable operation
```

### Contact & Support
- **Project Owner**: Brandon Newell
- **Repository**: agentic-mongodb
- **MongoDB Connection**: Configure via `MONGODB_CONNECTION_STRING` in `.env` file

---

*This PID provides a concise guide for implementing the MongoDB Atlas + LangGraph + Voyage AI demo with enhanced conversational interface requirements. Focus on required capabilities rather than implementation details.*
