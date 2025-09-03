"""
LangGraph workflow nodes for showcasing workflow orchestration with MongoDB Vector Search.
This demonstrates the core value proposition of the application.
"""

from langgraph.func import task
from typing import Dict, Any, List
import asyncio

def extract_personal_info_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract personal information from user messages using LangGraph workflow."""
    message = state.get("message", "")
    user_id = state.get("user_id", "unknown")
    
    # Extract personal info (name, preferences, etc.)
    personal_info = {}
    
    # Check for name mentions
    import re
    name_patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
        r"(\w+) is my name"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message.lower())
        if match:
            name = match.group(1).title()
            personal_info["name"] = name
            break
    
    # Check for preferences
    preference_patterns = [
        (r"i like (\w+)", "likes"),
        (r"i love (\w+)", "loves"),
        (r"i prefer (\w+)", "preferences"),
        (r"i'm from (\w+)", "location"),
        (r"i work as (\w+)", "profession")
    ]
    
    for pattern, category in preference_patterns:
        match = re.search(pattern, message.lower())
        if match:
            value = match.group(1).title()
            personal_info[category] = value
    
    state["personal_info"] = personal_info
    state["workflow_step"] = "personal_info_extracted"
    return state

def memory_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search unified memory using MongoDB Vector Search via LangGraph workflow."""
    message = state.get("message", "")
    user_id = state.get("user_id", "unknown")
    
    # This would integrate with the core_service for vector search
    # For now, simulate the workflow step
    state["workflow_step"] = "memory_searched"
    state["search_results"] = {
        "personal_info": [],
        "documents": [],
        "conversations": []
    }
    return state

def response_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI response using LangGraph workflow orchestration."""
    message = state.get("message", "")
    personal_info = state.get("personal_info", {})
    search_results = state.get("search_results", {})
    
    # This would integrate with the LLM service
    # For now, simulate the workflow step
    state["workflow_step"] = "response_generated"
    state["ai_response"] = f"Processed message: {message[:50]}..."
    
    if personal_info.get("name"):
        state["ai_response"] += f" Nice to meet you, {personal_info['name']}!"
    
    return state

def memory_storage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store conversation and personal info in unified memory via LangGraph workflow."""
    message = state.get("message", "")
    user_id = state.get("user_id", "unknown")
    personal_info = state.get("personal_info", {})
    ai_response = state.get("ai_response", "")
    
    # This would integrate with the core_service for storage
    state["workflow_step"] = "memory_stored"
    state["stored_memories"] = {
        "user_message": message,
        "ai_response": ai_response,
        "personal_info": personal_info,
        "user_id": user_id
    }
    
    return state

def document_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process documents using LangGraph workflow orchestration."""
    file_path = state.get("file_path", "")
    user_id = state.get("user_id", "unknown")
    
    # This would integrate with the document_service
    state["workflow_step"] = "document_processed"
    state["document_chunks"] = []
    state["document_embeddings"] = []
    
    return state

def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform web search using LangGraph workflow orchestration."""
    query = state.get("search_query", "")
    
    # This would integrate with the search_service
    state["workflow_step"] = "web_searched"
    state["web_results"] = ""
    
    return state

# Workflow orchestration functions
async def run_personal_memory_workflow(message: str, user_id: str) -> Dict[str, Any]:
    """Run the personal memory workflow using LangGraph task orchestration."""
    state = {
        "message": message,
        "user_id": user_id,
        "workflow_step": "started"
    }
    
    # Execute workflow steps sequentially
    state = await extract_personal_info_node(state)
    state = await memory_search_node(state)
    state = await response_generation_node(state)
    state = await memory_storage_node(state)
    
    state["workflow_step"] = "completed"
    return state

async def run_document_workflow(file_path: str, user_id: str) -> Dict[str, Any]:
    """Run the document processing workflow using LangGraph task orchestration."""
    state = {
        "file_path": file_path,
        "user_id": user_id,
        "workflow_step": "started"
    }
    
    # Execute workflow steps sequentially
    state = await document_processing_node(state)
    state = await memory_storage_node(state)
    
    state["workflow_step"] = "completed"
    return state

async def run_research_workflow(query: str) -> Dict[str, Any]:
    """Run the research workflow using LangGraph task orchestration."""
    state = {
        "search_query": query,
        "workflow_step": "started"
    }
    
    # Execute workflow steps sequentially
    state = await web_search_node(state)
    state = await memory_storage_node(state)
    
    state["workflow_step"] = "completed"
    return state
