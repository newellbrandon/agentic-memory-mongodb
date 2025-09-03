"""
LangGraph workflow graph for showcasing workflow orchestration with MongoDB Vector Search.
This demonstrates the core value proposition of the application.
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from .nodes import (
    extract_personal_info_node,
    memory_search_node,
    response_generation_node,
    memory_storage_node,
    document_processing_node,
    web_search_node
)

# Define the state structure for our workflows
class WorkflowState(TypedDict):
    message: str
    user_id: str
    workflow_step: str
    personal_info: Dict[str, Any]
    search_results: Dict[str, Any]
    ai_response: str
    stored_memories: Dict[str, Any]
    file_path: str
    document_chunks: list
    document_embeddings: list
    search_query: str
    web_results: list
    error: str

def create_personal_memory_workflow() -> StateGraph:
    """Create the personal memory workflow graph using LangGraph."""
    
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("extract_personal_info", extract_personal_info_node)
    workflow.add_node("memory_search", memory_search_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("memory_storage", memory_storage_node)
    
    # Define the workflow flow
    workflow.set_entry_point("extract_personal_info")
    workflow.add_edge("extract_personal_info", "memory_search")
    workflow.add_edge("memory_search", "response_generation")
    workflow.add_edge("response_generation", "memory_storage")
    workflow.add_edge("memory_storage", END)
    
    return workflow.compile()

def create_document_workflow() -> StateGraph:
    """Create the document processing workflow graph using LangGraph."""
    
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("document_processing", document_processing_node)
    workflow.add_node("memory_storage", memory_storage_node)
    
    # Define the workflow flow
    workflow.set_entry_point("document_processing")
    workflow.add_edge("document_processing", "memory_storage")
    workflow.add_edge("memory_storage", END)
    
    return workflow.compile()

def create_research_workflow() -> StateGraph:
    """Create the research workflow graph using LangGraph."""
    
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("memory_storage", memory_storage_node)
    
    # Define the workflow flow
    workflow.set_entry_point("web_search")
    workflow.add_edge("web_search", "memory_storage")
    workflow.add_edge("memory_storage", END)
    
    return workflow.compile()

# Workflow orchestrator class
class WorkflowOrchestrator:
    """Orchestrates LangGraph workflows for showcasing workflow capabilities."""
    
    def __init__(self):
        """Initialize the workflow orchestrator."""
        self.personal_memory_workflow = create_personal_memory_workflow()
        self.document_workflow = create_document_workflow()
        self.research_workflow = create_research_workflow()
    
    async def run_personal_memory_workflow(self, message: str, user_id: str) -> Dict[str, Any]:
        """Run the personal memory workflow using LangGraph."""
        try:
            initial_state = {
                "message": message,
                "user_id": user_id,
                "workflow_step": "started",
                "personal_info": {},
                "search_results": {},
                "ai_response": "",
                "stored_memories": {},
                "file_path": "",
                "document_chunks": [],
                "document_embeddings": [],
                "search_query": "",
                "web_results": [],
                "error": ""
            }
            
            # Execute the workflow
            result = await self.personal_memory_workflow.ainvoke(initial_state)
            return result
            
        except Exception as e:
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "workflow_step": "failed"
            }
    
    async def run_document_workflow(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """Run the document processing workflow using LangGraph."""
        try:
            initial_state = {
                "message": "",
                "user_id": user_id,
                "workflow_step": "started",
                "personal_info": {},
                "search_results": {},
                "ai_response": "",
                "stored_memories": {},
                "file_path": file_path,
                "document_chunks": [],
                "document_embeddings": [],
                "search_query": "",
                "web_results": [],
                "error": ""
            }
            
            # Execute the workflow
            result = await self.document_workflow.ainvoke(initial_state)
            return result
            
        except Exception as e:
            return {
                "error": f"Document workflow execution failed: {str(e)}",
                "workflow_step": "failed"
            }
    
    async def run_research_workflow(self, query: str) -> Dict[str, Any]:
        """Run the research workflow using LangGraph."""
        try:
            initial_state = {
                "message": "",
                "user_id": "",
                "workflow_step": "started",
                "personal_info": {},
                "search_results": {},
                "ai_response": "",
                "stored_memories": {},
                "file_path": "",
                "document_chunks": [],
                "document_embeddings": [],
                "search_query": query,
                "web_results": [],
                "error": ""
            }
            
            # Execute the workflow
            result = await self.research_workflow.ainvoke(initial_state)
            return result
            
        except Exception as e:
            return {
                "error": f"Research workflow execution failed: {str(e)}",
                "workflow_step": "failed"
            }
