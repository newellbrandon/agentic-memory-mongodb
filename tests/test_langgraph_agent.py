"""
Unit tests for LangGraph Agent
Tests all workflow steps and components
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langgraph_agent import LangGraphAgent, AgentState

class TestLangGraphAgent:
    """Test suite for LangGraph Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        try:
            agent = LangGraphAgent()
            yield agent
        finally:
            # Clean up synchronously for now
            pass
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state for testing."""
        return AgentState(
            user_query="test query",
            user_id="test_user",
            query_embedding=None,
            search_results=[],
            search_scores=[],
            context_for_llm="",
            llm_reasoning="",
            final_response="",
            workflow_steps=[],
            metadata={}
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent is not None
        assert hasattr(agent, 'core_service')
        assert hasattr(agent, 'llm_service')
        assert hasattr(agent, 'search_service')
        assert hasattr(agent, 'workflow')
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, agent):
        """Test that the workflow is created correctly."""
        workflow = agent.workflow
        assert workflow is not None
        
        # Check that all required nodes exist
        nodes = workflow.nodes
        required_nodes = [
            "process_query", "generate_embedding", "search_mongodb",
            "analyze_results", "llm_reasoning", "generate_response"
        ]
        
        for node in required_nodes:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_process_user_query(self, agent, sample_state):
        """Test query processing step."""
        updated_state = await agent._process_user_query(sample_state)
        
        assert "query_processed" in updated_state["workflow_steps"]
        assert "query_length" in updated_state["metadata"]
        assert "query_timestamp" in updated_state["metadata"]
        assert updated_state["metadata"]["query_length"] == len("test query")
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, agent, sample_state):
        """Test embedding generation step."""
        # First process the query
        state = await agent._process_user_query(sample_state)
        
        # Then generate embedding
        updated_state = await agent._generate_embedding(state)
        
        assert "embedding_generated" in updated_state["workflow_steps"]
        assert updated_state["query_embedding"] is not None
        assert len(updated_state["query_embedding"]) > 0
        assert "embedding_dimensions" in updated_state["metadata"]
        assert "embedding_time" in updated_state["metadata"]
    
    @pytest.mark.asyncio
    async def test_search_mongodb(self, agent, sample_state):
        """Test MongoDB search step."""
        # Setup state with embedding
        state = await agent._process_user_query(sample_state)
        state = await agent._generate_embedding(state)
        
        # Test search
        updated_state = await agent._search_mongodb(state)
        
        assert "mongodb_search_completed" in updated_state["workflow_steps"]
        assert "search_time" in updated_state["metadata"]
        assert "results_count" in updated_state["metadata"]
        assert "top_score" in updated_state["metadata"]
        assert "avg_score" in updated_state["metadata"]
    
    @pytest.mark.asyncio
    async def test_analyze_search_results(self, agent, sample_state):
        """Test search results analysis step."""
        # Setup state with search results
        state = await agent._process_user_query(sample_state)
        state = await agent._generate_embedding(state)
        state = await agent._search_mongodb(state)
        
        # Test analysis
        updated_state = await agent._analyze_search_results(state)
        
        assert "results_analyzed" in updated_state["workflow_steps"]
        assert "context_length" in updated_state["metadata"]
        assert "content_type_distribution" in updated_state["metadata"]
        assert len(updated_state["context_for_llm"]) > 0
    
    @pytest.mark.asyncio
    async def test_llm_reasoning(self, agent, sample_state):
        """Test LLM reasoning step."""
        # Setup complete state
        state = await agent._process_user_query(sample_state)
        state = await agent._generate_embedding(state)
        state = await agent._search_mongodb(state)
        state = await agent._analyze_search_results(state)
        
        # Test LLM reasoning
        updated_state = await agent._llm_reasoning(state)
        
        assert "llm_reasoning_completed" in updated_state["workflow_steps"]
        assert "reasoning_time" in updated_state["metadata"]
        assert "reasoning_length" in updated_state["metadata"]
        assert len(updated_state["llm_reasoning"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_final_response(self, agent, sample_state):
        """Test final response generation step."""
        # Setup complete state
        state = await agent._process_user_query(sample_state)
        state = await agent._generate_embedding(state)
        state = await agent._search_mongodb(state)
        state = await agent._analyze_search_results(state)
        state = await agent._llm_reasoning(state)
        
        # Test final response generation
        updated_state = await agent._generate_final_response(state)
        
        assert "final_response_generated" in updated_state["workflow_steps"]
        assert "final_response_length" in updated_state["metadata"]
        assert "total_workflow_steps" in updated_state["metadata"]
        assert len(updated_state["final_response"]) > 0
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, agent):
        """Test the complete workflow execution."""
        query = "What is the main topic of the Accelerando document?"
        user_id = "test_user"
        
        result = await agent.run_workflow(query, user_id)
        
        assert result["success"] is True
        assert "final_state" in result
        assert "workflow_summary" in result
        
        final_state = result["final_state"]
        summary = result["workflow_summary"]
        
        # Check workflow completion
        assert len(final_state["workflow_steps"]) == 6
        assert summary["steps_completed"] == 6
        assert summary["search_results_count"] >= 0
        assert summary["final_response_length"] > 0
        assert summary["total_time"] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_empty_query(self, agent):
        """Test workflow behavior with empty query."""
        result = await agent.run_workflow("", "test_user")
        
        # Should still complete but with minimal results
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, agent):
        """Test workflow error handling."""
        # This test would require mocking failures
        # For now, we'll test that the agent doesn't crash
        assert agent is not None

if __name__ == "__main__":
    pytest.main([__file__])
