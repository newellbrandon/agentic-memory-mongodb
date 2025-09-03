"""
Integration tests for LangGraph Agent
Tests actual workflow execution with real services
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langgraph_agent import LangGraphAgent

class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        try:
            agent = LangGraphAgent()
            yield agent
        finally:
            # Clean up synchronously for now
            pass
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_health(self, agent):
        """Test that all service connections are healthy."""
        # Test MongoDB connection
        assert agent.core_service.client is not None
        assert agent.core_service.database is not None
        
        # Test Voyage AI connection
        assert agent.llm_service.voyage_client is not None
        assert agent.llm_service.voyage_model is not None
        
        # Test search service
        assert agent.search_service is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_generation(self, agent):
        """Test Voyage AI embedding generation."""
        test_text = "This is a test query for embedding generation"
        
        start_time = time.time()
        embedding = await agent.llm_service.generate_embedding(test_text)
        end_time = time.time()
        
        # Verify embedding
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding[0], (int, float))
        
        # Check performance
        generation_time = end_time - start_time
        assert generation_time < 5.0  # Should complete within 5 seconds
        
        print(f"✅ Embedding generated: {len(embedding)} dimensions in {generation_time:.3f}s")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mongodb_search(self, agent):
        """Test MongoDB vector search functionality."""
        # Generate test embedding
        test_query = "test search query"
        embedding = await agent.llm_service.generate_embedding(test_query)
        
        # Perform search
        start_time = time.time()
        search_results = await agent.core_service.search_memories(
            embedding,
            content_types=['document', 'conversation'],
            limit=5
        )
        end_time = time.time()
        
        # Verify search results
        assert search_results is not None
        assert isinstance(search_results, list)
        
        # Check performance
        search_time = end_time - start_time
        assert search_time < 3.0  # Should complete within 3 seconds
        
        print(f"✅ MongoDB search completed: {len(search_results)} results in {search_time:.3f}s")
        
        # If we have results, verify structure
        if search_results:
            result = search_results[0]
            assert 'content' in result
            assert 'content_type' in result
            assert 'score' in result
            assert 'metadata' in result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_workflow(self, agent):
        """Test a simple workflow execution."""
        query = "test query"
        user_id = "integration_test_user"
        
        start_time = time.time()
        result = await agent.run_workflow(query, user_id)
        end_time = time.time()
        
        # Verify workflow completion
        assert result["success"] is True
        assert "final_state" in result
        assert "workflow_summary" in result
        
        # Check performance
        total_time = end_time - start_time
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Verify workflow steps
        final_state = result["final_state"]
        assert len(final_state["workflow_steps"]) == 6
        
        print(f"✅ Simple workflow completed in {total_time:.3f}s")
        print(f"   Steps: {len(final_state['workflow_steps'])}")
        print(f"   Results: {result['workflow_summary']['search_results_count']}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_query_workflow(self, agent):
        """Test workflow with a document-related query."""
        query = "What is the main topic of the Accelerando document?"
        user_id = "document_test_user"
        
        start_time = time.time()
        result = await agent.run_workflow(query, user_id)
        end_time = time.time()
        
        # Verify workflow completion
        assert result["success"] is True
        
        # Check that we got meaningful results
        final_state = result["final_state"]
        assert len(final_state["search_results"]) > 0
        
        # Check performance
        total_time = end_time - start_time
        assert total_time < 45.0  # Document queries might take longer
        
        print(f"✅ Document query workflow completed in {total_time:.3f}s")
        print(f"   Search results: {len(final_state['search_results'])}")
        print(f"   Top score: {final_state['metadata'].get('top_score', 'N/A')}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_character_query_workflow(self, agent):
        """Test workflow with a character-related query."""
        query = "Tell me about the characters in the story"
        user_id = "character_test_user"
        
        start_time = time.time()
        result = await agent.run_workflow(query, user_id)
        end_time = time.time()
        
        # Verify workflow completion
        assert result["success"] is True
        
        # Check that we got meaningful results
        final_state = result["final_state"]
        assert len(final_state["search_results"]) > 0
        
        # Check performance
        total_time = end_time - start_time
        assert total_time < 45.0
        
        print(f"✅ Character query workflow completed in {total_time:.3f}s")
        print(f"   Search results: {len(final_state['search_results'])}")
        print(f"   Response length: {len(final_state['final_response'])}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, agent):
        """Test workflow error handling with invalid inputs."""
        # Test with very long query
        long_query = "test " * 1000
        user_id = "error_test_user"
        
        result = await agent.run_workflow(long_query, user_id)
        
        # Should still complete (though might take longer)
        assert result["success"] is True
        
        print("✅ Long query workflow handled successfully")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_performance_benchmark(self, agent):
        """Benchmark workflow performance with multiple queries."""
        test_queries = [
            "What is the main topic?",
            "Tell me about characters",
            "What are the themes?",
            "Who is the protagonist?",
            "What happens in the story?"
        ]
        
        total_time = 0
        total_results = 0
        successful_workflows = 0
        
        for i, query in enumerate(test_queries):
            user_id = f"benchmark_user_{i}"
            
            start_time = time.time()
            result = await agent.run_workflow(query, user_id)
            end_time = time.time()
            
            if result["success"]:
                successful_workflows += 1
                workflow_time = end_time - start_time
                total_time += workflow_time
                total_results += result["workflow_summary"]["search_results_count"]
                
                print(f"✅ Query {i+1}: {workflow_time:.3f}s")
            else:
                print(f"❌ Query {i+1}: Failed")
        
        # Calculate averages
        if successful_workflows > 0:
            avg_time = total_time / successful_workflows
            avg_results = total_results / successful_workflows
            
            print(f"\n📊 Performance Benchmark Results:")
            print(f"   Successful workflows: {successful_workflows}/{len(test_queries)}")
            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Average results: {avg_results:.1f}")
            
            # Performance assertions
            assert avg_time < 20.0  # Average should be under 20 seconds
            assert successful_workflows >= len(test_queries) * 0.8  # 80% success rate
        else:
            pytest.fail("No successful workflows to benchmark")

if __name__ == "__main__":
    pytest.main([__file__])
