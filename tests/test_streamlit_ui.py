"""
Tests for Streamlit UI components
Tests UI functionality and workflow integration
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Mock streamlit for testing
sys.modules['streamlit'] = Mock()
sys.modules['plotly'] = Mock()
sys.modules['pandas'] = Mock()

from frontend.langgraph_ui import LangGraphUI

class TestStreamlitUI:
    """Test suite for Streamlit UI components."""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock streamlit components."""
        with patch('frontend.langgraph_ui.st') as mock_st:
            # Mock streamlit components
            mock_st.set_page_config = Mock()
            mock_st.title = Mock()
            mock_st.markdown = Mock()
            mock_st.sidebar = Mock()
            mock_st.columns = Mock(return_value=[Mock(), Mock()])
            mock_st.header = Mock()
            mock_st.subheader = Mock()
            mock_st.text_area = Mock()
            mock_st.text_input = Mock()
            mock_st.button = Mock()
            mock_st.info = Mock()
            mock_st.success = Mock()
            mock_st.error = Mock()
            mock_st.warning = Mock()
            mock_st.progress = Mock()
            mock_st.metric = Mock()
            mock_st.expander = Mock()
            mock_st.write = Mock()
            mock_st.json = Mock()
            mock_st.plotly_chart = Mock()
            mock_st.spinner = Mock()
            mock_st.rerun = Mock()
            
            # Mock session state
            mock_st.session_state = {}
            
            yield mock_st
    
    @pytest.fixture
    def ui(self, mock_streamlit):
        """Create a test UI instance."""
        return LangGraphUI()
    
    def test_ui_initialization(self, ui, mock_streamlit):
        """Test UI initialization."""
        assert ui is not None
        mock_streamlit.set_page_config.assert_called_once()
        mock_streamlit.title.assert_called_once()
        mock_streamlit.markdown.assert_called_once()
    
    def test_sidebar_setup(self, ui, mock_streamlit):
        """Test sidebar configuration setup."""
        # Check that sidebar methods were called
        mock_streamlit.sidebar.header.assert_called()
        mock_streamlit.sidebar.subheader.assert_called()
        mock_streamlit.sidebar.button.assert_called()
        mock_streamlit.sidebar.slider.assert_called()
        mock_streamlit.sidebar.checkbox.assert_called()
    
    def test_query_section_setup(self, ui, mock_streamlit):
        """Test query input section setup."""
        mock_streamlit.header.assert_any_call("🔍 Query Input")
        mock_streamlit.text_area.assert_called_once()
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called()
    
    def test_workflow_display_setup(self, ui, mock_streamlit):
        """Test workflow display section setup."""
        mock_streamlit.header.assert_any_call("🔄 Workflow Execution")
    
    def test_metrics_display_setup(self, ui, mock_streamlit):
        """Test metrics display section setup."""
        mock_streamlit.header.assert_any_call("📊 Workflow Metrics")
    
    def test_workflow_history_setup(self, ui, mock_streamlit):
        """Test workflow history section setup."""
        mock_streamlit.header.assert_any_call("📚 Workflow History")
    
    @patch('frontend.langgraph_ui.LangGraphAgent')
    def test_connection_testing(self, mock_agent_class, ui, mock_streamlit):
        """Test connection testing functionality."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Test successful connection
        ui.test_connections()
        mock_streamlit.sidebar.success.assert_called_with("✅ All connections successful!")
        
        # Test failed connection
        mock_agent_class.side_effect = Exception("Connection failed")
        ui.test_connections()
        mock_streamlit.sidebar.error.assert_called()
    
    @patch('frontend.langgraph_ui.asyncio.run')
    def test_workflow_execution(self, mock_asyncio_run, ui, mock_streamlit):
        """Test workflow execution."""
        mock_result = {
            "success": True,
            "final_state": {
                "user_query": "test query",
                "user_id": "test_user",
                "workflow_steps": ["step1", "step2"],
                "final_response": "test response",
                "metadata": {
                    "total_workflow_time": 1.5,
                    "results_count": 5,
                    "top_score": 0.85,
                    "avg_score": 0.75
                }
            },
            "workflow_summary": {
                "total_time": 1.5,
                "steps_completed": 2,
                "search_results_count": 5,
                "final_response_length": 13
            }
        }
        mock_asyncio_run.return_value = mock_result
        
        # Mock agent
        ui.agent = Mock()
        
        # Test workflow execution
        ui.run_workflow("test query", "test_user")
        
        mock_asyncio_run.assert_called_once()
        mock_streamlit.success.assert_called_with("✅ Workflow completed successfully!")
    
    def test_workflow_stats_display(self, ui, mock_streamlit):
        """Test workflow statistics display."""
        # Mock workflow history
        ui.workflow_history = [
            {
                "metadata": {"total_workflow_time": 1.5, "results_count": 5},
                "user_query": "test query",
                "user_id": "test_user"
            }
        ]
        
        ui.show_workflow_stats()
        
        mock_streamlit.header.assert_any_call("📊 Workflow Statistics")
        mock_streamlit.metric.assert_called()
    
    def test_demo_queries_execution(self, ui, mock_streamlit):
        """Test demo queries execution."""
        with patch.object(ui, 'run_workflow') as mock_run:
            ui.run_demo_queries()
            
            # Should call run_workflow for each demo query
            assert mock_run.call_count == 4  # 4 demo queries
    
    def test_step_details_display(self, ui, mock_streamlit):
        """Test step details display."""
        workflow = {
            "workflow_steps": ["embedding_generated"],
            "metadata": {
                "embedding_dimensions": 1024,
                "embedding_time": 0.5
            },
            "query_embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        ui.show_step_details(workflow)
        
        # Should call appropriate display methods
        mock_streamlit.metric.assert_called()
    
    def test_search_details_display(self, ui, mock_streamlit):
        """Test search details display."""
        workflow = {
            "search_results": [
                {
                    "rank": 1,
                    "score": 0.85,
                    "content_type": "document",
                    "content_preview": "test content",
                    "metadata": {"test": "data"}
                }
            ],
            "metadata": {
                "results_count": 1,
                "search_time": 0.2,
                "top_score": 0.85,
                "avg_score": 0.85
            }
        }
        
        ui.show_search_details(workflow)
        
        # Should display search results
        mock_streamlit.subheader.assert_called_with("🔍 Top Search Results")
    
    def test_reasoning_details_display(self, ui, mock_streamlit):
        """Test reasoning details display."""
        workflow = {
            "llm_reasoning": "This is a test reasoning response",
            "metadata": {
                "reasoning_time": 2.5,
                "reasoning_length": 35
            }
        }
        
        ui.show_reasoning_details(workflow)
        
        # Should display reasoning metrics
        mock_streamlit.metric.assert_called()
        mock_streamlit.write.assert_called()

if __name__ == "__main__":
    pytest.main([__file__])

