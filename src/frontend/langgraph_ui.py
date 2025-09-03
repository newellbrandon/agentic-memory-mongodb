"""
Streamlit UI for LangGraph Agent Workflow
Shows the complete flow: Query → Embedding → Vector Search → LLM Reasoning → Response
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our LangGraph agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph_agent import LangGraphAgent

from src.frontend.document_management_ui import DocumentManagementUI

class LangGraphUI:
    """Streamlit UI for the LangGraph Agent workflow."""
    
    def __init__(self):
        """Initialize the UI."""
        st.set_page_config(
            page_title="LangGraph Agent Workflow",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        if 'current_workflow' not in st.session_state:
            st.session_state.current_workflow = None
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        
        # Setup UI synchronously
        self.setup_ui_sync()
    
    def setup_ui_sync(self):
        """Setup the main UI components synchronously."""
        st.title("🚀 LangGraph Agent Workflow")
        st.markdown("**Complete AI Workflow: Query → Embedding → Vector Search → LLM Reasoning → Response**")
        
        # Sidebar for configuration
        self.setup_sidebar()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 LangGraph Agent", 
            "📚 Document Management",
            "📊 Memory Analytics", 
            "⚙️ Settings"
        ])

        with tab1:
            # Check for pending workflow execution
            if hasattr(st.session_state, 'pending_query') and st.session_state.pending_query:
                query = st.session_state.pending_query
                user_id = st.session_state.pending_user_id
                # Clear the pending state
                del st.session_state.pending_query
                del st.session_state.pending_user_id
                # Run the workflow
                self.run_workflow_sync(query, user_id)
            else:
                self._render_agent_tab()
        
        with tab2:
            self._render_document_management_tab_sync()
        
        with tab3:
            self._render_analytics_tab()
        
        with tab4:
            self._render_settings_tab()
    
    async def setup_ui(self):
        """Setup the main UI components."""
        st.title("🚀 LangGraph Agent Workflow")
        st.markdown("**Complete AI Workflow: Query → Embedding → Vector Search → LLM Reasoning → Response**")
        
        # Sidebar for configuration
        self.setup_sidebar()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 LangGraph Agent", 
            "📚 Document Management",
            "📊 Memory Analytics", 
            "⚙️ Settings"
        ])

        with tab1:
            self._render_agent_tab()
        
        with tab2:
            await self._render_document_management_tab()
        
        with tab3:
            self._render_analytics_tab()
        
        with tab4:
            self._render_settings_tab()
    
    def setup_sidebar(self):
        """Setup the sidebar configuration."""
        st.sidebar.header("⚙️ Configuration")
        
        # Connection status
        st.sidebar.subheader("🔗 Connection Status")
        
        if st.sidebar.button("🔄 Test Connections"):
            self.test_connections()
        
        # Workflow settings
        st.sidebar.subheader("⚙️ Workflow Settings")
        max_results = st.sidebar.slider("Max Search Results", 5, 20, 10)
        st.session_state.max_results = max_results
        
        # Display options
        st.sidebar.subheader("📊 Display Options")
        show_cli_details = st.sidebar.checkbox("Show Detailed CLI-Style Output", value=True, help="Show comprehensive step-by-step details like the CLI")
        show_embeddings = st.sidebar.checkbox("Show Embedding Details", value=True)
        show_timing = st.sidebar.checkbox("Show Timing Details", value=True)
        show_metadata = st.sidebar.checkbox("Show Metadata", value=False)
        
        st.session_state.show_cli_details = show_cli_details
        st.session_state.show_embeddings = show_embeddings
        st.session_state.show_timing = show_timing
        st.session_state.show_metadata = show_metadata
        
        # Clear history
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.workflow_history = []
            st.rerun()
    
    def setup_query_section(self):
        """Setup the query input section."""
        st.header("🔍 Query Input")
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            placeholder="Ask a question about the documents in your knowledge base...",
            height=100
        )
        
        # User ID input
        user_id = st.text_input("User ID (optional):", value="default_user")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🚀 Run Workflow", type="primary"):
                if query.strip():
                    # Store the query and user_id for async processing
                    st.session_state.pending_query = query
                    st.session_state.pending_user_id = user_id
                    st.rerun()
                else:
                    st.error("Please enter a query first!")
        
        with col2:
            if st.button("📊 Show Stats"):
                self.show_workflow_stats()
        
        with col3:
            if st.button("🧪 Demo Queries"):
                self.run_demo_queries()
    
    def setup_workflow_display(self):
        """Setup the workflow display section."""
        st.header("🔄 Workflow Execution")
        
        if st.session_state.current_workflow:
            workflow = st.session_state.current_workflow
            
            # Workflow progress
            st.subheader("📈 Workflow Progress")
            
            # Create progress indicators
            steps = [
                "Query Processing",
                "Query Optimization",
                "Embedding Generation", 
                "MongoDB Search",
                "Results Analysis",
                "LLM Reasoning",
                "Response Generation",
                "Conversation Storage",
                "Personal Info Extraction"
            ]
            
            completed_steps = len(workflow.get('workflow_steps', []))
            
            # Progress bar - ensure progress is between 0.0 and 1.0
            progress = min(completed_steps / len(steps), 1.0)
            st.progress(progress, text=f"Step {completed_steps} of {len(steps)}")
            
            # Step indicators
            cols = st.columns(len(steps))
            for i, (col, step) in enumerate(zip(cols, steps)):
                if i < completed_steps:
                    col.success(f"✅ {step}")
                else:
                    col.info(f"⏳ {step}")
            
            # Current step details
            if workflow.get('workflow_steps'):
                st.subheader("📍 Current Step Details")
                
                current_step = workflow['workflow_steps'][-1] if workflow['workflow_steps'] else "None"
                st.info(f"**Current Step:** {current_step}")
                
                # Toggle for detailed steps
                show_details = st.checkbox("Show detailed workflow steps", value=False)
                if show_details:
                    # Show step-specific information
                    self.show_step_details(workflow)
                else:
                    st.info("💡 Check the box above to see detailed workflow execution steps")
            
            # Final response - Show ONLY the raw LLM response
            if workflow.get('llm_reasoning'):
                # Use the actual LLM reasoning as the response
                llm_response = workflow['llm_reasoning']
                
                # Check if there are <think> tags
                if '<think>' in llm_response and '</think>' in llm_response:
                    # Show expandable think content
                    self.show_response_with_expandable_thinking(llm_response)
                else:
                    # Show raw LLM response directly
                    st.markdown(llm_response)
            else:
                st.warning("⚠️ No LLM response available in workflow state")
        
        else:
            st.info("👆 Enter a query and click 'Run Workflow' to start the LangGraph agent workflow.")
    

    def show_response_with_expandable_thinking(self, text):
        """Show response with expandable thinking sections."""
        import re
        import streamlit as st
        
        # Find all <think> tags
        think_pattern = r'<think>(.*?)</think>'
        matches = list(re.finditer(think_pattern, text, flags=re.DOTALL))
        
        if not matches:
            st.markdown(text)
            return
        
        # Process the text and show expandable sections
        last_end = 0
        
        for i, match in enumerate(matches):
            # Add text before the think tag
            before_text = text[last_end:match.start()]
            if before_text.strip():
                st.markdown(before_text)
            
            # Extract think content
            think_content = match.group(1).strip()
            
            # Create expandable section for think content
            with st.expander(f"🧠 **Thinking Process {i+1}**", expanded=False):
                st.markdown(think_content)
            
            last_end = match.end()
        
        # Add remaining text after the last think tag
        remaining_text = text[last_end:]
        if remaining_text.strip():
            st.markdown(remaining_text)
    
    def process_think_tags(self, text):
        """Convert <think> tags to italics for better display."""
        import re
        
        # Replace <think>content</think> with *content* (italics)
        def replace_think_tag(match):
            content = match.group(1)
            # Clean up the content - remove extra whitespace and newlines
            content = content.strip()
            # Convert to italics
            return f"*{content}*"
        
        # Use regex to find and replace <think> tags
        processed_text = re.sub(r'<think>(.*?)</think>', replace_think_tag, text, flags=re.DOTALL)
        
        return processed_text
    
    def setup_metrics_display(self):
        """Setup the metrics display section."""
        st.header("📊 Workflow Metrics")
        
        if st.session_state.current_workflow:
            workflow = st.session_state.current_workflow
            metadata = workflow.get('metadata', {})
            
            # Key metrics
            st.metric("Total Time", f"{metadata.get('total_workflow_time', 0):.3f}s")
            st.metric("Search Results", metadata.get('results_count', 0))
            st.metric("Top Score", f"{metadata.get('top_score', 0):.4f}")
            st.metric("Avg Score", f"{metadata.get('avg_score', 0):.4f}")
            
            # Timing breakdown
            if st.session_state.show_timing:
                st.subheader("⏱️ Timing Breakdown")
                
                timing_data = {
                    "Embedding": metadata.get('embedding_time', 0),
                    "Search": metadata.get('search_time', 0),
                    "Reasoning": metadata.get('reasoning_time', 0)
                }
                
                if any(timing_data.values()):
                    fig = px.bar(
                        x=list(timing_data.keys()),
                        y=list(timing_data.values()),
                        title="Step Timing (seconds)",
                        color=list(timing_data.values()),
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Score distribution
            if workflow.get('search_scores'):
                st.subheader("📈 Score Distribution")
                
                scores = workflow['search_scores']
                if scores:
                    fig = px.histogram(
                        x=scores,
                        nbins=10,
                        title="Search Result Scores",
                        labels={'x': 'Similarity Score', 'y': 'Count'}
                    )
                    fig.add_vline(x=sum(scores)/len(scores), line_dash="dash", line_color="red", 
                                annotation_text=f"Mean: {sum(scores)/len(scores):.3f}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No workflow data to display.")
    
    def setup_workflow_history(self):
        """Setup the workflow history section."""
        st.header("📚 Workflow History")
        
        if st.session_state.workflow_history:
            # Show recent workflows
            for i, workflow in enumerate(reversed(st.session_state.workflow_history[-5:])):
                with st.expander(f"Workflow {len(st.session_state.workflow_history) - i}"):
                    st.write(f"**Query:** {workflow.get('user_query', 'N/A')}")
                    st.write(f"**User:** {workflow.get('user_id', 'N/A')}")
                    st.write(f"**Time:** {workflow.get('metadata', {}).get('query_timestamp', 'N/A')}")
                    st.write(f"**Results:** {workflow.get('metadata', {}).get('results_count', 0)}")
                    
                    if st.button(f"Load Workflow {len(st.session_state.workflow_history) - i}", key=f"load_{i}"):
                        st.session_state.current_workflow = workflow
                        st.rerun()
        else:
            st.info("No workflow history yet.")
    
    def show_step_details(self, workflow):
        """Show comprehensive details for all workflow steps."""
        # Check if user wants CLI-style details
        if st.session_state.get('show_cli_details', True):
            # Show detailed step-by-step execution like CLI
            st.subheader("🔍 Detailed Workflow Execution")
            
            # Step 1: Query Processing
            self.show_query_processing_details(workflow)
            
            # Step 2: Query Optimization
            self.show_query_optimization_details(workflow)
            
            # Step 3: Embedding Generation
            self.show_embedding_details(workflow)
            
            # Step 4: MongoDB Search
            self.show_search_details(workflow)
            
            # Step 5: Results Analysis
            self.show_analysis_details(workflow)
            
            # Step 6: LLM Reasoning
            self.show_reasoning_details(workflow)
            
            # Step 7: Response Generation
            self.show_response_details(workflow)
            
            # Step 8: Conversation Storage
            self.show_conversation_storage_details(workflow)
            
            # Step 9: Personal Info Extraction
            self.show_personal_info_extraction_details(workflow)
        else:
            # Show simplified view
            self.show_simplified_details(workflow)
    
    def show_query_processing_details(self, workflow):
        """Show query processing details like CLI."""
        with st.expander("🔍 STEP 1: PROCESSING USER QUERY", expanded=True):
            st.write(f"📝 **User Query:** {workflow.get('user_query', 'N/A')}")
            st.write(f"👤 **User ID:** {workflow.get('user_id', 'N/A')}")
            st.write(f"⏰ **Timestamp:** {workflow.get('timestamp', 'N/A')}")
            st.success("✅ Query processed successfully")

    def show_query_optimization_details(self, workflow):
        """Show query optimization details like CLI."""
        with st.expander("🔍 STEP 2: OPTIMIZING SEARCH QUERY WITH LLM", expanded=True):
            metadata = workflow.get('metadata', {})
            optimized_query = workflow.get('optimized_query', '')
            alternative_queries = workflow.get('alternative_queries', [])
            
            st.write(f"📝 **Original Query:** {workflow.get('user_query', 'N/A')}")
            st.write("🤖 **Optimizing query for better search results...**")
            st.success("✅ Query optimization completed")
            
            if optimized_query:
                st.write(f"📝 **Optimized Query:** {optimized_query}")
            
            if alternative_queries:
                st.write(f"📋 **Alternative Queries:** {', '.join(alternative_queries)}")
            else:
                st.write("📋 **Alternative Queries:** None")
            
            if 'optimization_time' in metadata:
                st.write(f"⏱️ **Optimization time:** {metadata['optimization_time']:.3f} seconds")

    def show_embedding_details(self, workflow):
        """Show embedding generation details like CLI."""
        with st.expander("🧠 STEP 3: GENERATING EMBEDDING WITH VOYAGE AI", expanded=True):
            metadata = workflow.get('metadata', {})
            
            st.write(f"🔗 **Voyage AI Model:** {metadata.get('voyage_model', 'voyage-large-2-instruct')}")
            st.write(f"📝 **Generating embedding for:** {workflow.get('user_query', 'N/A')}...")
            st.success("✅ Embedding generated successfully")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Embedding dimensions", metadata.get('embedding_dimensions', 0))
                st.metric("⏱️ Generation time", f"{metadata.get('embedding_time', 0):.3f} seconds")
            
            with col2:
                if workflow.get('query_embedding') and st.session_state.show_embeddings:
                    embedding = workflow['query_embedding']
                    if len(embedding) > 5:
                        st.write("🔢 **First 5 values:**", embedding[:5])
                        st.write("🔢 **Last 5 values:**", embedding[-5:])
    
    def show_search_details(self, workflow):
        """Show MongoDB search details like CLI."""
        with st.expander("🗄️ STEP 4: VECTOR SEARCH IN MONGODB ATLAS", expanded=True):
            metadata = workflow.get('metadata', {})
            results = workflow.get('search_results', [])
            
            st.write("🔍 **Searching MongoDB collection:** memory")
            st.write("🗃️ **Database:** personal_ai")
            st.write("🔗 **Connection:** MongoDB Atlas")
            
            # Show document prioritization info if available
            if metadata.get('is_document_query'):
                st.info("📚 **Detected document-related query - prioritizing document content**")
            
            if results:
                st.success("✅ Search completed successfully")
                st.write(f"📊 **Results found:** {len(results)}")
                st.write(f"⏱️ **Search time:** {metadata.get('search_time', 0):.3f} seconds")
                
                # Show all results like CLI
                st.subheader("🔍 Search Results")
                for i, result in enumerate(results, 1):
                    content_preview = result.get('content_preview', result.get('content', ''))[:100] + "..."
                    st.write(f"   {i:2d}. [{result.get('content_type', 'unknown'):<12}] Score: {result.get('score', 0):.4f} | {content_preview}")
            else:
                st.warning("⚠️ No search results found")
    
    def show_analysis_details(self, workflow):
        """Show results analysis details like CLI."""
        with st.expander("📊 STEP 5: ANALYZING SEARCH RESULTS", expanded=True):
            results = workflow.get('search_results', [])
            metadata = workflow.get('metadata', {})
            
            if results:
                st.write(f"📈 **Analyzing {len(results)} search results...**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Top score", f"{metadata.get('top_score', 0):.4f}")
                with col2:
                    st.metric("📊 Average score", f"{metadata.get('avg_score', 0):.4f}")
                with col3:
                    st.metric("📉 Lowest score", f"{metadata.get('lowest_score', 0):.4f}")
                
                # Content type distribution
                content_types = {}
                for result in results:
                    content_type = result.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                
                st.write("📋 **Content type distribution:**")
                for content_type, count in content_types.items():
                    st.write(f"  - {content_type}: {count} results")
                
                st.write(f"📝 **Context prepared for LLM** ({metadata.get('context_length', 0)} characters)")
            else:
                st.warning("⚠️ No search results to analyze")
    
    def show_reasoning_details(self, workflow):
        """Show LLM reasoning details like CLI."""
        with st.expander("🤖 STEP 6: LLM REASONING ON SEARCH RESULTS", expanded=True):
            metadata = workflow.get('metadata', {})
            reasoning = workflow.get('llm_reasoning', '')
            
            st.write(f"🔗 **LM Studio URL:** {metadata.get('lm_studio_url', 'http://localhost:1234/v1')}")
            # Get the actual model from the LLM service
            actual_model = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b-2507")
            st.write(f"🤖 **Model:** {metadata.get('lm_model', actual_model)}")
            st.write(f"📝 **User Query:** {workflow.get('user_query', 'N/A')}")
            st.write(f"📊 **Context Length:** {metadata.get('context_length', 0)} characters")
            
            if reasoning:
                st.success("✅ LLM reasoning completed successfully")
                st.write(f"⏱️ **Reasoning time:** {metadata.get('reasoning_time', 0):.3f} seconds")
                st.write(f"📝 **Reasoning length:** {len(reasoning)} characters")
                
                st.write("🧠 **Reasoning preview:**")
                processed_reasoning = self.process_think_tags(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
                st.markdown(processed_reasoning)
            else:
                st.warning("⚠️ No reasoning available")
    
    def show_response_details(self, workflow):
        """Show response generation details like CLI."""
        with st.expander("🎯 STEP 7: GENERATING FINAL RESPONSE", expanded=True):
            metadata = workflow.get('metadata', {})
            response = workflow.get('final_response', '')
            
            if response:
                st.success("✅ Final response generated successfully")
                st.write(f"📝 **Response length:** {len(response)} characters")
                st.write(f"📊 **Workflow steps completed:** {metadata.get('workflow_steps_completed', 0)}")
                
                # Show completion summary
                st.subheader("🎉 LANGGRAPH WORKFLOW COMPLETED SUCCESSFULLY!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Total workflow time", f"{metadata.get('total_workflow_time', 0):.3f}s")
                with col2:
                    st.metric("📊 Workflow steps", metadata.get('workflow_steps_completed', 0))
                with col3:
                    st.metric("🔍 Search results", metadata.get('results_count', 0))
                with col4:
                    st.metric("🧠 LLM reasoning", f"{len(workflow.get('llm_reasoning', ''))} chars")
                
                st.write(f"📝 **Final response:** {len(response)} characters")
            else:
                st.warning("⚠️ No final response available")

    def show_conversation_storage_details(self, workflow):
        """Show conversation storage details like CLI."""
        with st.expander("💾 STEP 8: CHECKING AND STORING CONVERSATION", expanded=True):
            metadata = workflow.get('metadata', {})
            user_id = workflow.get('user_id', 'N/A')
            
            st.write("📝 **Checking for duplicate conversation...**")
            st.write(f"👤 **User ID:** {user_id}")
            
            if metadata.get('conversation_stored', False):
                st.success("✅ Conversation stored successfully")
                if 'conversation_id' in metadata:
                    st.write(f"🆔 **Conversation ID:** {metadata['conversation_id']}")
            else:
                st.info("ℹ️ Duplicate conversation found. Skipping storage.")

    def show_personal_info_extraction_details(self, workflow):
        """Show personal info extraction details like CLI."""
        with st.expander("👤 STEP 9: EXTRACTING PERSONAL INFORMATION", expanded=True):
            metadata = workflow.get('metadata', {})
            
            st.write("🔍 **Analyzing conversation for personal information...**")
            
            if metadata.get('personal_info_extracted', False):
                st.success("✅ Personal information extracted")
                if 'personal_info_id' in metadata:
                    st.write(f"🆔 **Personal Info ID:** {metadata['personal_info_id']}")
            else:
                st.info("ℹ️ No personal information found.")
    
    def show_simplified_details(self, workflow):
        """Show simplified workflow details."""
        st.subheader("📊 Workflow Summary")
        
        metadata = workflow.get('metadata', {})
        results = workflow.get('search_results', [])
        
        # Key metrics in a compact format
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Total Time", f"{metadata.get('total_workflow_time', 0):.3f}s")
        with col2:
            st.metric("🔍 Results Found", len(results))
        with col3:
            st.metric("🏆 Top Score", f"{metadata.get('top_score', 0):.4f}")
        with col4:
            st.metric("📝 Response Length", f"{len(workflow.get('final_response', ''))} chars")
        
        # Show top 3 results
        if results:
            st.subheader("🔍 Top Search Results")
            for i, result in enumerate(results[:3], 1):
                st.write(f"**{i}.** [{result.get('content_type', 'unknown')}] Score: {result.get('score', 0):.4f}")
                st.write(f"   {result.get('content_preview', '')[:100]}...")
                st.write("---")
    
    def test_connections(self):
        """Test all service connections."""
        st.sidebar.info("Testing connections...")
        
        try:
            # Test MongoDB connection
            agent = LangGraphAgent()
            st.sidebar.success("✅ All connections successful!")
            st.session_state.agent = agent
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
    
    def run_workflow_sync(self, query: str, user_id: str):
        """Run the LangGraph workflow synchronously."""
        try:
            # Initialize agent if not exists
            if not st.session_state.agent:
                st.session_state.agent = LangGraphAgent()
            
            # Show progress
            with st.spinner("Running LangGraph workflow..."):
                # Run the workflow using asyncio.run
                import asyncio
                result = asyncio.run(st.session_state.agent.run_workflow(query, user_id))
                
                if result["success"]:
                    # Store in history
                    workflow_data = result["final_state"]
                    workflow_data["user_query"] = query
                    workflow_data["user_id"] = user_id
                    workflow_data["timestamp"] = datetime.now()
                    
                    st.session_state.workflow_history.append(workflow_data)
                    st.session_state.current_workflow = workflow_data
                    
                    st.success("✅ Workflow completed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Workflow failed: {result['error']}")
        
        except Exception as e:
            st.error(f"❌ Error running workflow: {str(e)}")
    
    async def run_workflow(self, query: str, user_id: str):
        """Run the LangGraph workflow."""
        try:
            # Initialize agent if not exists
            if not st.session_state.agent:
                st.session_state.agent = LangGraphAgent()
            
            # Show progress
            with st.spinner("Running LangGraph workflow..."):
                # Run the workflow
                result = await st.session_state.agent.run_workflow(query, user_id)
                
                if result["success"]:
                    # Store in history
                    workflow_data = result["final_state"]
                    workflow_data["user_query"] = query
                    workflow_data["user_id"] = user_id
                    
                    st.session_state.workflow_history.append(workflow_data)
                    st.session_state.current_workflow = workflow_data
                    
                    st.success("✅ Workflow completed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Workflow failed: {result['error']}")
        
        except Exception as e:
            st.error(f"❌ Error running workflow: {str(e)}")
    
    def show_workflow_stats(self):
        """Show workflow statistics."""
        if not st.session_state.workflow_history:
            st.info("No workflow history to analyze.")
            return
        
        st.header("📊 Workflow Statistics")
        
        # Calculate stats
        total_workflows = len(st.session_state.workflow_history)
        total_time = sum(w.get('metadata', {}).get('total_workflow_time', 0) for w in st.session_state.workflow_history)
        avg_time = total_time / total_workflows if total_workflows > 0 else 0
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Workflows", total_workflows)
        col2.metric("Total Time", f"{total_time:.3f}s")
        col3.metric("Avg Time", f"{avg_time:.3f}s")
        
        # Create timeline
        if total_workflows > 1:
            st.subheader("📈 Workflow Timeline")
            
            timeline_data = []
            for workflow in st.session_state.workflow_history:
                timestamp = workflow.get('metadata', {}).get('query_timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timeline_data.append({
                            'timestamp': dt,
                            'query': workflow.get('user_query', '')[:50] + '...',
                            'time': workflow.get('metadata', {}).get('total_workflow_time', 0)
                        })
                    except:
                        pass
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                fig = px.scatter(
                    df, 
                    x='timestamp', 
                    y='time',
                    title="Workflow Execution Timeline",
                    labels={'time': 'Execution Time (s)', 'timestamp': 'Timestamp'},
                    hover_data=['query']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def run_demo_queries(self):
        """Run a set of demo queries."""
        demo_queries = [
            "What is the main topic of the Accelerando document?",
            "Tell me about the characters in the story",
            "What are the key themes discussed?",
            "Who is the protagonist and what happens to them?"
        ]
        
        st.info("Running demo queries...")
        
        for i, query in enumerate(demo_queries):
            st.write(f"Demo {i+1}: {query}")
            self.run_workflow_sync(query, f"demo_user_{i+1}")
            time.sleep(1)  # Small delay between demos

    def _render_agent_tab(self):
        """Render the main agent workflow interface."""
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.setup_query_section()
            self.setup_workflow_display()
        
        with col2:
            self.setup_metrics_display()
            self.setup_workflow_history()
    
    def _render_document_management_tab_sync(self):
        """Render the document management interface synchronously."""
        doc_ui = DocumentManagementUI()
        # Call the synchronous render method directly
        try:
            doc_ui.render()
        except Exception as e:
            # If there's any error, show a simplified version
            st.info("🔄 Document management features are loading...")
            st.info("If this message persists, please refresh the page.")
            st.error(f"Error: {str(e)}")
            # Show a basic document management interface
            self._render_basic_document_management()
    
    def _render_basic_document_management(self):
        """Render a basic document management interface without async."""
        st.header("📚 Document Management")
        st.info("🔄 Full document management features are temporarily unavailable due to async constraints.")
        st.info("💡 You can still use the CLI for document management:")
        
        st.code("""
# List documents
python document_manager_cli.py list

# Upload a document
python document_manager_cli.py upload filename.txt

# Enable/disable documents
python document_manager_cli.py enable doc_id
python document_manager_cli.py disable doc_id

# Delete documents
python document_manager_cli.py delete doc_id

# Compress memories
python document_manager_cli.py compress

# Get statistics
python document_manager_cli.py stats
        """, language="bash")
        
        st.info("🔄 Please refresh the page to try loading the full interface again.")
    
    async def _render_document_management_tab(self):
        """Render the document management interface."""
        doc_ui = DocumentManagementUI()
        await doc_ui.render()
    
    def _render_analytics_tab(self):
        """Render the analytics interface."""
        st.header("📊 Memory Analytics")
        st.info("Analytics features coming soon...")
    
    def _render_settings_tab(self):
        """Render the settings interface."""
        st.header("⚙️ Settings")
        st.info("Settings features coming soon...")

def main():
    """Main function to run the Streamlit UI."""
    ui = LangGraphUI()

if __name__ == "__main__":
    main()
