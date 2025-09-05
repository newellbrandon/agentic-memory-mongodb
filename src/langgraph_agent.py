"""
LangGraph Agent for MongoDB + Voyage AI + Local LLM Integration
Demonstrates complete flow: Query → Embedding → Vector Search → LLM Reasoning → Response
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import our services
from src.services.core_service import CoreService
from src.services.llm_service import LLMService
from src.services.search_service import SearchService

load_dotenv()

# State definition for the LangGraph workflow
class AgentState(TypedDict):
    """State for the LangGraph agent workflow."""
    user_query: str
    user_id: str
    query_embedding: Optional[List[float]]
    search_results: List[Dict[str, Any]]
    search_scores: List[float]
    context_for_llm: str
    llm_reasoning: str
    final_response: str
    workflow_steps: List[str]
    metadata: Dict[str, Any]

class LangGraphAgent:
    """
    LangGraph agent that demonstrates the complete AI workflow:
    1. User query processing
    2. Embedding generation with Voyage AI
    3. Vector search in MongoDB Atlas
    4. LLM reasoning on search results
    5. Response generation
    """
    
    def __init__(self):
        """Initialize the LangGraph agent with all services."""
        try:
            self.core_service = CoreService()
            self.llm_service = LLMService()
            self.search_service = SearchService()
            
            # Create the workflow graph
            self.workflow = self._create_workflow()
            
            print("✅ LangGraph Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LangGraph Agent: {e}")
            raise
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_query", self._process_user_query)
        workflow.add_node("optimize_query", self._optimize_search_query)
        workflow.add_node("generate_embedding", self._generate_embedding)
        workflow.add_node("search_mongodb", self._search_mongodb)
        workflow.add_node("rerank_results", self._rerank_results)
        workflow.add_node("analyze_results", self._analyze_search_results)
        workflow.add_node("llm_reasoning", self._llm_reasoning)
        workflow.add_node("generate_response", self._generate_final_response)
        workflow.add_node("store_conversation", self._check_and_store_conversation)
        workflow.add_node("extract_personal_info", self._extract_personal_info)
        
        # Define the workflow flow
        workflow.set_entry_point("process_query")
        workflow.add_edge("process_query", "optimize_query")
        workflow.add_edge("optimize_query", "generate_embedding")
        workflow.add_edge("generate_embedding", "search_mongodb")
        workflow.add_edge("search_mongodb", "rerank_results")
        workflow.add_edge("rerank_results", "analyze_results")
        workflow.add_edge("analyze_results", "llm_reasoning")
        workflow.add_edge("llm_reasoning", "generate_response")
        workflow.add_edge("generate_response", "store_conversation")
        workflow.add_edge("store_conversation", "extract_personal_info")
        workflow.add_edge("extract_personal_info", END)
        
        # Compile the workflow
        return workflow.compile()
    
    async def _process_user_query(self, state: AgentState) -> AgentState:
        """Process the user query and prepare for embedding."""
        print("\n" + "="*80)
        print("🔍 STEP 1: PROCESSING USER QUERY")
        print("="*80)
        
        query = state["user_query"]
        user_id = state["user_id"]
        
        print(f"📝 User Query: {query}")
        print(f"👤 User ID: {user_id}")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        
        # Add to workflow steps
        state["workflow_steps"].append("query_processed")
        state["metadata"]["query_length"] = len(query)
        state["metadata"]["query_timestamp"] = datetime.now().isoformat()
        
        print("✅ Query processed successfully")
        return state
    
    async def _optimize_search_query(self, state: AgentState) -> AgentState:
        """Use LLM to optimize the search query for better results."""
        print("\n" + "="*80)
        print("🔍 STEP 2: OPTIMIZING SEARCH QUERY WITH LLM")
        print("="*80)
        
        user_query = state["user_query"]
        
        print(f"📝 Original Query: {user_query}")
        print(f"🤖 Optimizing query for better search results...")
        
        try:
            start_time = datetime.now()
            
            # Create optimization prompt
            optimization_prompt = f"""You are a search query optimization expert. Transform user queries into optimized search queries for a knowledge base containing documents, conversations, and personal information.

User Query: "{user_query}"

Create 1-3 optimized search queries that will find the most relevant information. Consider:
- What specific information is the user seeking?
- What keywords would appear in relevant documents?
- How might this be stored in conversations or personal data?

Return ONLY the optimized search queries, one per line, no explanations."""

            # Get optimized queries from LLM
            messages = [
                {"role": "system", "content": "You are a search query optimization expert. Return ONLY the optimized search queries, one per line. Do not include any reasoning, explanations, or formatting. Just the queries."},
                {"role": "user", "content": optimization_prompt}
            ]
            
            optimized_queries_text = await self.llm_service._call_lm_studio(messages, "", user_query)
            
            # Parse the optimized queries and filter out verbose responses
            lines = optimized_queries_text.strip().split('\n')
            optimized_queries = []
            
            for line in lines:
                line = line.strip()
                # Skip lines that look like reasoning or explanations
                if (line and 
                    not line.startswith('<think>') and 
                    not line.startswith('</think>') and
                    not line.startswith('Okay,') and
                    not line.startswith('Looking at') and
                    not line.startswith('Wait,') and
                    not line.startswith('I need to') and
                    not line.startswith('So the') and
                    not line.startswith('The user') and
                    not line.startswith('Your') and
                    not line.startswith('This') and
                    len(line) < 200):  # Skip very long lines
                    optimized_queries.append(line)
            
            # Use the first optimized query as primary, or fallback to original
            primary_query = optimized_queries[0] if optimized_queries else user_query
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Query optimization completed")
            print(f"📝 Optimized Query: {primary_query}")
            print(f"📋 Alternative Queries: {optimized_queries[1:] if len(optimized_queries) > 1 else 'None'}")
            print(f"⏱️  Optimization time: {optimization_time:.3f} seconds")
            
            # Store in state
            state["optimized_query"] = primary_query
            state["alternative_queries"] = optimized_queries[1:] if len(optimized_queries) > 1 else []
            state["workflow_steps"].append("query_optimized")
            state["metadata"]["optimization_time"] = optimization_time
            state["metadata"]["alternative_queries_count"] = len(optimized_queries) - 1
            
        except Exception as e:
            print(f"❌ Query optimization failed: {str(e)}")
            print(f"🔄 Falling back to original query")
            state["optimized_query"] = user_query
            state["alternative_queries"] = []
            state["workflow_steps"].append("query_optimization_failed")
            state["metadata"]["optimization_error"] = str(e)
        
        return state
    
    async def _generate_embedding(self, state: AgentState) -> AgentState:
        """Generate embedding using Voyage AI."""
        print("\n" + "="*80)
        print("🧠 STEP 3: GENERATING EMBEDDING WITH VOYAGE AI")
        print("="*80)
        
        # Use optimized query if available, otherwise use original
        query = state.get("optimized_query", state["user_query"])
        
        print(f"🔗 Voyage AI Model: {self.llm_service.voyage_model}")
        print(f"📝 Generating embedding for: {query[:100]}...")
        
        try:
            # Generate embedding
            start_time = datetime.now()
            embedding = await self.llm_service.generate_embedding(query)
            end_time = datetime.now()
            
            # Calculate timing
            embedding_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Embedding generated successfully")
            print(f"📊 Embedding dimensions: {len(embedding)}")
            print(f"⏱️  Generation time: {embedding_time:.3f} seconds")
            print(f"🔢 First 5 values: {embedding[:5]}")
            print(f"🔢 Last 5 values: {embedding[-5:]}")
            
            # Store in state
            state["query_embedding"] = embedding
            state["workflow_steps"].append("embedding_generated")
            state["metadata"]["embedding_dimensions"] = len(embedding)
            state["metadata"]["embedding_time"] = embedding_time
            
        except Exception as e:
            print(f"❌ Failed to generate embedding: {str(e)}")
            # Use zero vector as fallback
            state["query_embedding"] = [0.0] * 1024
            state["workflow_steps"].append("embedding_failed")
            state["metadata"]["embedding_error"] = str(e)
        
        return state
    
    async def _rerank_results(self, state: AgentState) -> AgentState:
        """Step 5: Rerank search results using Voyage AI reranking model."""
        print("\n" + "="*80)
        print("🔄 STEP 5: RERANKING SEARCH RESULTS")
        print("="*80)
        
        try:
            start_time = datetime.now()
            
            # Get the search results and query
            search_results = state.get("search_results", [])
            user_query = state.get("user_query", "")
            
            if not search_results:
                print("⚠️ No search results to rerank")
                state["workflow_steps"].append("Rerank Results (Skipped - No Results)")
                return state
            
            print(f"🔄 Reranking {len(search_results)} search results...")
            print(f"🔗 Rerank Model: {self.llm_service.voyage_rerank_model}")
            
            # Rerank the results
            reranked_results = await self.llm_service.rerank_results(
                query=user_query,
                documents=search_results,
                top_k=10
            )
            
            end_time = datetime.now()
            rerank_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Reranking completed successfully")
            print(f"📊 Reranked results: {len(reranked_results)}")
            print(f"⏱️  Rerank time: {rerank_time:.3f} seconds")
            
            # Show top reranked results
            if reranked_results:
                print(f"\n📋 Top reranked results:")
                for i, result in enumerate(reranked_results[:5], 1):
                    content_type = result.get("content_type", "unknown")
                    rerank_score = result.get("rerank_score", 0.0)
                    original_score = result.get("score", 0.0)
                    content_preview = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
                    
                    print(f"   {i}. [{content_type:12}] Rerank: {rerank_score:.4f} | Original: {original_score:.4f} | {content_preview}")
            
            # Update state with reranked results
            state["search_results"] = reranked_results
            state["workflow_steps"].append("Rerank Results (Voyage AI)")
            state["metadata"]["rerank_time"] = rerank_time
            state["metadata"]["rerank_model"] = self.llm_service.voyage_rerank_model
            
            return state
            
        except Exception as e:
            print(f"❌ Reranking failed: {str(e)}")
            state["workflow_steps"].append("Rerank Results (Failed)")
            return state

    async def _search_mongodb(self, state: AgentState) -> AgentState:
        """Search MongoDB Atlas using vector similarity."""
        print("\n" + "="*80)
        print("🗄️  STEP 4: VECTOR SEARCH IN MONGODB ATLAS")
        print("="*80)
        
        embedding = state["query_embedding"]
        query = state["user_query"]
        
        if not embedding:
            print("❌ No embedding available for search")
            state["search_results"] = []
            state["search_scores"] = []
            return state
        
        print(f"🔍 Searching MongoDB collection: {self.core_service.memory_collection.name}")
        print(f"🗃️  Database: {self.core_service.database}")
        print(f"🔗 Connection: {self.core_service.client.address}")
        
        try:
            start_time = datetime.now()
            
            # Search for relevant memories with document prioritization
            # For document-related queries, prioritize document content
            query_lower = query.lower()
            document_keywords = ['document', 'accelerando', 'story', 'book', 'text', 'content', 'chapter', 'novel']
            is_document_query = any(keyword in query_lower for keyword in document_keywords)
            
            if is_document_query:
                print("📚 Detected document-related query - prioritizing document content")
                # For document queries, search documents first, then conversations
                document_results = await self.core_service.search_memories(
                    embedding,
                    content_types=['document'],
                    limit=8
                )
                conversation_results = await self.core_service.search_memories(
                    embedding,
                    content_types=['conversation', 'personal_info'],
                    limit=7
                )
                # Combine results with documents first
                search_results = document_results + conversation_results
            else:
                # Regular search for other types of queries
                search_results = await self.core_service.search_memories(
                    embedding,
                    content_types=['document', 'conversation', 'personal_info'],
                    limit=10
                )
            
            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Search completed successfully")
            print(f"📊 Results found: {len(search_results)}")
            print(f"⏱️  Search time: {search_time:.3f} seconds")
            
            # Extract scores and results
            scores = []
            results = []
            
            for i, result in enumerate(search_results):
                score = result.get("score", 0.0)
                content_type = result.get("content_type", "unknown")
                content_preview = result.get("content", "")[:100] + "..."
                metadata = result.get("metadata", {})
                timestamp = metadata.get("timestamp", "Unknown")
                
                scores.append(score)
                results.append({
                    "rank": i + 1,
                    "score": score,
                    "content_type": content_type,
                    "content_preview": content_preview,
                    "full_content": result.get("content", ""),
                    "metadata": metadata,
                    "timestamp": timestamp
                })
                
                print(f"  {i+1:2d}. [{content_type:15s}] Score: {score:.4f} | {content_preview}")
            
            # Store in state
            state["search_results"] = results
            state["search_scores"] = scores
            state["workflow_steps"].append("mongodb_search_completed")
            state["metadata"]["search_time"] = search_time
            state["metadata"]["results_count"] = len(results)
            state["metadata"]["top_score"] = max(scores) if scores else 0.0
            state["metadata"]["avg_score"] = sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            print(f"❌ MongoDB search failed: {str(e)}")
            state["search_results"] = []
            state["search_scores"] = []
            state["workflow_steps"].append("mongodb_search_failed")
            state["metadata"]["search_error"] = str(e)
        
        return state
    
    async def _analyze_search_results(self, state: AgentState) -> AgentState:
        """Analyze and prepare search results for LLM reasoning."""
        print("\n" + "="*80)
        print("📊 STEP 6: ANALYZING SEARCH RESULTS")
        print("="*80)
        
        results = state["search_results"]
        scores = state["search_scores"]
        
        if not results:
            print("⚠️  No search results to analyze")
            state["context_for_llm"] = "No relevant information found in the knowledge base."
            return state
        
        print(f"📈 Analyzing {len(results)} search results...")
        
        # Analyze score distribution
        if scores:
            print(f"🏆 Top score: {max(scores):.4f}")
            print(f"📊 Average score: {sum(scores) / len(scores):.4f}")
            print(f"📉 Lowest score: {min(scores):.4f}")
        
        # Group results by content type
        content_type_counts = {}
        for result in results:
            content_type = result["content_type"]
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        print(f"📋 Content type distribution:")
        for content_type, count in content_type_counts.items():
            print(f"  - {content_type}: {count} results")
        
        # Build context for LLM with content type prioritization
        context_parts = []
        context_parts.append(f"SEARCH RESULTS FOR: {state['user_query']}")
        context_parts.append(f"Found {len(results)} relevant pieces of information:")
        
        # Separate results by content type for better organization
        documents = [r for r in results if r["content_type"] == "document"]
        conversations = [r for r in results if r["content_type"] == "conversation"]
        other_content = [r for r in results if r["content_type"] not in ["document", "conversation"]]
        
        # Prioritize document content first
        if documents:
            context_parts.append(f"\n📚 DOCUMENT CONTENT ({len(documents)} results):")
            for i, result in enumerate(documents[:3]):  # Top 3 document results
                rank = result["rank"]
                score = result["score"]
                content = result["content_preview"]
                metadata = result.get("metadata", {})
                
                # Add document metadata if available
                doc_info = ""
                if metadata.get("document_title"):
                    doc_info = f" - {metadata['document_title']}"
                elif metadata.get("document_id"):
                    doc_info = f" - Document {metadata['document_id']}"
                
                context_parts.append(f"\n{rank}. [DOCUMENT] (Relevance: {score:.4f}){doc_info}")
                context_parts.append(f"   {content}")
        
        # Add conversation context if available
        if conversations:
            context_parts.append(f"\n💬 CONVERSATION CONTEXT ({len(conversations)} results):")
            for i, result in enumerate(conversations[:2]):  # Top 2 conversation results
                rank = result["rank"]
                score = result["score"]
                content = result["content_preview"]
                timestamp = result.get("timestamp", "Unknown")
                
                context_parts.append(f"\n{rank}. [CONVERSATION] (Relevance: {score:.4f})")
                context_parts.append(f"   Time: {timestamp}")
                context_parts.append(f"   Content: {content}")
        
        # Add other content types (including personal_info)
        if other_content:
            context_parts.append(f"\n📝 OTHER CONTENT ({len(other_content)} results):")
            for i, result in enumerate(other_content[:2]):  # Top 2 other results
                rank = result["rank"]
                score = result["score"]
                content_type = result["content_type"]
                content = result["content_preview"]
                timestamp = result.get("timestamp", "Unknown")
                
                context_parts.append(f"\n{rank}. [{content_type.upper()}] (Relevance: {score:.4f})")
                context_parts.append(f"   Time: {timestamp}")
                context_parts.append(f"   Content: {content}")
        
        # Add score analysis with content type breakdown
        if scores:
            avg_score = sum(scores) / len(scores)
            context_parts.append(f"\n📊 RELEVANCE ANALYSIS:")
            context_parts.append(f"Overall average relevance: {avg_score:.4f}")
            
            if documents:
                doc_scores = [r["score"] for r in documents]
                avg_doc_score = sum(doc_scores) / len(doc_scores)
                context_parts.append(f"Document content relevance: {avg_doc_score:.4f}")
            
            if avg_score > 0.7:
                context_parts.append("✅ This indicates highly relevant information was found.")
            elif avg_score > 0.5:
                context_parts.append("⚠️  This indicates moderately relevant information was found.")
            else:
                context_parts.append("❌ This indicates the information may be less relevant to your query.")
        
        context_for_llm = "\n".join(context_parts)
        
        print(f"📝 Context prepared for LLM ({len(context_for_llm)} characters)")
        
        # Store in state
        state["context_for_llm"] = context_for_llm
        state["workflow_steps"].append("results_analyzed")
        state["metadata"]["context_length"] = len(context_for_llm)
        state["metadata"]["content_type_distribution"] = content_type_counts
        
        return state
    
    async def _llm_reasoning(self, state: AgentState) -> AgentState:
        """Use local LLM to reason about the search results."""
        print("\n" + "="*80)
        print("🤖 STEP 7: LLM REASONING ON SEARCH RESULTS")
        print("="*80)
        
        query = state["user_query"]
        context = state["context_for_llm"]
        
        print(f"🔗 LM Studio URL: {self.llm_service.lm_studio_base_url}")
        print(f"🤖 Model: {self.llm_service.lm_studio_model}")
        print(f"📝 User Query: {query}")
        print(f"📊 Context Length: {len(context)} characters")
        
        try:
            start_time = datetime.now()
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": """You are a concise AI assistant. You MUST follow these rules:

1. CHRONOLOGICAL PRIORITY: Use the most recent timestamp when information conflicts.
2. CONCISE RESPONSES: Maximum 1-2 sentences. No explanations.
3. NO THINKING TAGS: Do not use <think> tags or show reasoning.
4. DIRECT ANSWERS: Answer the question directly.

EXAMPLES:
- Question: "Where are my keys?"
- Results: "Keys under bed (Time: 2025-01-01)" and "Keys in car (Time: 2025-01-02)"
- Answer: "Your keys are in the car."

- Question: "What is my name?"
- Results: "Name: John (Time: 2025-01-01)" and "Name: Alice (Time: 2025-01-02)"
- Answer: "Your name is Alice."

RESPOND WITH ONLY THE ANSWER. NO EXPLANATIONS. NO THINKING TAGS."""},
                {"role": "user", "content": f"Question: {query}\n\nSearch Results:\n{context}\n\nAnswer directly. Use most recent timestamp if conflicts exist."}
            ]
            
            # Generate reasoning
            reasoning = await self.llm_service.generate_response(messages, context)
            
            end_time = datetime.now()
            reasoning_time = (end_time - start_time).total_seconds()
            
            print(f"✅ LLM reasoning completed successfully")
            print(f"⏱️  Reasoning time: {reasoning_time:.3f} seconds")
            print(f"📝 Reasoning length: {len(reasoning)} characters")
            print(f"🧠 Reasoning preview: {reasoning[:200]}...")
            
            # Store in state
            state["llm_reasoning"] = reasoning
            state["workflow_steps"].append("llm_reasoning_completed")
            state["metadata"]["reasoning_time"] = reasoning_time
            state["metadata"]["reasoning_length"] = len(reasoning)
            state["metadata"]["lm_model"] = self.llm_service.lm_studio_model
            state["metadata"]["lm_studio_url"] = self.llm_service.lm_studio_base_url
            
        except Exception as e:
            print(f"❌ LLM reasoning failed: {str(e)}")
            reasoning = f"I encountered an error while analyzing the search results: {str(e)}"
            state["llm_reasoning"] = reasoning
            state["workflow_steps"].append("llm_reasoning_failed")
            state["metadata"]["reasoning_error"] = str(e)
        
        return state
    
    async def _generate_final_response(self, state: AgentState) -> AgentState:
        """Generate the final response for the user."""
        print("\n" + "="*80)
        print("🎯 STEP 8: GENERATING FINAL RESPONSE")
        print("="*80)
        
        query = state["user_query"]
        reasoning = state["llm_reasoning"]
        results = state["search_results"]
        
        # Create a comprehensive final response
        response_parts = []
        response_parts.append(f"Based on my analysis of your query: '{query}'")
        
        if results:
            response_parts.append(f"\nI found {len(results)} relevant pieces of information in my knowledge base:")
            
            # Add top 3 results summary
            for i, result in enumerate(results[:3]):
                score = result["score"]
                content_type = result["content_type"]
                content = result["content_preview"]
                response_parts.append(f"\n• {content} (Source: {content_type}, Relevance: {score:.3f})")
        
        response_parts.append(f"\n\nMy Analysis:")
        response_parts.append(reasoning)
        
        response_parts.append(f"\n\n--- Workflow Summary ---")
        response_parts.append(f"✅ Query processed and embedded")
        response_parts.append(f"✅ Vector search completed in MongoDB Atlas")
        response_parts.append(f"✅ {len(results)} results analyzed")
        response_parts.append(f"✅ LLM reasoning completed")
        response_parts.append(f"✅ Final response generated")
        
        final_response = "\n".join(response_parts)
        
        print(f"✅ Final response generated successfully")
        print(f"📝 Response length: {len(final_response)} characters")
        print(f"📊 Workflow steps completed: {len(state['workflow_steps'])}")
        
        # Store in state
        state["final_response"] = final_response
        state["workflow_steps"].append("final_response_generated")
        state["metadata"]["final_response_length"] = len(final_response)
        state["metadata"]["total_workflow_steps"] = len(state["workflow_steps"])
        
        return state
    
    async def _check_and_store_conversation(self, state: AgentState) -> AgentState:
        """Check for duplicate conversations and store if unique."""
        print("\n" + "="*80)
        print("💾 STEP 9: CHECKING AND STORING CONVERSATION")
        print("="*80)
        
        user_query = state["user_query"]
        user_id = state["user_id"]
        final_response = state["final_response"]
        
        print(f"📝 Checking for duplicate conversation...")
        print(f"👤 User ID: {user_id}")
        
        try:
            # Check if this exact conversation already exists
            existing_conversation = self.core_service.memory_collection.find_one({
                "content_type": "conversation",
                "content": user_query,
                "metadata.user_id": user_id
            })
            
            if existing_conversation:
                print(f"⚠️  Duplicate conversation found - skipping storage")
                print(f"🔄 Conversation already exists in memory")
                state["workflow_steps"].append("conversation_duplicate_skipped")
                state["metadata"]["conversation_stored"] = False
                state["metadata"]["duplicate_detected"] = True
            else:
                # Store the conversation
                print(f"💾 Storing new conversation...")
                
                # Create conversation content
                conversation_content = f"User: {user_query}\nAssistant: {final_response}"
                
                # Generate embedding for the conversation
                conversation_embedding = await self.llm_service.generate_embedding(conversation_content)
                
                # Store in MongoDB
                conversation_id = await self.core_service.store_memory(
                    content=conversation_content,
                    content_type="conversation",
                    embedding=conversation_embedding,
                    metadata={
                        "user_id": user_id,
                        "user_query": user_query,
                        "assistant_response": final_response,
                        "timestamp": datetime.utcnow(),
                        "workflow_steps": len(state["workflow_steps"]),
                        "total_workflow_time": state["metadata"].get("total_workflow_time", 0)
                    }
                )
                
                print(f"✅ Conversation stored successfully")
                print(f"🆔 Conversation ID: {conversation_id}")
                state["workflow_steps"].append("conversation_stored")
                state["metadata"]["conversation_stored"] = True
                state["metadata"]["conversation_id"] = conversation_id
                state["metadata"]["duplicate_detected"] = False
                
        except Exception as e:
            print(f"❌ Failed to store conversation: {str(e)}")
            state["workflow_steps"].append("conversation_storage_failed")
            state["metadata"]["conversation_storage_error"] = str(e)
            state["metadata"]["conversation_stored"] = False
        
        return state
    
    async def _extract_personal_info(self, state: AgentState) -> AgentState:
        """Extract and store personal information from the conversation."""
        print("\n" + "="*80)
        print("👤 STEP 10: EXTRACTING PERSONAL INFORMATION")
        print("="*80)
        
        user_query = state["user_query"]
        final_response = state["final_response"]
        user_id = state["user_id"]
        
        print(f"🔍 Analyzing conversation for personal information...")
        
        try:
            # Create a prompt to extract personal information
            extraction_prompt = f"""Analyze the following conversation and extract any personal information about the user.

User Query: "{user_query}"
Assistant Response: "{final_response}"

Extract personal information such as:
- Name
- Age
- Location
- Preferences
- Interests
- Personal details
- Relationships
- Any other personal information

Return the information in this format:
NAME: [name if mentioned]
AGE: [age if mentioned]
LOCATION: [location if mentioned]
PREFERENCES: [preferences if mentioned]
OTHER: [any other personal information]

If no personal information is found, return: NO_PERSONAL_INFO"""

            messages = [
                {"role": "system", "content": "You are a personal information extraction expert. Extract only factual personal information mentioned in the conversation."},
                {"role": "user", "content": extraction_prompt}
            ]
            
            extracted_info = await self.llm_service._call_lm_studio(messages, "", user_query)
            
            if "NO_PERSONAL_INFO" not in extracted_info:
                # Clean up the extracted information - remove <think> tags and reasoning
                clean_info = extracted_info
                if "<think>" in clean_info:
                    # Extract only the content after </think>
                    parts = clean_info.split("</think>")
                    if len(parts) > 1:
                        clean_info = parts[1].strip()
                
                # Extract structured information
                structured_info = {}
                lines = clean_info.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('NAME:'):
                        structured_info['name'] = line.replace('NAME:', '').strip()
                    elif line.startswith('AGE:'):
                        structured_info['age'] = line.replace('AGE:', '').strip()
                    elif line.startswith('LOCATION:'):
                        structured_info['location'] = line.replace('LOCATION:', '').strip()
                    elif line.startswith('PREFERENCES:'):
                        structured_info['preferences'] = line.replace('PREFERENCES:', '').strip()
                    elif line.startswith('OTHER:'):
                        structured_info['other'] = line.replace('OTHER:', '').strip()
                
                # Create a clean summary
                if structured_info:
                    clean_summary = f"Personal Information: {', '.join([f'{k}: {v}' for k, v in structured_info.items() if v])}"
                else:
                    clean_summary = clean_info
                
                print(f"✅ Personal information extracted:")
                print(f"📝 {clean_summary}")
                
                # Generate embedding for the personal information
                personal_info_embedding = await self.llm_service.generate_embedding(clean_summary)
                
                # Store personal information
                personal_info_id = await self.core_service.store_memory(
                    content=clean_summary,
                    content_type="personal_info",
                    embedding=personal_info_embedding,
                    metadata={
                        "user_id": user_id,
                        "extracted_from_query": user_query,
                        "extracted_from_response": final_response,
                        "timestamp": datetime.utcnow(),
                        "extraction_method": "llm_analysis",
                        "structured_info": structured_info
                    }
                )
                
                print(f"💾 Personal information stored successfully")
                print(f"🆔 Personal Info ID: {personal_info_id}")
                state["workflow_steps"].append("personal_info_extracted")
                state["metadata"]["personal_info_extracted"] = True
                state["metadata"]["personal_info_id"] = personal_info_id
            else:
                print(f"ℹ️  No personal information found in conversation")
                state["workflow_steps"].append("no_personal_info_found")
                state["metadata"]["personal_info_extracted"] = False
                
        except Exception as e:
            print(f"❌ Failed to extract personal information: {str(e)}")
            state["workflow_steps"].append("personal_info_extraction_failed")
            state["metadata"]["personal_info_extraction_error"] = str(e)
            state["metadata"]["personal_info_extracted"] = False
        
        return state
    
    async def run_workflow(self, user_query: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Run the complete LangGraph workflow."""
        print("\n" + "🚀" + "="*78 + "🚀")
        print("🚀 STARTING LANGGRAPH AGENT WORKFLOW")
        print("🚀" + "="*78 + "🚀")
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            user_id=user_id,
            query_embedding=None,
            search_results=[],
            search_scores=[],
            context_for_llm="",
            llm_reasoning="",
            final_response="",
            workflow_steps=[],
            metadata={}
        )
        
        try:
            # Run the workflow
            start_time = datetime.now()
            final_state = await self.workflow.ainvoke(initial_state)
            end_time = datetime.now()
            
            total_time = (end_time - start_time).total_seconds()
            
            print("\n" + "🎉" + "="*78 + "🎉")
            print("🎉 LANGGRAPH WORKFLOW COMPLETED SUCCESSFULLY!")
            print("🎉" + "="*78 + "🎉")
            print(f"⏱️  Total workflow time: {total_time:.3f} seconds")
            print(f"📊 Workflow steps completed: {len(final_state['workflow_steps'])}")
            print(f"🔍 Search results found: {len(final_state['search_results'])}")
            print(f"🧠 LLM reasoning completed: {len(final_state['llm_reasoning'])} characters")
            print(f"📝 Final response: {len(final_state['final_response'])} characters")
            
            # Add timing to metadata
            final_state["metadata"]["total_workflow_time"] = total_time
            
            return {
                "success": True,
                "final_state": final_state,
                "workflow_summary": {
                    "total_time": total_time,
                    "steps_completed": len(final_state["workflow_steps"]),
                    "search_results_count": len(final_state["search_results"]),
                    "final_response_length": len(final_state["final_response"])
                }
            }
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "partial_state": initial_state
            }
    
    async def close(self):
        """Clean up resources."""
        try:
            self.core_service.close()
            self.llm_service.close()
            self.search_service.close()
            print("✅ LangGraph Agent resources cleaned up")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {str(e)}")

# Example usage
async def main():
    """Example usage of the LangGraph Agent."""
    try:
        # Initialize the agent
        agent = LangGraphAgent()
        
        # Example queries to test the workflow
        test_queries = [
            "What is the main topic of the Accelerando document?",
            "Tell me about the characters in the story",
            "What are the key themes discussed?",
            "Who is the protagonist and what happens to them?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"🧪 TEST QUERY {i}: {query}")
            print(f"{'='*80}")
            
            # Run the workflow
            result = await agent.run_workflow(query, f"test_user_{i}")
            
            if result["success"]:
                print(f"✅ Test {i} completed successfully")
            else:
                print(f"❌ Test {i} failed: {result['error']}")
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Clean up
        await agent.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
