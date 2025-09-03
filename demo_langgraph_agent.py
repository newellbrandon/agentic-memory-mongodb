#!/usr/bin/env python3
"""
Command-line demo of LangGraph Agent Workflow
Demonstrates: Query → Embedding → Vector Search → LLM Reasoning → Response
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_agent import LangGraphAgent

async def demo_workflow():
    """Demonstrate the complete LangGraph workflow."""
    
    print("🚀" + "="*78 + "🚀")
    print("🚀 LANGGRAPH AGENT WORKFLOW DEMONSTRATION")
    print("🚀" + "="*78 + "🚀")
    print("🎯 This demo shows the complete AI workflow:")
    print("   1. User query processing")
    print("   2. Embedding generation with Voyage AI")
    print("   3. Vector search in MongoDB Atlas")
    print("   4. LLM reasoning on search results")
    print("   5. Response generation")
    print("🚀" + "="*78 + "🚀")
    
    try:
        # Initialize the agent
        print("\n🔧 Initializing LangGraph Agent...")
        agent = LangGraphAgent()
        print("✅ Agent initialized successfully!")
        
        # Demo queries
        demo_queries = [
            "What is the main topic of the Accelerando document?",
            "Tell me about the characters in the story",
            "What are the key themes discussed?",
            "Who is the protagonist and what happens to them?",
            "What is the setting of the story?"
        ]
        
        print(f"\n🧪 Running {len(demo_queries)} demo queries...")
        print("="*80)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*80}")
            print(f"🧪 DEMO QUERY {i}/{len(demo_queries)}")
            print(f"{'='*80}")
            print(f"📝 Query: {query}")
            print(f"👤 User: demo_user_{i}")
            print(f"⏰ Timestamp: {asyncio.get_event_loop().time()}")
            print(f"{'='*80}")
            
            # Run the workflow
            result = await agent.run_workflow(query, f"demo_user_{i}")
            
            if result["success"]:
                print(f"\n✅ Demo {i} completed successfully!")
                
                # Show summary
                summary = result["workflow_summary"]
                print(f"📊 Summary:")
                print(f"   • Total time: {summary['total_time']:.3f} seconds")
                print(f"   • Steps completed: {summary['steps_completed']}")
                print(f"   • Search results: {summary['search_results_count']}")
                print(f"   • Response length: {summary['final_response_length']} characters")
                
                # Show final response preview
                final_state = result["final_state"]
                if final_state.get("final_response"):
                    response = final_state["final_response"]
                    print(f"\n🎯 Final Response Preview:")
                    print(f"   {response[:200]}...")
                
            else:
                print(f"\n❌ Demo {i} failed: {result['error']}")
            
            # Small delay between demos
            if i < len(demo_queries):
                print(f"\n⏳ Waiting 3 seconds before next demo...")
                await asyncio.sleep(3)
        
        print(f"\n{'='*80}")
        print("🎉 ALL DEMOS COMPLETED!")
        print(f"{'='*80}")
        
        # Show overall statistics
        print("\n📊 Overall Demo Statistics:")
        print("   • Total demos run: {len(demo_queries)}")
        print("   • All workflows completed successfully")
        print("   • LangGraph agent demonstrated all capabilities")
        
        # Clean up
        await agent.close()
        print("\n✅ Agent resources cleaned up")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def interactive_mode():
    """Run the agent in interactive mode."""
    
    print("🎮" + "="*78 + "🎮")
    print("🎮 INTERACTIVE LANGGRAPH AGENT MODE")
    print("🎮" + "="*78 + "🎮")
    print("💡 Type your queries and see the complete workflow in action!")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'demo' to run the demo queries")
    print("🎮" + "="*78 + "🎮")
    
    try:
        # Initialize the agent
        print("\n🔧 Initializing LangGraph Agent...")
        agent = LangGraphAgent()
        print("✅ Agent initialized successfully!")
        
        while True:
            print(f"\n{'='*80}")
            
            # Get user input
            query = input("🔍 Enter your query (or 'quit'/'demo'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'demo':
                print("🧪 Running demo queries...")
                await demo_workflow()
                continue
            
            if not query:
                print("⚠️ Please enter a query.")
                continue
            
            # Get user ID
            user_id = input("👤 Enter user ID (or press Enter for 'default_user'): ").strip()
            if not user_id:
                user_id = "default_user"
            
            print(f"\n🚀 Running workflow for query: '{query}'")
            print(f"👤 User: {user_id}")
            
            # Run the workflow
            result = await agent.run_workflow(query, user_id)
            
            if result["success"]:
                print(f"\n✅ Workflow completed successfully!")
                
                # Show summary
                summary = result["workflow_summary"]
                print(f"📊 Summary:")
                print(f"   • Total time: {summary['total_time']:.3f} seconds")
                print(f"   • Steps completed: {summary['steps_completed']}")
                print(f"   • Search results: {summary['search_results_count']}")
                print(f"   • Response length: {summary['final_response_length']} characters")
                
                # Show final response
                final_state = result["final_state"]
                if final_state.get("final_response"):
                    print(f"\n🎯 Final Response:")
                    print(f"{'='*80}")
                    print(final_state["final_response"])
                    print(f"{'='*80}")
                
            else:
                print(f"\n❌ Workflow failed: {result['error']}")
        
        # Clean up
        await agent.close()
        print("\n✅ Agent resources cleaned up")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Interactive mode failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the demo."""
    
    print("🚀 LangGraph Agent Workflow Demo")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "demo":
            print("🧪 Running demo mode...")
            asyncio.run(demo_workflow())
        elif mode == "interactive":
            print("🎮 Running interactive mode...")
            asyncio.run(interactive_mode())
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: demo, interactive")
            return
    else:
        # Default to demo mode
        print("🧪 Running demo mode (use 'interactive' for interactive mode)")
        asyncio.run(demo_workflow())

if __name__ == "__main__":
    main()

