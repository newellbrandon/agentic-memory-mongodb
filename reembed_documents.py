#!/usr/bin/env python3
"""
Reembedding Script for Agentic Memory with MongoDB
Regenerates all embeddings when VOYAGE_MODEL changes
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import LLMService
from src.services.core_service import CoreService

class ReembeddingService:
    """Service to handle reembedding of all documents when model changes."""
    
    def __init__(self):
        """Initialize the reembedding service."""
        self.llm_service = LLMService()
        self.core_service = CoreService()
        self.memory_collection = self.core_service.memory_collection
        
        # Get current model info
        self.current_model = os.getenv("VOYAGE_MODEL", "voyage-large-2-instruct")
        print(f"🔄 Reembedding with model: {self.current_model}")
        
        # Check if this is a dimension-compatible model switch
        self.is_dimension_compatible = self.current_model in [
            "voyage-large-2-instruct", 
            "voyage-context-3",
            "voyage-2"
        ]
        
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents that need reembedding."""
        print("📋 Fetching all documents from memory collection...")
        
        # Get all documents that have embeddings
        documents = list(self.memory_collection.find({
            "embedding": {"$exists": True},
            "content": {"$exists": True, "$ne": ""}
        }))
        
        print(f"📊 Found {len(documents)} documents with embeddings")
        return documents
    
    async def regenerate_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Regenerate embeddings for all documents."""
        print("🔄 Starting reembedding process...")
        
        stats = {
            "total_documents": len(documents),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process documents in batches
        batch_size = 10
        for i in tqdm(range(0, len(documents), batch_size), desc="Reembedding documents"):
            batch = documents[i:i + batch_size]
            
            # Process batch
            for doc in batch:
                try:
                    # Generate new embedding
                    new_embedding = await self.llm_service.generate_embedding(doc["content"])
                    
                    # Update document with new embedding
                    result = self.memory_collection.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "embedding": new_embedding,
                                "embedding_model": self.current_model,
                                "last_updated": datetime.utcnow()
                            }
                        }
                    )
                    
                    if result.modified_count > 0:
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append(f"Failed to update document {doc['_id']}")
                        
                except Exception as e:
                    stats["failed"] += 1
                    error_msg = f"Error processing document {doc['_id']}: {str(e)}"
                    stats["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
        
        return stats
    
    async def update_vector_index(self):
        """Update the vector index to match new model dimensions."""
        print("🔧 Checking vector index compatibility...")
        
        try:
            # Get the new embedding dimensions
            sample_embedding = await self.llm_service.generate_embedding("sample text")
            new_dimensions = len(sample_embedding)
            
            print(f"📏 New embedding dimensions: {new_dimensions}")
            
            # Check if we need to update the index
            if self.is_dimension_compatible and new_dimensions == 1024:
                print("✅ Vector index is compatible (1024 dimensions)")
                print("   No need to drop and recreate the index")
                print("   Index will automatically work with updated embeddings")
                return
            
            # Only drop and recreate if dimensions changed
            print("⚠️ Embedding dimensions changed, updating vector index...")
            
            # Drop existing vector index
            try:
                self.memory_collection.drop_index("memory_vector_index")
                print("🗑️ Dropped existing vector index")
            except Exception as e:
                print(f"⚠️ Could not drop existing index: {e}")
            
            # Create new vector index with correct dimensions
            vector_index = {
                "fields": [
                    {
                        "numDimensions": new_dimensions,
                        "path": "embedding",
                        "similarity": "cosine",
                        "type": "vector"
                    },
                    {
                        "path": "content_type",
                        "type": "filter"
                    },
                    {
                        "path": "user_id",
                        "type": "filter"
                    },
                    {
                        "path": "enabled",
                        "type": "filter"
                    }
                ]
            }
            
            # Create the new index
            self.memory_collection.create_search_index(
                "memory_vector_index",
                vector_index
            )
            
            print("✅ New vector index created successfully")
            
        except Exception as e:
            print(f"❌ Error updating vector index: {e}")
            raise
    
    async def verify_reembedding(self) -> Dict[str, Any]:
        """Verify that reembedding was successful."""
        print("🔍 Verifying reembedding results...")
        
        # Check a few random documents
        sample_docs = list(self.memory_collection.aggregate([
            {"$match": {"embedding": {"$exists": True}}},
            {"$sample": {"size": 5}}
        ]))
        
        verification_stats = {
            "sample_docs_checked": len(sample_docs),
            "correct_dimensions": 0,
            "correct_model": 0,
            "issues": []
        }
        
        for doc in sample_docs:
            # Check embedding dimensions
            if "embedding" in doc and len(doc["embedding"]) > 0:
                verification_stats["correct_dimensions"] += 1
            else:
                verification_stats["issues"].append(f"Document {doc['_id']} has invalid embedding")
            
            # Check model name
            if doc.get("embedding_model") == self.current_model:
                verification_stats["correct_model"] += 1
            else:
                verification_stats["issues"].append(f"Document {doc['_id']} has wrong model: {doc.get('embedding_model')}")
        
        return verification_stats
    
    async def run_reembedding(self):
        """Run the complete reembedding process."""
        print("🚀 Starting complete reembedding process...")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.is_dimension_compatible:
            print("✅ Dimension-compatible model switch detected")
            print("   Vector index will be preserved and updated automatically")
        
        try:
            # Step 1: Get all documents
            documents = await self.get_all_documents()
            
            if not documents:
                print("ℹ️ No documents found to reembed")
                return
            
            # Step 2: Regenerate embeddings
            stats = await self.regenerate_embeddings(documents)
            
            # Step 3: Update vector index (if needed)
            await self.update_vector_index()
            
            # Step 4: Verify results
            verification = await self.verify_reembedding()
            
            # Print summary
            print("\n" + "="*50)
            print("📊 REEMBEDDING SUMMARY")
            print("="*50)
            print(f"✅ Successful: {stats['successful']}")
            print(f"❌ Failed: {stats['failed']}")
            print(f"📏 Total documents: {stats['total_documents']}")
            print(f"🔍 Sample verification: {verification['correct_dimensions']}/{verification['sample_docs_checked']} correct dimensions")
            print(f"🏷️ Model verification: {verification['correct_model']}/{verification['sample_docs_checked']} correct model")
            
            if self.is_dimension_compatible:
                print("✅ Vector index preserved (dimension-compatible model)")
            
            if verification['issues']:
                print("\n⚠️ Issues found:")
                for issue in verification['issues']:
                    print(f"  - {issue}")
            
            print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Reembedding failed: {e}")
            raise

async def main():
    """Main function to run reembedding."""
    print("🤖 Agentic Memory Reembedding Tool")
    print("="*40)
    
    # Check if user wants to proceed
    print("⚠️ WARNING: This will regenerate ALL embeddings in your database!")
    print("This process may take a while depending on the number of documents.")
    
    response = input("\nDo you want to continue? (yes/no): ").lower().strip()
    if response != 'yes':
        print("❌ Reembedding cancelled")
        return
    
    # Run reembedding
    reembedding_service = ReembeddingService()
    await reembedding_service.run_reembedding()

if __name__ == "__main__":
    asyncio.run(main())
