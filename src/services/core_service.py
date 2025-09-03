"""
Core service for MongoDB + LangGraph + Voyage AI integration.
Follows the simplified architecture from the MongoDB article.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv
import asyncio

load_dotenv()

class CoreService:
    """
    Simplified core service that handles:
    - MongoDB connection and vector search
    - Memory storage and retrieval
    - Basic document processing
    """
    
    def __init__(self):
        """Initialize MongoDB connection and setup."""
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable is required")
        
        try:
            self.client = MongoClient(connection_string)
            self.client.admin.command('ping')
            self.database = os.getenv("MONGODB_DATABASE", "ai_memory")
            self.db = self.client[self.database]
            self.memory_collection: Collection = self.db.memory
            
            print("✅ MongoDB connected successfully")
            
            # Initialize indexes synchronously
            self.setup_indexes_sync()
                
        except Exception as e:
            raise Exception(f"MongoDB connection failed: {e}")
    
    def setup_indexes_sync(self):
        """Create vector search index for memory collection (synchronous version)."""
        try:
            # Check if index already exists
            existing_indexes = list(self.memory_collection.list_search_indexes())
            index_names = [idx["name"] for idx in existing_indexes]
            
            if "memory_vector_index" not in index_names:
                print("🔧 Creating memory_vector_index...")
                
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 1024,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name="memory_vector_index",
                    type="vectorSearch"
                )
                
                result = self.memory_collection.create_search_index(search_index_model)
                print(f"✅ Created vector index: {result}")
                
                # Wait for index to be ready
                max_wait = 60
                wait_time = 0
                
                while wait_time < max_wait:
                    try:
                        indices = list(self.memory_collection.list_search_indexes(result))
                        if len(indices) and indices[0].get("queryable") is True:
                            print("✅ memory_vector_index is ready")
                            break
                        import time
                        time.sleep(5)
                        wait_time += 5
                        print(f"⏳ Waiting for index... ({wait_time}s)")
                    except Exception as e:
                        print(f"⚠️ Error checking index status: {str(e)}")
                        break
            else:
                print("✅ memory_vector_index already exists")
                
        except Exception as e:
            print(f"❌ Failed to create memory index: {str(e)}")
    
    async def setup_indexes(self):
        """Create vector search index for memory collection (async version)."""
        # This method is kept for compatibility but the actual work is done in setup_indexes_sync
        pass
    
    async def store_memory(self, content: str, content_type: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a single memory entry in the memory collection."""
        try:
            memory_entry = {
                "content": content,
                "content_type": content_type,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            result = self.memory_collection.insert_one(memory_entry)
            return str(result.inserted_id)
            
        except Exception as e:
            raise Exception(f"Failed to store memory: {str(e)}")
    
    async def store_memories_bulk(self, memory_entries: List[Dict]) -> List[str]:
        """Store multiple memory entries in bulk for better performance."""
        try:
            if not memory_entries:
                return []
            
            # Ensure all entries have required fields
            for entry in memory_entries:
                if 'timestamp' not in entry:
                    entry['timestamp'] = datetime.utcnow()
                if 'created_at' not in entry:
                    entry['created_at'] = datetime.utcnow()
            
            # Use insert_many for bulk insert
            result = self.memory_collection.insert_many(memory_entries)
            return [str(inserted_id) for inserted_id in result.inserted_ids]
            
        except Exception as e:
            raise Exception(f"Failed to store memories in bulk: {str(e)}")
    
    async def search_memories(self, query_embedding: List[float], 
                             content_types: Optional[List[str]] = None,
                             exclude_documents: Optional[List[str]] = None,
                             limit: int = 5) -> List[Dict]:
        """Search memories using vector similarity."""
        try:
            # Build aggregation pipeline - $vectorSearch must be first!
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "embedding",
                        "numCandidates": limit * 20,  # Get more candidates for filtering
                        "limit": limit * 20,  # Get more results to filter from
                        "index": "memory_vector_index"
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "content_type": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Get all results first, then filter by content type in Python
            all_results = list(self.memory_collection.aggregate(pipeline))
            
            # Filter by content type if specified
            if content_types:
                filtered_results = [r for r in all_results if r.get('content_type') in content_types]
            else:
                filtered_results = all_results
            
            # Filter out excluded documents
            if exclude_documents:
                filtered_results = [
                    r for r in filtered_results 
                    if not (r.get('content_type') == 'document' and 
                           r.get('metadata', {}).get('document_id') in exclude_documents)
                ]
            
            # Filter out disabled documents
            filtered_results = [
                r for r in filtered_results 
                if not (r.get('content_type') == 'document' and 
                       r.get('metadata', {}).get('document_enabled') == False)
            ]
            
            # Sort by score and return top results
            filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return filtered_results[:limit]
            
        except Exception as e:
            print(f"⚠️ Vector search failed: {str(e)}")
            # Fallback to text search
            return await self.fallback_text_search(query_embedding, limit)
    
    async def fallback_text_search(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Fallback text search if vector search fails."""
        try:
            # Simple text search as fallback
            cursor = self.memory_collection.find().limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"⚠️ Fallback search failed: {str(e)}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """Retrieve a specific memory by ID."""
        try:
            from bson import ObjectId
            result = self.memory_collection.find_one({"_id": ObjectId(memory_id)})
            return result
        except Exception as e:
            print(f"⚠️ Failed to retrieve memory: {str(e)}")
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        try:
            from bson import ObjectId
            result = self.memory_collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"⚠️ Failed to delete memory: {str(e)}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            total_count = self.memory_collection.count_documents({})
            
            # Count by content type
            pipeline = [
                {"$group": {"_id": "$content_type", "count": {"$sum": 1}}}
            ]
            type_counts = list(self.memory_collection.aggregate(pipeline))
            
            return {
                "total_memories": total_count,
                "by_type": {item["_id"]: item["count"] for item in type_counts}
            }
            
        except Exception as e:
            print(f"⚠️ Failed to get memory stats: {str(e)}")
            return {"total_memories": 0, "by_type": {}}
    
    async def store_memories_batch(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory entries in a batch."""
        try:
            # Add timestamps if not present
            for memory in memories:
                if "timestamp" not in memory:
                    memory["timestamp"] = datetime.utcnow()
                if "created_at" not in memory:
                    memory["created_at"] = datetime.utcnow()
            
            result = await self.memory_collection.insert_many(memories)
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            raise Exception(f"Failed to store memories batch: {str(e)}")
    
    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
