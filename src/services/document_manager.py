"""
Document Manager Service
Handles document uploads, chunking, storage, and memory compression
"""

import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
from dataclasses import dataclass

from .core_service import CoreService
from .llm_service import LLMService

@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    content: str
    chunk_id: str
    chunk_index: int
    metadata: Dict[str, Any]

@dataclass
class DocumentInfo:
    """Represents document metadata."""
    document_id: str
    title: str
    filename: str
    content_type: str
    chunk_count: int
    total_size: int
    enabled: bool
    uploaded_at: datetime
    last_accessed: Optional[datetime] = None

class DocumentManager:
    """Manages document uploads, chunking, and memory compression."""
    
    def __init__(self):
        self.core_service = CoreService()
        self.llm_service = LLMService()
        self.chunk_size = 1000  # characters per chunk
        self.chunk_overlap = 200  # overlap between chunks
        
    async def upload_document(self, file_path: str, title: Optional[str] = None) -> str:
        """Upload and chunk a document, storing it in the memory collection."""
        try:
            print(f"🚀 STEP 1: Starting document upload process...")
            print(f"   📁 File: {file_path}")
            print(f"   📝 Title: {title or 'Auto-generated'}")
            print(f"   🔢 Chunk size: {self.chunk_size} characters")
            print()
            
            # Generate document ID
            print("🔄 STEP 2: Generating unique document ID...")
            document_id = self._generate_document_id(file_path)
            print(f"   ✅ Document ID: {document_id}")
            print()
            
            # Extract content based on file type
            print("🔄 STEP 3: Extracting document content...")
            content = await self._extract_document_content(file_path)
            print(f"   ✅ Content extracted: {len(content):,} characters")
            print()
            
            # Generate title if not provided
            if not title:
                title = Path(file_path).stem
            
            # Chunk the content
            print("🔄 STEP 4: Creating document chunks...")
            chunks = self._chunk_content(content, document_id)
            print(f"   ✅ Chunks created: {len(chunks)} chunks")
            print(f"   📊 Chunk sizes: {chunks[0].metadata['chunk_size']} - {chunks[-1].metadata['chunk_size']} characters")
            print()
            
            # Store document info
            print("🔄 STEP 5: Storing document metadata...")
            doc_info = DocumentInfo(
                document_id=document_id,
                title=title,
                filename=Path(file_path).name,
                content_type=self._get_file_type(file_path),
                chunk_count=len(chunks),
                total_size=len(content),
                enabled=True,
                uploaded_at=datetime.utcnow()
            )
            await self._store_document_metadata(doc_info)
            print(f"   ✅ Document metadata stored")
            print()
            
            # Store chunks with embeddings
            print("🔄 STEP 6: Processing chunks and generating embeddings...")
            await self._store_document_chunks(chunks, document_id)
            print()
            
            print("🎉 UPLOAD COMPLETED SUCCESSFULLY!")
            print(f"   📋 Document ID: {document_id}")
            print(f"   📄 Title: {title}")
            print(f"   🔢 Total chunks: {len(chunks)}")
            print(f"   💾 Total size: {len(content):,} characters")
            print(f"   🟢 Status: Enabled (ready for queries)")
            
            return document_id
            
        except Exception as e:
            print(f"❌ UPLOAD FAILED: {str(e)}")
            print(f"   💡 Check the error details above and try again")
            raise Exception(f"Failed to upload document: {str(e)}")
    
    async def _extract_document_content(self, file_path: str) -> str:
        """Extract text content from various document formats."""
        file_path = Path(file_path)
        
        print(f"      📖 Reading file: {file_path.name}")
        
        if file_path.suffix.lower() == '.txt':
            print(f"      📄 Processing as TEXT file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        elif file_path.suffix.lower() == '.md':
            print(f"      📝 Processing as MARKDOWN file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        elif file_path.suffix.lower() == '.html':
            print(f"      🌐 Processing as HTML file...")
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                content = soup.get_text()
        
        elif file_path.suffix.lower() == '.pdf':
            print(f"      📕 Processing as PDF file...")
            import PyPDF2
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        print(f"      ✅ File read successfully")
        return content
    
    def _chunk_content(self, content: str, document_id: str) -> List[DocumentChunk]:
        """Split content into overlapping chunks."""
        print(f"      ✂️  Splitting content into chunks...")
        print(f"         📊 Content length: {len(content)} characters")
        print(f"         🔢 Chunk size: {self.chunk_size} characters")
        print(f"         🔄 Overlap: {self.chunk_overlap} characters")
        
        chunks = []
        start = 0
        chunk_count = 0
        
        while start < len(content):
            chunk_count += 1
            print(f"         🔄 Processing chunk {chunk_count}...")
            
            end = start + self.chunk_size
            print(f"         📍 Start: {start}, End: {end}")
            
            # Extract chunk content
            chunk_content = content[start:end]
            print(f"         📝 Chunk content length: {len(chunk_content)}")
            
            # Create chunk ID
            chunk_id = f"{document_id}_chunk_{len(chunks)}"
            print(f"         🆔 Chunk ID: {chunk_id}")
            
            # Create metadata
            metadata = {
                "document_id": document_id,
                "chunk_index": len(chunks),
                "chunk_size": len(chunk_content),
                "start_position": start,
                "end_position": end,
                "chunk_type": "document"
            }
            print(f"         📋 Metadata created")
            
            print(f"         🔨 Creating DocumentChunk object...")
            chunks.append(DocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                chunk_index=len(chunks),
                metadata=metadata
            ))
            print(f"         ✅ Chunk {chunk_count} created successfully")
            
            # Move to next chunk with overlap (but don't exceed chunk size)
            overlap = min(self.chunk_overlap, self.chunk_size - 1)
            start = end - overlap
            print(f"         ➡️  Next start position: {start} (overlap: {overlap})")
            
            # If we've reached the end of content, break
            if end >= len(content):
                print(f"         🏁 Reached end of content, breaking loop")
                break
            
            # Prevent infinite loop - if we're not making progress, break
            if start <= 0 and chunk_count > 1:
                print(f"         🛑 Breaking loop to prevent infinite iteration")
                break
        
        print(f"      ✅ Created {len(chunks)} chunks with {self.chunk_overlap} character overlap")
        return chunks
    
    async def _store_document_chunks(self, chunks: List[DocumentChunk], document_id: str):
        """Store document chunks with embeddings in the memory collection using bulk insert."""
        total_chunks = len(chunks)
        print(f"   🔄 Processing {total_chunks} chunks...")
        print(f"   ⏱️  This step involves:")
        print(f"      • Generating embeddings via Voyage AI (batch processing)")
        print(f"      • Bulk storing chunks in MongoDB for better performance")
        print(f"      • Progress will be shown for each batch")
        print()
        
        # Process chunks in batches for better performance
        batch_size = 50  # Process 50 chunks at a time
        memory_entries = []
        
        for i, chunk in enumerate(chunks, 1):
            # Show progress
            progress = (i / total_chunks) * 100
            print(f"      [{progress:5.1f}%] Chunk {i}/{total_chunks}: Generating embedding...")
            
            # Generate embedding for the chunk
            print(f"         🧠 Sending to Voyage AI for embedding...")
            embedding = await self.llm_service.generate_embedding(chunk.content)
            print(f"         ✅ Embedding generated ({len(embedding)} dimensions)")
            
            # Prepare memory entry
            memory_entry = {
                "content": chunk.content,
                "content_type": "document",
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "timestamp": datetime.utcnow(),
                    "created_at": datetime.utcnow(),
                    "document_enabled": True
                }
            }
            
            memory_entries.append(memory_entry)
            
            # Process batch when we reach batch size or end of chunks
            if len(memory_entries) >= batch_size or i == total_chunks:
                print(f"         💾 Bulk storing {len(memory_entries)} chunks in MongoDB...")
                try:
                    await self.core_service.store_memories_bulk(memory_entries)
                    print(f"         ✅ Successfully stored {len(memory_entries)} chunks")
                except Exception as e:
                    print(f"         ❌ Failed to store batch: {str(e)}")
                    # Fallback to individual inserts for this batch
                    print(f"         🔄 Falling back to individual inserts...")
                    for entry in memory_entries:
                        try:
                            await self.core_service.store_memory(
                                content=entry["content"],
                                content_type=entry["content_type"],
                                embedding=entry["embedding"],
                                metadata=entry["metadata"]
                            )
                        except Exception as individual_error:
                            print(f"         ❌ Failed to store individual chunk: {str(individual_error)}")
                
                memory_entries = []  # Clear the batch
                print(f"      [{progress:5.1f}%] Batch completed: {i}/{total_chunks} chunks processed")
                print()
        
        print(f"   🎉 All {total_chunks} chunks processed and stored successfully!")
    async def _store_document_metadata(self, doc_info: DocumentInfo):
        """Store document metadata in a separate collection."""
        print(f"      📋 Storing document metadata...")
        
        # This could be stored in a separate 'documents' collection
        # For now, we'll store it in the memory collection as a special entry
        metadata_content = f"DOCUMENT_METADATA: {doc_info.title}"
        
        print(f"         🧠 Generating metadata embedding...")
        embedding = await self.llm_service.generate_embedding(metadata_content)
        
        print(f"         💾 Storing metadata in MongoDB...")
        await self.core_service.store_memory(
            content=metadata_content,
            content_type="document_metadata",
            embedding=embedding,
            metadata={
                "document_id": doc_info.document_id,
                "title": doc_info.title,
                "filename": doc_info.filename,
                "content_type": doc_info.content_type,
                "chunk_count": doc_info.chunk_count,
                "total_size": doc_info.total_size,
                "enabled": doc_info.enabled,
                "uploaded_at": doc_info.uploaded_at,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
        )
        
        print(f"      ✅ Document metadata stored successfully")
    
    async def toggle_document(self, document_id: str, enabled: bool) -> bool:
        """Enable or disable a document from appearing in queries."""
        try:
            action = "enable" if enabled else "disable"
            print(f"🔄 Attempting to {action} document: {document_id}")
            
            # Update all chunks for this document
            result = self.core_service.memory_collection.update_many(
                {"metadata.document_id": document_id, "content_type": "document"},
                {"$set": {"metadata.document_enabled": enabled}}
            )
            
            # Update document metadata
            self.core_service.memory_collection.update_many(
                {"metadata.document_id": document_id, "content_type": "document_metadata"},
                {"$set": {"metadata.enabled": enabled}}
            )
            
            print(f"✅ Document {document_id} {'enabled' if enabled else 'disabled'}")
            print(f"   Updated {result.modified_count} chunks")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to toggle document: {str(e)}")
            return False
    
    async def list_documents(self) -> List[DocumentInfo]:
        """List all documents with their metadata."""
        try:
            # Find all document metadata entries
            cursor = self.core_service.memory_collection.find(
                {"content_type": "document_metadata"}
            )
            
            documents = []
            for doc in cursor:
                metadata = doc.get("metadata", {})
                doc_info = DocumentInfo(
                    document_id=metadata.get("document_id"),
                    title=metadata.get("title"),
                    filename=metadata.get("filename"),
                    content_type=metadata.get("content_type"),
                    chunk_count=metadata.get("chunk_count"),
                    total_size=metadata.get("total_size"),
                    enabled=metadata.get("enabled", True),
                    uploaded_at=metadata.get("uploaded_at"),
                    last_accessed=metadata.get("last_accessed")
                )
                documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            print(f"❌ Failed to list documents: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Delete all chunks for this document
            result = self.core_service.memory_collection.delete_many({
                "metadata.document_id": document_id,
                "content_type": "document"
            })
            
            # Delete document metadata
            self.core_service.memory_collection.delete_many({
                "metadata.document_id": document_id,
                "content_type": "document_metadata"
            })
            
            print(f"✅ Document {document_id} deleted")
            print(f"   Removed {result.deleted_count} chunks")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete document: {str(e)}")
            return False
    
    async def compress_memories(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Compress duplicate conversations and generate summaries."""
        try:
            print("🧠 Starting memory compression...")
            
            # Find conversations with similar queries
            similar_conversations = await self._find_similar_conversations(user_id)
            
            compression_results = {
                "compressed_groups": 0,
                "total_conversations_compressed": 0,
                "new_summaries_created": 0,
                "details": []
            }
            
            for group in similar_conversations:
                if len(group) > 1:  # Only compress if we have duplicates
                    result = await self._compress_conversation_group(group)
                    compression_results["compressed_groups"] += 1
                    compression_results["total_conversations_compressed"] += len(group)
                    compression_results["new_summaries_created"] += 1
                    compression_results["details"].append(result)
            
            print(f"✅ Memory compression completed:")
            print(f"   Compressed groups: {compression_results['compressed_groups']}")
            print(f"   Total conversations compressed: {compression_results['total_conversations_compressed']}")
            print(f"   New summaries created: {compression_results['new_summaries_created']}")
            
            return compression_results
            
        except Exception as e:
            print(f"❌ Memory compression failed: {str(e)}")
            return {"error": str(e)}
    
    async def _find_similar_conversations(self, user_id: Optional[str] = None) -> List[List[Dict]]:
        """Find groups of similar conversations for compression."""
        try:
            # Find all conversations
            filter_query = {"content_type": "conversation"}
            if user_id:
                filter_query["metadata.user_id"] = user_id
            
            conversations = list(self.core_service.memory_collection.find(filter_query))
            
            # Group by similarity (simple approach: group by first 50 characters)
            groups = {}
            for conv in conversations:
                content = conv.get("content", "")
                if len(content) > 50:
                    key = content[:50].lower().strip()
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(conv)
            
            # Return groups with more than one conversation
            return [group for group in groups.values() if len(group) > 1]
            
        except Exception as e:
            print(f"❌ Failed to find similar conversations: {str(e)}")
            return []
    
    async def _compress_conversation_group(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Compress a group of similar conversations into a summary."""
        try:
            # Extract all responses
            responses = []
            for conv in conversations:
                metadata = conv.get("metadata", {})
                response = metadata.get("response", "")
                if response:
                    responses.append(response)
            
            if not responses:
                return {"error": "No responses found"}
            
            # Generate summary using LLM
            summary_prompt = f"""The following are multiple responses to similar questions. 
            Please create a comprehensive summary that captures the key information from all responses:
            
            {'\n\n'.join(responses)}
            
            Summary:"""
            
            summary = await self.llm_service.generate_response(summary_prompt)
            
            # Create new compressed memory entry
            embedding = await self.llm_service.generate_embedding(summary)
            
            compressed_id = await self.core_service.store_memory(
                content=summary,
                content_type="compressed_memory",
                embedding=embedding,
                metadata={
                    "compression_type": "conversation_summary",
                    "original_conversations": [str(conv["_id"]) for conv in conversations],
                    "compressed_count": len(conversations),
                    "timestamp": datetime.utcnow(),
                    "created_at": datetime.utcnow()
                }
            )
            
            # Delete original conversations
            conversation_ids = [conv["_id"] for conv in conversations]
            self.core_service.memory_collection.delete_many({
                "_id": {"$in": conversation_ids}
            })
            
            return {
                "compressed_id": compressed_id,
                "original_count": len(conversations),
                "summary_length": len(summary)
                }
                
        except Exception as e:
            print(f"❌ Failed to compress conversation group: {str(e)}")
            return {"error": str(e)}
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file path and content."""
        file_path = Path(file_path)
        timestamp = datetime.utcnow().isoformat()
        content_hash = hashlib.md5(f"{file_path}_{timestamp}".encode()).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def _get_file_type(self, file_path: str) -> str:
        """Get the file type from the file extension."""
        return Path(file_path).suffix.lower()[1:]  # Remove the dot
