#!/usr/bin/env python3
"""
Document Manager CLI
Command-line interface for managing documents in the memory collection
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append('src')

from src.services.document_manager import DocumentManager

class DocumentManagerCLI:
    """Command-line interface for document management."""
    
    def __init__(self):
        self.doc_manager = DocumentManager()
    
    async def upload_document(self, file_path: str, title: str = None, chunk_size: int = 1000):
        """Upload a document."""
        try:
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")
                return False
            
            file_size = Path(file_path).stat().st_size
            print(f"📤 Starting upload: {file_path}")
            print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   Chunk size: {chunk_size} characters")
            print()
            
            # Update chunk size in document manager
            self.doc_manager.chunk_size = chunk_size
            
            # Upload document with progress tracking
            print("🔄 Starting document upload process...")
            document_id = await self.doc_manager.upload_document(file_path, title)
            
            print(f"   ✅ Document uploaded successfully!")
            print()
            
            # Step 2: Verify upload
            print("🔄 Step 2/2: Verifying upload...")
            documents = await self.doc_manager.list_documents()
            uploaded_doc = next((doc for doc in documents if doc.document_id == document_id), None)
            
            if uploaded_doc:
                print(f"   ✅ Verification successful!")
                print(f"   📋 Document ID: {document_id}")
                print(f"   📄 Title: {uploaded_doc.title}")
                print(f"   🔢 Chunks stored: {uploaded_doc.chunk_count}")
                print(f"   💾 Total size: {uploaded_doc.total_size:,} characters")
                print(f"   🟢 Status: {'Enabled' if uploaded_doc.enabled else 'Disabled'}")
            else:
                print(f"   ⚠️  Upload verification failed - document not found in database")
            
            print()
            print("🎉 Document upload process completed!")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            print(f"   💡 Check the error details above and try again")
            return False
    
    async def list_documents(self):
        """List all documents."""
        try:
            print("📋 Listing documents...")
            documents = await self.doc_manager.list_documents()
            
            if not documents:
                print("📭 No documents found.")
                return
            
            print(f"\n📚 Found {len(documents)} documents:\n")
            
            for i, doc in enumerate(documents, 1):
                status = "✅ Enabled" if doc.enabled else "❌ Disabled"
                print(f"{i}. {doc.title} ({doc.filename})")
                print(f"   ID: {doc.document_id}")
                print(f"   Type: {doc.content_type}")
                print(f"   Chunks: {doc.chunk_count}")
                print(f"   Size: {doc.total_size} characters")
                print(f"   Status: {status}")
                print(f"   Uploaded: {doc.uploaded_at.strftime('%Y-%m-%d %H:%M')}")
                print()
            
        except Exception as e:
            print(f"❌ Failed to list documents: {str(e)}")
    
    async def toggle_document(self, document_id: str, enable: bool):
        """Enable or disable a document."""
        try:
            action = "enable" if enable else "disable"
            print(f"🔄 Attempting to {action} document: {document_id}")
            
            success = await self.doc_manager.toggle_document(document_id, enable)
            
            if success:
                print(f"✅ Document {action}d successfully!")
            else:
                print(f"❌ Failed to {action} document.")
            
        except Exception as e:
            print(f"❌ Error toggling document: {str(e)}")
    
    async def delete_document(self, document_id: str, force: bool = False):
        """Delete a document."""
        try:
            if not force:
                print(f"⚠️  Are you sure you want to delete document {document_id}?")
                print("   This will permanently delete the document and all its chunks.")
                confirm = input("   Type 'yes' to confirm: ")
                
                if confirm.lower() != 'yes':
                    print("❌ Deletion cancelled.")
                    return
            
            print(f"🗑️  Deleting document: {document_id}")
            success = await self.doc_manager.delete_document(document_id)
            
            if success:
                print("✅ Document deleted successfully!")
            else:
                print("❌ Failed to delete document.")
            
        except Exception as e:
            print(f"❌ Error deleting document: {str(e)}")
    
    async def compress_memories(self, user_id: str = None):
        """Compress duplicate memories."""
        try:
            print("🧠 Starting memory compression...")
            
            if user_id:
                print(f"   User ID: {user_id}")
            else:
                print("   All users")
            
            result = await self.doc_manager.compress_memories(user_id)
            
            if "error" in result:
                print(f"❌ Compression failed: {result['error']}")
                return
            
            print("✅ Memory compression completed!")
            print(f"   Compressed groups: {result['compressed_groups']}")
            print(f"   Total conversations compressed: {result['total_conversations_compressed']}")
            print(f"   New summaries created: {result['new_summaries_created']}")
            
        except Exception as e:
            print(f"❌ Memory compression failed: {str(e)}")
    
    async def get_stats(self):
        """Get document and memory statistics."""
        try:
            print("📊 Getting statistics...")
            
            # Get memory stats
            memory_stats = await self.doc_manager.core_service.get_memory_stats()
            
            print(f"\n📈 Memory Statistics:")
            print(f"   Total memories: {memory_stats['total_memories']}")
            
            for content_type, count in memory_stats['by_type'].items():
                print(f"   {content_type}: {count}")
            
            # Get document stats
            documents = await self.doc_manager.list_documents()
            
            if documents:
                print(f"\n📚 Document Statistics:")
                print(f"   Total documents: {len(documents)}")
                
                total_chunks = sum(doc.chunk_count for doc in documents)
                total_size = sum(doc.total_size for doc in documents)
                enabled_count = sum(1 for doc in documents if doc.enabled)
                
                print(f"   Total chunks: {total_chunks}")
                print(f"   Total size: {total_size:,} characters")
                print(f"   Enabled documents: {enabled_count}")
                print(f"   Disabled documents: {len(documents) - enabled_count}")
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {str(e)}")
    
    async def reembed_documents(self, force: bool = False):
        """Reembed all documents with the current model."""
        try:
            print("🔄 Starting reembedding process...")
            
            if not force:
                print("⚠️  WARNING: This will regenerate ALL embeddings in your database!")
                print("   This process may take a while depending on the number of documents.")
                confirm = input("   Type 'yes' to confirm: ")
                
                if confirm.lower() != 'yes':
                    print("❌ Reembedding cancelled.")
                    return
            
            # Import the reembedding service
            from reembed_documents import ReembeddingService
            
            # Run reembedding
            reembedding_service = ReembeddingService()
            await reembedding_service.run_reembedding()
            
        except Exception as e:
            print(f"❌ Reembedding failed: {str(e)}")

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Document Manager CLI for LangGraph Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a document
  python document_manager_cli.py upload document.txt --title "My Document"
  
  # List all documents
  python document_manager_cli.py list
  
  # Enable a document
  python document_manager_cli.py enable doc_abc123
  
  # Disable a document
  python document_manager_cli.py disable doc_abc123
  
  # Delete a document
  python document_manager_cli.py delete doc_abc123 --force
  
  # Compress memories
  python document_manager_cli.py compress
  
  # Get statistics
  python document_manager_cli.py stats
  
  # Reembed all documents (when model changes)
  python document_manager_cli.py reembed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a document')
    upload_parser.add_argument('file_path', help='Path to the document file')
    upload_parser.add_argument('--title', help='Document title (optional)')
    upload_parser.add_argument('--chunk-size', type=int, default=1000, 
                              help='Chunk size in characters (default: 1000)')
    
    # List command
    subparsers.add_parser('list', help='List all documents')
    
    # Enable command
    enable_parser = subparsers.add_parser('enable', help='Enable a document')
    enable_parser.add_argument('document_id', help='Document ID to enable')
    
    # Disable command
    disable_parser = subparsers.add_parser('disable', help='Disable a document')
    disable_parser.add_argument('document_id', help='Document ID to disable')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('document_id', help='Document ID to delete')
    delete_parser.add_argument('--force', action='store_true', 
                              help='Skip confirmation prompt')
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress duplicate memories')
    compress_parser.add_argument('--user-id', help='User ID to compress (optional)')
    
    # Stats command
    subparsers.add_parser('stats', help='Get document and memory statistics')
    
    # Reembed command
    reembed_parser = subparsers.add_parser('reembed', help='Reembed all documents with current model')
    reembed_parser.add_argument('--force', action='store_true',
                               help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance
    cli = DocumentManagerCLI()
    
    # Run the appropriate command
    async def run_command():
        if args.command == 'upload':
            await cli.upload_document(args.file_path, args.title, args.chunk_size)
        elif args.command == 'list':
            await cli.list_documents()
        elif args.command == 'enable':
            await cli.toggle_document(args.document_id, True)
        elif args.command == 'disable':
            await cli.toggle_document(args.document_id, False)
        elif args.command == 'delete':
            await cli.delete_document(args.document_id, args.force)
        elif args.command == 'compress':
            await cli.compress_memories(args.user_id)
        elif args.command == 'stats':
            await cli.get_stats()
        elif args.command == 'reembed':
            await cli.reembed_documents(args.force)
    
    # Run the async command
    await run_command()

if __name__ == "__main__":
    asyncio.run(main())
