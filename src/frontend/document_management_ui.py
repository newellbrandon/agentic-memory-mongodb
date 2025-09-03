"""
Document Management UI Component
Provides interface for uploading, managing, and controlling documents
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Any

from src.services.document_manager import DocumentManager, DocumentInfo

class DocumentManagementUI:
    """Streamlit UI for document management."""
    
    def __init__(self):
        self.doc_manager = DocumentManager()
    
    def render(self):
        """Render the document management interface."""
        st.header("📚 Document Management")
        st.markdown("Upload, manage, and control which documents appear in your queries.")
        
        # Create tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs([
            "📤 Upload Documents", 
            "📋 Manage Documents", 
            "🧠 Memory Compression",
            "📊 Document Stats"
        ])
        
        with tab1:
            self._render_upload_tab()
        
        with tab2:
            self._render_manage_tab_sync()
        
        with tab3:
            self._render_compression_tab_sync()
        
        with tab4:
            self._render_stats_tab_sync()
    
    def _render_manage_tab_sync(self):
        """Render the document management interface synchronously."""
        st.subheader("Manage Existing Documents")
        
        try:
            # Get documents using asyncio.run
            import asyncio
            documents = asyncio.run(self.doc_manager.list_documents())
            
            if not documents:
                st.info("No documents found. Upload some documents first!")
                return
            
            # Display documents in a table
            st.write(f"Found {len(documents)} documents:")
            
            # Create columns for different actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Enable/disable documents
                toggle_doc = st.selectbox(
                    "Select document to toggle:",
                    options=[doc.title for doc in documents],
                    key="toggle_doc"
                )
                
                if toggle_doc:
                    doc = next(doc for doc in documents if doc.title == toggle_doc)
                    
                    if st.button("🔄 Toggle Status", type="secondary", key="toggle_btn"):
                        success = asyncio.run(self.doc_manager.toggle_document(doc.document_id, not doc.enabled))
                        if success:
                            st.success(f"Document {'enabled' if not doc.enabled else 'disabled'} successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to toggle document.")
            
            with col2:
                # Delete documents
                delete_doc = st.selectbox(
                    "Select document to delete:",
                    options=[doc.title for doc in documents],
                    key="delete_doc"
                )
                
                if delete_doc:
                    doc = next(doc for doc in documents if doc.title == delete_doc)
                    
                    if st.button("🗑️ Delete Document", type="secondary", key="delete_btn"):
                        # Confirmation
                        if st.checkbox("I understand this will permanently delete the document and all its chunks"):
                            success = asyncio.run(self.doc_manager.delete_document(doc.document_id))
                            if success:
                                st.success("Document deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")
            
            with col3:
                # Document info
                info_doc = st.selectbox(
                    "Select document for info:",
                    options=[doc.title for doc in documents],
                    key="info_doc"
                )
                
                if info_doc:
                    doc = next(doc for doc in documents if doc.title == info_doc)
                    
                    st.info(f"""
                    **Document Details:**
                    - **ID:** {doc.document_id}
                    - **Title:** {doc.title}
                    - **Filename:** {doc.filename}
                    - **Type:** {doc.content_type}
                    - **Chunks:** {doc.chunk_count}
                    - **Size:** {doc.total_size} characters
                    - **Status:** {'Enabled' if doc.enabled else 'Disabled'}
                    - **Uploaded:** {doc.uploaded_at.strftime("%Y-%m-%d %H:%M")}
                    """)
        
        except Exception as e:
            st.error(f"❌ Failed to load documents: {str(e)}")
    
    def _render_compression_tab_sync(self):
        """Render the memory compression interface synchronously."""
        st.subheader("Memory Compression")
        st.markdown("Compress duplicate conversations and generate summaries to save space.")
        
        # Compression options
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input(
                "User ID (optional)",
                placeholder="Leave empty to compress all users",
                help="Specify a user ID to compress only their conversations"
            )
        
        with col2:
            compression_type = st.selectbox(
                "Compression Type",
                options=["Conversations Only", "All Memory Types"],
                help="Choose what types of memories to compress"
            )
        
        # Compression button
        if st.button("🧠 Start Memory Compression", type="primary"):
            self._handle_memory_compression_sync(user_id, compression_type)
    
    def _render_stats_tab_sync(self):
        """Render the statistics interface synchronously."""
        st.subheader("Document and Memory Statistics")
        
        try:
            # Get memory stats using asyncio.run
            import asyncio
            memory_stats = asyncio.run(self.doc_manager.core_service.get_memory_stats())
            
            # Display memory statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Memories", memory_stats.get("total_memories", 0))
            
            with col2:
                st.metric("Documents", memory_stats.get("document_count", 0))
            
            with col3:
                st.metric("Conversations", memory_stats.get("conversation_count", 0))
            
            with col4:
                st.metric("Personal Info", memory_stats.get("personal_info_count", 0))
            
            # Get document list
            documents = asyncio.run(self.doc_manager.list_documents())
            
            if documents:
                st.subheader("Document Details")
                
                # Create a DataFrame for better display
                import pandas as pd
                
                doc_data = []
                for doc in documents:
                    doc_data.append({
                        "Title": doc.title,
                        "Filename": doc.filename,
                        "Type": doc.content_type,
                        "Chunks": doc.chunk_count,
                        "Size": f"{doc.total_size:,} chars",
                        "Status": "Enabled" if doc.enabled else "Disabled",
                        "Uploaded": doc.uploaded_at.strftime("%Y-%m-%d %H:%M")
                    })
                
                df = pd.DataFrame(doc_data)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary")
                total_chunks = sum(doc.chunk_count for doc in documents)
                total_size = sum(doc.total_size for doc in documents)
                enabled_docs = sum(1 for doc in documents if doc.enabled)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", f"{total_chunks:,}")
                with col2:
                    st.metric("Total Size", f"{total_size:,} characters")
                with col3:
                    st.metric("Enabled Documents", f"{enabled_docs}/{len(documents)}")
            else:
                st.info("No documents found.")
        
        except Exception as e:
            st.error(f"❌ Failed to load statistics: {str(e)}")
    
    def _render_upload_tab(self):
        """Render the document upload interface."""
        st.subheader("Upload New Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['txt', 'md', 'html', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: TXT, Markdown, HTML, PDF"
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} files selected**")
            
            # Document title input
            col1, col2 = st.columns([2, 1])
            with col1:
                document_title = st.text_input(
                    "Document Title (optional)",
                    placeholder="Enter a title for the document"
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Chunk Size (chars)",
                    min_value=500,
                    max_value=3000,
                    value=1000,
                    step=100,
                    help="Number of characters per chunk"
                )
            
            # Upload button
            if st.button("🚀 Upload Documents", type="primary"):
                self._handle_document_upload_sync(uploaded_files, document_title, chunk_size)
    
    def _handle_document_upload_sync(self, uploaded_files, title: str, chunk_size: int):
        """Handle the document upload process synchronously."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if not uploaded_files:
                st.error("❌ Please select at least one file to upload.")
                return
            
            total_files = len(uploaded_files)
            uploaded_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                status_text.text(f"Uploading {file_name}... ({i+1}/{total_files})")
                progress_bar.progress((i / total_files) * 0.8)
                
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Upload document using asyncio.run
                    import asyncio
                    document_id = asyncio.run(self.doc_manager.upload_document(tmp_file_path, title or file_name))
                    
                    if document_id:
                        uploaded_count += 1
                        st.success(f"✅ {file_name} uploaded successfully! (ID: {document_id})")
                    else:
                        st.error(f"❌ Failed to upload {file_name}")
                
                except Exception as e:
                    st.error(f"❌ Error uploading {file_name}: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            
            progress_bar.progress(1.0)
            
            if uploaded_count > 0:
                st.success(f"🎉 Successfully uploaded {uploaded_count}/{total_files} documents!")
                status_text.text(f"Upload completed: {uploaded_count}/{total_files} documents")
                st.rerun()
            else:
                st.error("❌ No documents were uploaded successfully.")
                status_text.text("Upload failed")
        
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Upload failed")
    
    async def _handle_document_upload(self, uploaded_files, title: str, chunk_size: int):
        """Handle the document upload process."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_files = len(uploaded_files)
            uploaded_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / total_files)
                
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Use provided title or generate from filename
                    doc_title = title if title else uploaded_file.name
                    
                    # Update chunk size in document manager
                    self.doc_manager.chunk_size = chunk_size
                    
                    # Upload document
                    document_id = await self.doc_manager.upload_document(tmp_file_path, doc_title)
                    
                    uploaded_count += 1
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text(f"🎉 Upload complete! {uploaded_count}/{total_files} documents uploaded successfully.")
            
            # Refresh the page to show new documents
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            status_text.text("Upload failed. Please try again.")
    
    async def _render_manage_tab(self):
        """Render the document management interface."""
        st.subheader("Manage Existing Documents")
        
        # Load documents
        try:
            documents = await self.doc_manager.list_documents()
            
            if not documents:
                st.info("📭 No documents found. Upload some documents to get started!")
                return
            
            # Create a DataFrame for better display
            df_data = []
            for doc in documents:
                df_data.append({
                    "Title": doc.title,
                    "Filename": doc.filename,
                    "Type": doc.content_type.upper(),
                    "Chunks": doc.chunk_count,
                    "Size (KB)": round(doc.total_size / 1024, 1),
                    "Status": "✅ Enabled" if doc.enabled else "❌ Disabled",
                    "Uploaded": doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                    "Document ID": doc.document_id
                })
            
            df = pd.DataFrame(df_data)
            
            # Display documents table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Document actions
            st.subheader("Document Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Enable/Disable document
                selected_doc = st.selectbox(
                    "Select document to toggle:",
                    options=[doc.title for doc in documents],
                    key="toggle_doc"
                )
                
                if selected_doc:
                    doc = next(doc for doc in documents if doc.title == selected_doc)
                    current_status = "Enable" if not doc.enabled else "Disable"
                    
                    if st.button(f"{current_status} Document", key="toggle_btn"):
                        success = asyncio.run(self.doc_manager.toggle_document(doc.document_id, not doc.enabled))
                        if success:
                            st.success(f"Document {current_status.lower()}d successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update document status.")
            
            with col2:
                # Delete document
                delete_doc = st.selectbox(
                    "Select document to delete:",
                    options=[doc.title for doc in documents],
                    key="delete_doc"
                )
                
                if delete_doc:
                    doc = next(doc for doc in documents if doc.title == delete_doc)
                    
                    if st.button("🗑️ Delete Document", type="secondary", key="delete_btn"):
                        # Confirmation
                        if st.checkbox("I understand this will permanently delete the document and all its chunks"):
                            success = asyncio.run(self.doc_manager.delete_document(doc.document_id))
                            if success:
                                st.success("Document deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")
            
            with col3:
                # Document info
                info_doc = st.selectbox(
                    "Select document for info:",
                    options=[doc.title for doc in documents],
                    key="info_doc"
                )
                
                if info_doc:
                    doc = next(doc for doc in documents if doc.title == info_doc)
                    
                    st.info(f"""
                    **Document Details:**
                    - **ID:** {doc.document_id}
                    - **Title:** {doc.title}
                    - **Filename:** {doc.filename}
                    - **Type:** {doc.content_type}
                    - **Chunks:** {doc.chunk_count}
                    - **Size:** {doc.total_size} characters
                    - **Status:** {'Enabled' if doc.enabled else 'Disabled'}
                    - **Uploaded:** {doc.uploaded_at.strftime("%Y-%m-%d %H:%M")}
                    """)
        
        except Exception as e:
            st.error(f"❌ Failed to load documents: {str(e)}")
    
    async def _render_compression_tab(self):
        """Render the memory compression interface."""
        st.subheader("Memory Compression")
        st.markdown("Compress duplicate conversations and generate summaries to save space.")
        
        # Compression options
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input(
                "User ID (optional)",
                placeholder="Leave empty to compress all users",
                help="Specify a user ID to compress only their conversations"
            )
        
        with col2:
            compression_type = st.selectbox(
                "Compression Type",
                options=["Conversations Only", "All Memory Types"],
                help="Choose what types of memories to compress"
            )
        
        # Compression button
        if st.button("🧠 Start Memory Compression", type="primary"):
            self._handle_memory_compression_sync(user_id, compression_type)
    
    def _handle_memory_compression_sync(self, user_id: str, compression_type: str):
        """Handle the memory compression process synchronously."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Starting memory compression...")
            progress_bar.progress(0.2)
            
            # Perform compression using asyncio.run
            import asyncio
            result = asyncio.run(self.doc_manager.compress_memories(user_id if user_id else None))
            progress_bar.progress(0.8)
            
            if "error" in result:
                st.error(f"❌ Compression failed: {result['error']}")
                return
            
            # Display results
            st.success("🎉 Memory compression completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Compressed Groups", result["compressed_groups"])
            
            with col2:
                st.metric("Total Conversations Compressed", result["total_conversations_compressed"])
            
            with col3:
                st.metric("New Summaries Created", result["new_summaries_created"])
            
            with col4:
                space_saved = result["total_conversations_compressed"] - result["new_summaries_created"]
                st.metric("Space Saved", f"{space_saved} conversations")
            
            # Show summary
            if result["compressed_groups"] > 0:
                st.info(f"✅ Successfully compressed {result['compressed_groups']} groups of similar conversations.")
                st.info(f"📊 Reduced {result['total_conversations_compressed']} conversations to {result['new_summaries_created']} summaries.")
            else:
                st.info("ℹ️ No duplicate conversations found to compress.")
            
            progress_bar.progress(1.0)
            status_text.text("Memory compression completed!")
            
        except Exception as e:
            st.error(f"❌ Compression failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Compression failed")
    
    async def _handle_memory_compression(self, user_id: str, compression_type: str):
        """Handle the memory compression process."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Starting memory compression...")
            progress_bar.progress(0.2)
            
            # Perform compression
            result = await self.doc_manager.compress_memories(user_id if user_id else None)
            progress_bar.progress(0.8)
            
            if "error" in result:
                st.error(f"❌ Compression failed: {result['error']}")
                return
            
            # Display results
            st.success("🎉 Memory compression completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Compressed Groups", result["compressed_groups"])
            
            with col2:
                st.metric("Conversations Compressed", result["total_conversations_compressed"])
            
            with col3:
                st.metric("New Summaries", result["new_summaries_created"])
            
            with col4:
                space_saved = result["total_conversations_compressed"] * 0.5  # Estimate
                st.metric("Space Saved (est.)", f"{space_saved:.1f} KB")
            
            # Show detailed results
            if result["details"]:
                st.subheader("Compression Details")
                for detail in result["details"]:
                    if "error" not in detail:
                        st.info(f"""
                        **Compressed ID:** {detail['compressed_id']}
                        **Original Count:** {detail['original_count']}
                        **Summary Length:** {detail['summary_length']} characters
                        """)
            
            progress_bar.progress(1.0)
            status_text.text("Compression completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Compression failed: {str(e)}")
            status_text.text("Compression failed. Please try again.")
    
    async def _render_stats_tab(self):
        """Render the document statistics interface."""
        st.subheader("Document Statistics")
        
        try:
            # Get memory stats
            memory_stats = await self.doc_manager.core_service.get_memory_stats()
            
            # Display overall stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Memories", memory_stats["total_memories"])
            
            with col2:
                doc_count = memory_stats["by_type"].get("document", 0)
                st.metric("Document Chunks", doc_count)
            
            with col3:
                conv_count = memory_stats["by_type"].get("conversation", 0)
                st.metric("Conversations", conv_count)
            
            with col4:
                other_count = sum(v for k, v in memory_stats["by_type"].items() 
                                if k not in ["document", "conversation"])
                st.metric("Other Content", other_count)
            
            # Content type breakdown
            st.subheader("Content Type Breakdown")
            
            if memory_stats["by_type"]:
                # Create a pie chart
                import plotly.express as px
                
                labels = list(memory_stats["by_type"].keys())
                values = list(memory_stats["by_type"].values())
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Memory Distribution by Content Type"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Document-specific stats
            documents = await self.doc_manager.list_documents()
            
            if documents:
                st.subheader("Document Details")
                
                doc_stats = []
                for doc in documents:
                    doc_stats.append({
                        "Title": doc.title,
                        "Chunks": doc.chunk_count,
                        "Size (KB)": round(doc.total_size / 1024, 1),
                        "Status": "Enabled" if doc.enabled else "Disabled",
                        "Upload Date": doc.uploaded_at.strftime("%Y-%m-%d")
                    })
                
                doc_df = pd.DataFrame(doc_stats)
                st.dataframe(doc_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Failed to load statistics: {str(e)}")

# Helper function to run the UI
def run_document_management():
    """Run the document management UI."""
    ui = DocumentManagementUI()
    ui.render()

if __name__ == "__main__":
    run_document_management()

