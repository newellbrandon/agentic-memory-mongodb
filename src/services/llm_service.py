"""
Simplified LLM service for the core application.
Handles LLM interactions and embeddings generation.
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import asyncio

load_dotenv()

class LLMService:
    """
    Simplified LLM service that handles:
    - LLM interactions
    - Embeddings generation via Voyage AI
    - Basic text processing
    """
    
    def __init__(self):
        """Initialize LLM service."""
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not self.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY environment variable is required")
        
        # Get Voyage model from environment, with fallback
        self.voyage_model = os.getenv("VOYAGE_MODEL", "voyage-2")
        
        # Initialize Voyage AI client
        try:
            from voyageai import Client
            self.voyage_client = Client(api_key=self.voyage_api_key)
            print(f"✅ Voyage AI client initialized with model: {self.voyage_model}")
        except Exception as e:
            raise Exception(f"Failed to initialize Voyage AI: {e}")
        
        # Initialize LM Studio client for LLM responses
        self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL")
        self.lm_studio_model = os.getenv("LM_STUDIO_MODEL")
        if not self.lm_studio_base_url:
            print("⚠️ LM_STUDIO_BASE_URL not set, will use fallback responses")
        else:
            print(f"✅ LM Studio client initialized with base URL: {self.lm_studio_base_url}")
            print(f"✅ LM Studio model: {self.lm_studio_model}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Voyage AI."""
        try:
            # Use the configured model from environment
            embedding = self.voyage_client.embed(
                texts=[text],
                model=self.voyage_model,
                input_type="query"
            )
            return embedding.embeddings[0]
        except Exception as e:
            print(f"⚠️ Failed to generate embedding with model {self.voyage_model}: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 1024
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              context: Optional[str] = None) -> str:
        """Generate intelligent LLM response using LM Studio."""
        try:
            if not context:
                return "I understand your message. How can I help you further?"
            
            # Extract the user's question from messages
            user_message = messages[0].get("content", "") if messages else ""
            
            # If LM Studio is available, use it to generate intelligent responses
            if self.lm_studio_base_url:
                return await self._call_lm_studio(messages, context, user_message)
            else:
                # Fallback to simple response if LM Studio not available
                return f"Based on the context: {context}\n\nI understand your message. How can I help you further?"
                
        except Exception as e:
            return f"I encountered an error while processing the information: {str(e)}"
    
    async def _call_lm_studio(self, messages: List[Dict[str, str]], context: str, user_message: str) -> str:
        """Call LM Studio to generate intelligent response."""
        try:
            import aiohttp
            
            # Prepare the prompt for the LLM
            system_prompt = """You are an intelligent AI assistant that analyzes search results and provides coherent, helpful answers to user questions. 

When given search results, carefully read and analyze them to provide accurate, informative responses. Don't just list the results - synthesize the information and answer the user's question directly.

If the search results contain relevant information, use it to provide a comprehensive answer. If the information is incomplete or unclear, acknowledge what you found and what might be missing."""
            
            user_prompt = f"""User Question: {user_message}

Search Results:
{context}

IMPORTANT: When analyzing these search results, pay special attention to personal information entries (content_type: personal_info). These entries contain information about the current user that should be used to answer questions about their identity, preferences, or personal details.

If you find personal information about the user (like their name, preferences, location, etc.), use that information directly to answer their questions. Don't treat it as just data - it's information about the person you're talking to.

Please analyze these search results and provide a clear, intelligent answer to the user's question. Focus on answering the question directly rather than just listing what was found."""
            
            # Prepare the request payload
            payload = {
                "model": self.lm_studio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }
        
            # Make the request to LM Studio
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.lm_studio_base_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        print(f"⚠️ LM Studio API error: {response.status} - {error_text}")
                        return f"Based on the context: {context}\n\nI understand your message. How can I help you further?"
                        
        except Exception as e:
            print(f"⚠️ Failed to call LM Studio: {str(e)}")
            # Fallback response
            return f"Based on the context: {context}\n\nI understand your message. How can I help you further?"
    
    async def summarize_text(self, text: str) -> str:
        """Generate a summary of the given text."""
        try:
            # Simple summary logic - in production, use actual LLM
            words = text.split()
            if len(words) <= 50:
                return text
            
            # Take first and last sentences for summary
            sentences = text.split('.')
            if len(sentences) <= 2:
                return text
            
            summary = sentences[0] + "." + sentences[-1] + "."
            return summary.strip()
        except Exception as e:
            print(f"⚠️ Failed to summarize text: {str(e)}")
            return text[:200] + "..." if len(text) > 200 else text
    
    async def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        try:
            # Simple keyword extraction - in production, use NLP
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Return top 10 most common keywords
            from collections import Counter
            return [word for word, count in Counter(keywords).most_common(10)]
        except Exception as e:
            print(f"⚠️ Failed to extract keywords: {str(e)}")
            return []
    
    def close(self):
        """Clean up resources."""
        pass
