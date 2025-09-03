"""
Simplified search service for the core application.
Handles web search and basic search operations.
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import asyncio

load_dotenv()

class SearchService:
    """
    Simplified search service that handles:
    - Web search via DuckDuckGo
    - Basic search operations
    - Search result processing
    """
    
    def __init__(self):
        """Initialize search service."""
        self.max_results = 5
        self.timeout = 10
    
    async def web_search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform web search using DuckDuckGo."""
        try:
            if max_results is None:
                max_results = self.max_results
            
            # Use ddgs for DuckDuckGo search
            from ddgs import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    if len(results) >= max_results:
                        break
                    
                    result_data = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("body", ""),
                        "source": "duckduckgo"
                    }
                    results.append(result_data)
            
            return results
            
        except Exception as e:
            print(f"⚠️ Web search failed: {str(e)}")
            return []
    
    async def search_url_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from a specific URL."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract main content (simplified)
            content = ""
            
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to body text
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean up content
            if content:
                # Remove extra whitespace
                import re
                content = re.sub(r'\s+', ' ', content).strip()
                # Limit content length
                if len(content) > 5000:
                    content = content[:5000] + "..."
            
            return {
                "url": url,
                "title": title_text,
                "content": content,
                "source": "web_scrape"
            }
            
        except Exception as e:
            print(f"⚠️ URL content extraction failed: {str(e)}")
            return None
    
    async def search_with_context(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Perform search with additional context."""
        try:
            # Basic web search
            web_results = await self.web_search(query)
            
            # If context provided, try to enhance search
            enhanced_query = query
            if context:
                # Combine query with context for better results
                enhanced_query = f"{query} {context[:100]}"
                enhanced_results = await self.web_search(enhanced_query, max_results=3)
                
                # Merge results, prioritizing enhanced search
                all_results = enhanced_results + web_results
                # Remove duplicates based on URL
                seen_urls = set()
                unique_results = []
                for result in all_results:
                    if result["link"] not in seen_urls:
                        seen_urls.add(result["link"])
                        unique_results.append(result)
                
                web_results = unique_results[:self.max_results]
            
            return {
                "query": query,
                "enhanced_query": enhanced_query if context else query,
                "results": web_results,
                "total_results": len(web_results),
                "context_used": bool(context)
            }
            
        except Exception as e:
            print(f"⚠️ Contextual search failed: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e)
            }
    
    async def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query."""
        try:
            from ddgs import DDGS
            
            suggestions = []
            with DDGS() as ddgs:
                # Get autocomplete suggestions
                autocomplete_results = ddgs.autocomplete(query, max_results=5)
                for suggestion in autocomplete_results:
                    suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️ Search suggestions failed: {str(e)}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as readable text."""
        if not results:
            return "No search results found."
        
        formatted = f"Found {len(results)} results:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result.get('title', 'No title')}**\n"
            formatted += f"   URL: {result.get('link', 'No link')}\n"
            formatted += f"   {result.get('snippet', 'No description')}\n\n"
        
        return formatted
    
    def close(self):
        """Clean up resources."""
        pass
