"""
Brave Search API tools for AI Deep Research Assistant.

This module provides enhanced web search functionality using the Brave Search API
with proper error handling, rate limiting, and result processing optimized for
research applications.
"""
import logging
import os
import httpx
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict

try:
    from ..config.settings import get_settings
except ImportError:
    # For direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured representation of a search result"""
    url: str
    title: str
    description: str
    relevance_score: float
    snippet: str = ""
    domain: str = ""
    published_date: Optional[str] = None
    content_type: Literal["web", "news", "academic"] = "web"
    accessed_at: str = ""
    
    def __post_init__(self):
        """Initialize computed fields after creation"""
        if not self.accessed_at:
            self.accessed_at = datetime.now(timezone.utc).isoformat()
        
        if not self.domain and self.url:
            try:
                from urllib.parse import urlparse
                self.domain = urlparse(self.url).netloc
            except Exception:
                self.domain = "unknown"
        
        # Use description as snippet if snippet is empty
        if not self.snippet:
            self.snippet = self.description[:200] + "..." if len(self.description) > 200 else self.description


class BraveSearchError(Exception):
    """Custom exception for Brave Search API errors"""
    pass


async def search_web_tool(
    api_key: str,
    query: str,
    count: int = 8,
    offset: int = 0,
    search_type: Literal["general", "news", "academic"] = "general",
    freshness: Optional[str] = None,
    country: str = "US",
    safe_search: Literal["strict", "moderate", "off"] = "moderate"
) -> List[Dict[str, Any]]:
    """
    Enhanced web search using Brave Search API with comprehensive error handling.
    
    Args:
        api_key: Brave Search API key
        query: Search query string
        count: Number of results to return (1-20, clamped to API limits)
        offset: Pagination offset for results
        search_type: Type of search (general, news, academic)
        freshness: Time filter (e.g., "24h", "7d", "30d", "1y")
        country: Country code for localized results
        safe_search: Safe search filter level
        
    Returns:
        List of structured search results as dictionaries
        
    Raises:
        BraveSearchError: For API-related errors
        ValueError: For invalid input parameters
    """
    # Input validation
    if not api_key or not api_key.strip():
        raise ValueError("Brave API key is required and cannot be empty")
    
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    # Ensure count is within Brave API limits (1-20)
    count = max(1, min(count, 20))
    offset = max(0, offset)
    
    # Prepare headers
    headers = {
        "X-Subscription-Token": api_key.strip(),
        "Accept": "application/json",
        "User-Agent": "AI-Deep-Research-Assistant/1.0"
    }
    
    # Prepare search parameters
    params = {
        "q": query.strip(),
        "count": count,
        "offset": offset,
        "country": country,
        "safesearch": safe_search,
        "search_lang": "en"
    }
    
    # Add freshness filter if specified
    if freshness:
        params["freshness"] = freshness
    
    # Adjust parameters based on search type
    if search_type == "news":
        params["freshness"] = freshness or "7d"  # Default to recent for news
    elif search_type == "academic":
        # Academic searches might need different parameters in the future
        pass
    
    logger.info(f"🔍 Searching Brave API: '{query}' (type: {search_type}, count: {count})")
    
    # Rate limiting: 1-second delay to prevent quota issues
    await asyncio.sleep(1.0)
    
    # Perform the search request
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            # Handle specific HTTP status codes
            if response.status_code == 429:
                # Return empty results with rate limit info instead of raising
                logger.warning(f"⚠️ Brave API rate limit hit for query: '{query}'")
                return [{
                    "url": "",
                    "title": "Search Temporarily Unavailable",
                    "description": "The Brave Search API rate limit has been exceeded. Please try again later or check your API quota.",
                    "relevance_score": 0.0,
                    "snippet": "Rate limit exceeded - search results unavailable",
                    "domain": "api.error",
                    "published_date": None,
                    "content_type": "error",
                    "accessed_at": datetime.now(timezone.utc).isoformat(),
                    "is_rate_limited": True
                }]
            
            if response.status_code == 401:
                error_msg = "Invalid Brave API key. Please check your API key configuration."
                logger.error(f"❌ {error_msg}")
                raise BraveSearchError(error_msg)
            
            if response.status_code == 400:
                error_msg = f"Bad request to Brave API: {response.text}"
                logger.error(f"❌ {error_msg}")
                raise BraveSearchError(error_msg)
            
            if response.status_code != 200:
                error_msg = f"Brave API returned status {response.status_code}: {response.text}"
                logger.error(f"❌ {error_msg}")
                raise BraveSearchError(error_msg)
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Brave API response as JSON: {e}"
                logger.error(f"❌ {error_msg}")
                raise BraveSearchError(error_msg)
            
            # Extract search results
            web_results = data.get("web", {}).get("results", [])
            news_results = data.get("news", {}).get("results", []) if search_type == "news" else []
            
            # Process and structure results
            processed_results = []
            all_results = web_results + news_results
            
            for idx, result in enumerate(all_results[:count]):
                try:
                    # Calculate relevance score based on position and other factors
                    position_score = 1.0 - (idx * 0.05)  # Decrease by 5% per position
                    position_score = max(position_score, 0.1)  # Minimum score of 10%
                    
                    # Create structured result
                    search_result = SearchResult(
                        url=result.get("url", ""),
                        title=result.get("title", "Untitled"),
                        description=result.get("description", ""),
                        relevance_score=position_score,
                        content_type="news" if result in news_results else "web",
                        published_date=result.get("published_date")
                    )
                    
                    # Convert to dictionary for return
                    result_dict = asdict(search_result)
                    processed_results.append(result_dict)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process search result {idx}: {e}")
                    continue
            
            logger.info(f"✅ Successfully retrieved {len(processed_results)} results for query: '{query}'")
            return processed_results
            
        except httpx.RequestError as e:
            error_msg = f"Network error during Brave API request: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise BraveSearchError(error_msg)
        
        except Exception as e:
            if isinstance(e, (BraveSearchError, ValueError)):
                raise  # Re-raise our custom exceptions
            
            error_msg = f"Unexpected error during Brave search: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise BraveSearchError(error_msg)


async def search_with_retry(
    api_key: str,
    query: str,
    max_retries: int = 3,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Search with automatic retry logic for resilient operation.
    
    Args:
        api_key: Brave Search API key
        query: Search query
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments passed to search_web_tool
        
    Returns:
        Search results or empty list if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await search_web_tool(api_key, query, **kwargs)
        
        except BraveSearchError as e:
            last_exception = e
            
            # Don't retry on authentication errors
            if "Invalid Brave API key" in str(e):
                logger.error("❌ Authentication error - not retrying")
                break
            
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                logger.warning(f"⚠️ Search attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ All {max_retries + 1} search attempts failed")
        
        except Exception as e:
            last_exception = e
            logger.error(f"❌ Unexpected error in search attempt {attempt + 1}: {e}")
            break
    
    # If we get here, all retries failed
    logger.error(f"❌ Search failed after {max_retries + 1} attempts: {last_exception}")
    return []  # Return empty list instead of raising to allow graceful degradation


def generate_search_cache_key(
    query: str,
    count: int = 8,
    search_type: str = "general",
    freshness: Optional[str] = None
) -> str:
    """
    Generate a cache key for search results to avoid duplicate API calls.
    
    Args:
        query: Search query
        count: Number of results
        search_type: Type of search
        freshness: Time filter
        
    Returns:
        SHA256 hash as cache key
    """
    cache_data = {
        "query": query.strip().lower(),
        "count": count,
        "search_type": search_type,
        "freshness": freshness
    }
    
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()[:16]  # First 16 chars


def extract_search_terms(query: str) -> List[str]:
    """
    Extract key search terms from a query for enhanced search strategies.
    
    Args:
        query: Original search query
        
    Returns:
        List of extracted key terms
    """
    # Simple term extraction - can be enhanced with NLP libraries
    import re
    
    # Remove common stop words and punctuation
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"
    }
    
    # Extract words (letters and numbers only)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    
    # Filter out stop words and short words
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return key_terms[:10]  # Limit to top 10 terms


if __name__ == "__main__":
    """Test search functionality when run directly"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_search():
        settings = get_settings()
        api_key = settings.brave_api_key
        
        if not api_key:
            print("❌ BRAVE_API_KEY not found in settings")
            print("   Add your API key to .env file to test search functionality")
            return
        
        print("🧪 Testing Brave Search API integration...")
        
        # Test basic search
        try:
            results = await search_web_tool(
                api_key=api_key,
                query="LangGraph multi-agent systems",
                count=3
            )
            
            print(f"✅ Search successful! Found {len(results)} results")
            
            for i, result in enumerate(results[:2], 1):
                print(f"\n📄 Result {i}:")
                print(f"  Title: {result['title']}")
                print(f"  URL: {result['url']}")
                print(f"  Score: {result['relevance_score']:.2f}")
                print(f"  Description: {result['description'][:100]}...")
        
        except Exception as e:
            print(f"❌ Search failed: {e}")
        
        # Test cache key generation
        cache_key = generate_search_cache_key("test query", 5, "general")
        print(f"\n🔑 Cache key example: {cache_key}")
        
        # Test term extraction
        terms = extract_search_terms("What is LangGraph and how does it work?")
        print(f"🔍 Extracted terms: {terms}")
    
    # Run the test
    asyncio.run(test_search())