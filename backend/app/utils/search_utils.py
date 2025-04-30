"""
Utilities for web search functionality.
Enhanced search capabilities to provide better context for user queries.
"""

import logging
import asyncio
import re
import json
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import aiohttp
from bs4 import BeautifulSoup
import httpx
from urllib.parse import quote_plus, urlparse
import time
from langdetect import detect, LangDetectException

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Common headers to avoid being blocked by search engines
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]

# Cache for search results to avoid duplicated searches
SEARCH_CACHE = {}

class SearchResult:
    """Structured search result."""
    def __init__(
        self, 
        title: str, 
        snippet: str, 
        url: str, 
        source: str,
        rank: int = 0
    ):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
        self.rank = rank
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source,
            "rank": self.rank
        }
    
    def format_for_context(self) -> str:
        """Format for inclusion in context."""
        return f"Source: {self.url}\nTitle: {self.title}\n{self.snippet}"

async def search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search using DuckDuckGo.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://duckduckgo.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=settings.web_search.search_timeout_seconds) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    rank = 0
                    
                    # Extract search results
                    for result in soup.select(".result__body"):
                        if len(results) >= max_results:
                            break
                            
                        title_elem = result.select_one(".result__title")
                        snippet_elem = result.select_one(".result__snippet")
                        link_elem = result.select_one(".result__url")
                        
                        if not title_elem or not snippet_elem:
                            continue
                            
                        title = title_elem.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True)
                        
                        # Get URL - different ways it might be available
                        url = None
                        if link_elem:
                            url = link_elem.get_text(strip=True)
                        else:
                            a_tag = title_elem.find("a")
                            if a_tag and a_tag.has_attr("href"):
                                url = a_tag["href"]
                                # Handle DuckDuckGo's redirects
                                if url.startswith("/"):
                                    url_match = re.search(r"uddg=([^&]+)", url)
                                    if url_match:
                                        url = url_match.group(1)
                        
                        if not url:
                            continue
                            
                        # Some basic URL normalization
                        if not url.startswith(("http://", "https://")):
                            url = "https://" + url
                            
                        rank += 1
                        results.append(SearchResult(
                            title=title,
                            snippet=snippet,
                            url=url,
                            source="duckduckgo",
                            rank=rank
                        ))
                    
                    return results
                else:
                    logger.warning(f"DuckDuckGo search failed with status {response.status}")
                    return []
    
    except asyncio.TimeoutError:
        logger.warning(f"DuckDuckGo search timed out for query: {query}")
        return []
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search: {str(e)}")
        return []

async def fetch_webpage_content(url: str, max_chars: int = 10000) -> Optional[str]:
    """
    Fetch and extract the main content from a webpage.
    
    Args:
        url: The URL to fetch
        max_chars: Maximum number of characters to return
        
    Returns:
        The extracted content or None if fetching failed
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                return None
                
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
                
            # Get text content
            text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if needed
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                
            return text
            
    except Exception as e:
        logger.error(f"Error fetching webpage content from {url}: {str(e)}")
        return None

async def search_web(query: str, use_cache: bool = True) -> str:
    """
    Search the web and return formatted results for use as context.
    
    Args:
        query: The search query
        use_cache: Whether to use cached results if available
        
    Returns:
        Formatted search results as context
    """
    # Check cache
    cache_key = query.lower().strip()
    if use_cache and cache_key in SEARCH_CACHE:
        cached_result = SEARCH_CACHE[cache_key]
        # Check if cache is still valid (less than 1 hour old)
        if time.time() - cached_result["timestamp"] < 3600:
            logger.info(f"Using cached search results for: {query}")
            return cached_result["content"]
    
    # Apply search settings
    max_results = settings.web_search.max_results
    
    # Detect query language
    query_language = "en"
    try:
        query_language = detect(query)
    except LangDetectException:
        pass
    
    # If not supported language, default to English
    if query_language not in settings.model.supported_languages:
        logger.info(f"Query language {query_language} not in supported languages, defaulting to English")
        query_language = "en"
    
    # Search with DuckDuckGo
    search_results = await search_duckduckgo(query, max_results)
    
    if not search_results:
        no_results = "No relevant information found on the web for this query."
        return no_results
    
    # Format results
    formatted_results = []
    for result in search_results:
        formatted_results.append(f"[{result.rank}] {result.title}\n{result.snippet}\nSource: {result.url}\n")
    
    context = "Web search results:\n\n" + "\n".join(formatted_results)
    
    # Cache the results
    SEARCH_CACHE[cache_key] = {
        "content": context,
        "timestamp": time.time(),
        "results": [r.to_dict() for r in search_results]
    }
    
    return context

def clear_search_cache():
    """Clear the search cache."""
    global SEARCH_CACHE
    SEARCH_CACHE = {}
    logger.info("Search cache cleared")

def get_search_stats() -> Dict[str, Any]:
    """Get statistics about the search cache."""
    return {
        "cache_size": len(SEARCH_CACHE),
        "cache_keys": list(SEARCH_CACHE.keys())[:10],  # First 10 keys
        "total_results_cached": sum(len(item["results"]) for item in SEARCH_CACHE.values())
    }
