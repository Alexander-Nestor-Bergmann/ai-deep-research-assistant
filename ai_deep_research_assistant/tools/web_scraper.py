"""
Web content scraping and summarization tools for research enhancement.

This module provides functionality to extract and process content from web pages
to supplement search results with deeper content analysis for better research quality.
"""
import logging
import httpx
import asyncio
import re
from typing import Dict, List
from datetime import datetime, timezone
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Structured representation of scraped web content"""
    url: str
    title: str
    content: str
    summary: str
    word_count: int
    content_type: str
    language: str = "en"
    scraped_at: str = ""
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        """Initialize computed fields after creation"""
        if not self.scraped_at:
            self.scraped_at = datetime.now(timezone.utc).isoformat()
        
        if not self.word_count:
            self.word_count = len(self.content.split())


class WebScrapingError(Exception):
    """Custom exception for web scraping errors"""
    pass


async def scrape_webpage(
    url: str,
    max_content_length: int = 10000,
    timeout: float = 15.0,
    user_agent: str = "AI-Deep-Research-Assistant/1.0"
) -> ScrapedContent:
    """
    Scrape content from a web page with robust error handling.
    
    Args:
        url: URL to scrape
        max_content_length: Maximum content length to extract
        timeout: Request timeout in seconds
        user_agent: User agent string for requests
        
    Returns:
        ScrapedContent object with extracted information
    """
    # Validate URL
    if not url or not url.strip():
        return ScrapedContent(
            url=url,
            title="",
            content="",
            summary="",
            word_count=0,
            content_type="error",
            success=False,
            error_message="Invalid URL provided"
        )
    
    url = url.strip()
    
    # Check URL format
    if not (url.startswith('http://') or url.startswith('https://')):
        return ScrapedContent(
            url=url,
            title="",
            content="",
            summary="",
            word_count=0,
            content_type="error",
            success=False,
            error_message="URL must start with http:// or https://"
        )
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    logger.info(f"🌐 Scraping webpage: {url}")
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True)
            
            # Check status code
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.reason_phrase}"
                logger.warning(f"⚠️ Failed to fetch {url}: {error_msg}")
                return ScrapedContent(
                    url=url,
                    title="",
                    content="",
                    summary="",
                    word_count=0,
                    content_type="error",
                    success=False,
                    error_message=error_msg
                )
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                error_msg = f"Unsupported content type: {content_type}"
                logger.warning(f"⚠️ {error_msg}")
                return ScrapedContent(
                    url=url,
                    title="",
                    content="",
                    summary="",
                    word_count=0,
                    content_type=content_type,
                    success=False,
                    error_message=error_msg
                )
            
            # Extract content
            html_content = response.text
            extracted_content = extract_content_from_html(html_content, max_content_length)
            
            # Generate summary
            summary = generate_summary(extracted_content['content'])
            
            return ScrapedContent(
                url=url,
                title=extracted_content['title'],
                content=extracted_content['content'],
                summary=summary,
                word_count=len(extracted_content['content'].split()),
                content_type="article" if len(extracted_content['content']) > 500 else "snippet",
                success=True
            )
            
        except httpx.TimeoutException:
            error_msg = f"Timeout after {timeout}s"
            logger.warning(f"⚠️ Scraping timeout for {url}: {error_msg}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                summary="",
                word_count=0,
                content_type="error",
                success=False,
                error_message=error_msg
            )
        
        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logger.warning(f"⚠️ Scraping failed for {url}: {error_msg}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                summary="",
                word_count=0,
                content_type="error", 
                success=False,
                error_message=error_msg
            )
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Scraping error for {url}: {error_msg}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                summary="",
                word_count=0,
                content_type="error",
                success=False,
                error_message=error_msg
            )


def extract_content_from_html(html: str, max_length: int = 10000) -> Dict[str, str]:
    """
    Extract title and main content from HTML using simple regex patterns.
    
    Args:
        html: Raw HTML content
        max_length: Maximum content length to extract
        
    Returns:
        Dictionary with 'title' and 'content' keys
    """
    # Extract title
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    title = ""
    if title_match:
        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    
    # Remove script and style tags
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove HTML tags but keep the text
    text_content = re.sub(r'<[^>]+>', ' ', html)
    
    # Normalize whitespace
    text_content = re.sub(r'\s+', ' ', text_content)
    
    # Remove common navigation/footer text patterns
    text_content = remove_boilerplate_text(text_content)
    
    # Limit content length
    if len(text_content) > max_length:
        text_content = text_content[:max_length] + "..."
    
    return {
        'title': title,
        'content': text_content.strip()
    }


def remove_boilerplate_text(text: str) -> str:
    """
    Remove common boilerplate text found in web pages.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text content
    """
    # Common boilerplate patterns to remove
    boilerplate_patterns = [
        r'cookie policy',
        r'privacy policy',
        r'terms of service',
        r'subscribe to our newsletter',
        r'follow us on',
        r'share this article',
        r'related articles',
        r'advertisement',
        r'sponsored content',
        r'© \d{4}',  # Copyright notices
        r'all rights reserved'
    ]
    
    # Remove boilerplate patterns (case insensitive)
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def generate_summary(content: str, max_sentences: int = 3) -> str:
    """
    Generate a simple extractive summary of the content.
    
    Args:
        content: Full text content
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summary text
    """
    if not content or len(content.strip()) < 100:
        return content.strip()
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return content.strip()
    
    # Simple scoring based on sentence length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Score based on position (earlier sentences get higher scores)
        position_score = 1.0 - (i / len(sentences)) * 0.5
        
        # Score based on length (medium-length sentences preferred)
        length_score = min(len(sentence) / 100, 1.0)
        
        total_score = position_score * 0.6 + length_score * 0.4
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = scored_sentences[:max_sentences]
    
    # Sort by original order to maintain coherence
    top_sentences.sort(key=lambda x: sentences.index(x[0]))
    
    summary = '. '.join([s[0] for s in top_sentences])
    return summary + '.' if summary else content[:200] + "..."


async def scrape_multiple_urls(
    urls: List[str],
    max_concurrent: int = 3,
    **kwargs
) -> List[ScrapedContent]:
    """
    Scrape multiple URLs concurrently with rate limiting.
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum concurrent requests
        **kwargs: Additional arguments passed to scrape_webpage
        
    Returns:
        List of ScrapedContent objects
    """
    if not urls:
        return []
    
    logger.info(f"🌐 Scraping {len(urls)} URLs with max concurrency {max_concurrent}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(url: str) -> ScrapedContent:
        async with semaphore:
            return await scrape_webpage(url, **kwargs)
    
    # Execute all scraping tasks concurrently
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and handle exceptions
    scraped_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Failed to scrape {urls[i]}: {result}")
            scraped_results.append(ScrapedContent(
                url=urls[i],
                title="",
                content="",
                summary="",
                word_count=0,
                content_type="error",
                success=False,
                error_message=str(result)
            ))
        else:
            scraped_results.append(result)
    
    successful = sum(1 for r in scraped_results if r.success)
    logger.info(f"✅ Successfully scraped {successful}/{len(urls)} URLs")
    
    return scraped_results


def is_scrapable_url(url: str) -> bool:
    """
    Check if a URL is likely to be scrapable (not a PDF, video, etc.)
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears scrapable
    """
    if not url:
        return False
    
    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    
    # Check for non-scrapable file extensions
    non_scrapable_extensions = {
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        '.zip', '.rar', '.tar', '.gz', '.mp4', '.avi', '.mov',
        '.mp3', '.wav', '.jpg', '.jpeg', '.png', '.gif', '.svg'
    }
    
    path_lower = parsed.path.lower()
    for ext in non_scrapable_extensions:
        if path_lower.endswith(ext):
            return False
    
    # Check for social media and other problematic domains
    problematic_domains = {
        'youtube.com', 'youtu.be', 'twitter.com', 'x.com',
        'facebook.com', 'instagram.com', 'linkedin.com',
        'tiktok.com', 'pinterest.com'
    }
    
    domain = parsed.netloc.lower()
    for prob_domain in problematic_domains:
        if prob_domain in domain:
            return False
    
    return True


if __name__ == "__main__":
    """Test web scraping functionality when run directly"""
    
    async def test_scraping():
        print("🧪 Testing web scraping functionality...")
        
        # Test URL validation
        test_urls = [
            "https://example.com",  # Should work
            "invalid-url",          # Should fail
            "https://example.com/document.pdf"  # Should be detected as non-scrapable
        ]
        
        for url in test_urls:
            scrapable = is_scrapable_url(url)
            print(f"🔍 URL: {url} - Scrapable: {scrapable}")
        
        # Test content extraction
        sample_html = """
        <html>
            <head><title>Test Article Title</title></head>
            <body>
                <script>alert('test');</script>
                <h1>Main Heading</h1>
                <p>This is the first paragraph of content.</p>
                <p>This is the second paragraph with more details.</p>
                <footer>© 2024 All rights reserved</footer>
            </body>
        </html>
        """
        
        extracted = extract_content_from_html(sample_html)
        print(f"📄 Extracted title: {extracted['title']}")
        print(f"📄 Extracted content: {extracted['content'][:100]}...")
        
        # Test summary generation
        sample_content = "This is a long article about machine learning. It covers various topics including neural networks and deep learning. The article explains how these technologies work and their applications in different fields."
        summary = generate_summary(sample_content, max_sentences=2)
        print(f"📋 Generated summary: {summary}")
    
    # Run the test
    asyncio.run(test_scraping())