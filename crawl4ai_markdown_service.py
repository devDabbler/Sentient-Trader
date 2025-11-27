"""
🚀 Enhanced Crawl4AI Markdown Generation Service
================================================

This service leverages Crawl4AI's advanced markdown generation capabilities to provide
clean, structured content for LLM analysis. It implements the latest v0.7.x features
including content filtering, BM25 ranking, and optimized markdown generation.

Features:
- Advanced markdown generation with content filtering
- BM25-based content ranking for relevance
- Pruning filters to remove noise and boilerplate
- Optimized for LinkedIn profile extraction
- Integration with existing LLM analysis pipeline
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Crawl4AI imports
try:
    from crawl4ai import (
        AsyncWebCrawler, 
        CrawlerRunConfig,
        DefaultMarkdownGenerator,
        BM25ContentFilter, 
        PruningContentFilter,
        LLMExtractionStrategy,
        BrowserConfig,
        CacheMode,
        LLMConfig
    )
    CRAWL4AI_AVAILABLE = True
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    logging.warning(f"Crawl4AI not available: {e}")

# Import Chrome Profile Manager for authenticated sessions
try:
    from src.services.chrome_profile_manager import ChromeProfileManager
    CHROME_PROFILE_AVAILABLE = True
except ImportError as e:
    CHROME_PROFILE_AVAILABLE = False
    logging.warning(f"Chrome Profile Manager not available: {e}")

logger = logging.getLogger(__name__)

class ContentFilterType(Enum):
    """Types of content filters available"""
    NONE = "none"
    PRUNING = "pruning"
    BM25 = "bm25"
    COMBINED = "combined"

@dataclass
class MarkdownGenerationResult:
    """Result of markdown generation with metadata"""
    raw_markdown: str
    filtered_markdown: str
    references_markdown: str
    content_quality_score: float
    processing_time: float
    filter_type: ContentFilterType
    word_count: int
    metadata: Dict[str, Any]

class Crawl4AIMarkdownService:
    """
    Enhanced Crawl4AI service with advanced markdown generation capabilities.
    
    This service provides clean, structured markdown content optimized for LLM analysis
    by leveraging Crawl4AI's content filtering and markdown generation features.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-4o-mini", browser_config: Optional[BrowserConfig] = None, existing_driver: Optional[Any] = None):
        """
        Initialize the Crawl4AI markdown service.
        
        Args:
            api_key: API key for LLM services (optional)
            model: LLM model to use for content analysis
            browser_config: Browser configuration to use (for authenticated sessions)
            existing_driver: Existing Selenium WebDriver to use instead of creating new browser
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("Crawl4AI is not available. Please install with: pip install crawl4ai>=0.7.0")
        
        self.api_key = api_key
        self.model = model
        self.browser_config = browser_config
        self.existing_driver = existing_driver  # Store existing driver
        self.crawler = None
        self._shared_browser = None  # For tab-based browsing
        
        logger.info("✅ Crawl4AI Markdown Service initialized")
    
    @staticmethod
    def create_authenticated_browser_config() -> Optional[BrowserConfig]:
        """
        Create a BrowserConfig that uses the existing authenticated Chrome session.
        
        Returns:
            BrowserConfig configured to use the existing session, or None if not available
        """
        # Return None to force using existing browser session instead of creating new ones
        logger.info("🔐 Using existing authenticated browser session (no new browser creation)")
        return None
    
    async def __aenter__(self):
        """Async context manager entry - use existing browser session with tab support"""
        # Always use default configuration to avoid creating new browsers
        logger.info("🔧 Using default browser configuration to maintain existing session")
        self.crawler = AsyncWebCrawler()
        
        await self.crawler.__aenter__()
        logger.info("🔗 Using existing browser session (no new browser creation)")
        
        # Store reference to browser for tab operations
        if hasattr(self.crawler, 'browser'):
            self._shared_browser = self.crawler.browser
        elif hasattr(self.crawler, '_browser'):
            self._shared_browser = self.crawler._browser
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def generate_clean_markdown(
        self,
        url: str,
        filter_type: ContentFilterType = ContentFilterType.PRUNING,
        user_query: Optional[str] = None,
        markdown_options: Optional[Dict[str, Any]] = None
    ) -> MarkdownGenerationResult:
        """
        Generate clean, filtered markdown with progressive fallback strategy.
        
        Args:
            url: URL to crawl and convert to markdown
            filter_type: Type of content filtering to apply
            user_query: Query for BM25 filtering (required if filter_type is BM25 or COMBINED)
            markdown_options: Custom options for markdown generation
            
        Returns:
            MarkdownGenerationResult with clean markdown content
        """
        start_time = time.time()
        
        # Check if we should use existing driver to navigate
        if self.existing_driver:
            logger.info(f"🔗 Using existing driver to navigate to: {url}")
            try:
                # Navigate existing driver to the URL
                self.existing_driver.get(url)
                
                # Wait for page to load
                import time as time_module
                time_module.sleep(3)  # Wait 3 seconds for page load
                
                # Get the HTML from the profile page
                raw_html = self.existing_driver.page_source
                logger.info(f"✅ Successfully navigated to profile page, got {len(raw_html)} chars of HTML")
                
                # Use HTML-based processing with adaptive escalation
                return await self._adaptive_markdown_from_html(
                    raw_html=raw_html,
                    url=url,
                    filter_type=filter_type,
                    user_query=user_query,
                    markdown_options=markdown_options
                )
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to navigate existing driver to {url}: {e}")
                logger.info("🔄 Falling back to standard Crawl4AI navigation")
        
        # Original URL-based approach
        if not self.crawler:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        # Default markdown options optimized for LinkedIn profiles
        default_options = {
            "ignore_links": False,  # Keep links for LinkedIn profiles
            "ignore_images": True,  # Remove images to focus on text
            "escape_html": True,
            "body_width": 0,  # No line wrapping
            "skip_internal_links": True,
            "include_sup_sub": True,
            "mark_code": True,
            "handle_code_in_pre": True
        }
        
        if markdown_options:
            default_options.update(markdown_options)
        
        # Progressive fallback strategy
        fallback_strategies = []
        
        if filter_type == ContentFilterType.COMBINED:
            if not user_query:
                raise ValueError("user_query is required for combined filtering")
            fallback_strategies = [
                (ContentFilterType.COMBINED, "Hybrid filtering (pruning + BM25)"),
                (ContentFilterType.PRUNING, "Pruning filter fallback"),
                (ContentFilterType.BM25, "BM25 filter fallback"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        elif filter_type == ContentFilterType.BM25:
            if not user_query:
                raise ValueError("user_query is required for BM25 filtering")
            fallback_strategies = [
                (ContentFilterType.BM25, "BM25 filtering"),
                (ContentFilterType.PRUNING, "Pruning filter fallback"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        elif filter_type == ContentFilterType.PRUNING:
            fallback_strategies = [
                (ContentFilterType.PRUNING, "Pruning filtering"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        else:
            fallback_strategies = [
                (ContentFilterType.NONE, "Basic extraction")
            ]
        
        best_result = None
        best_quality = 0.0
        
        for strategy_filter, strategy_name in fallback_strategies:
            try:
                logger.info(f"🔄 Trying {strategy_name}...")
                
                # Generate markdown with current strategy
                if strategy_filter == ContentFilterType.NONE:
                    result = await self._generate_basic_markdown(url, default_options)
                elif strategy_filter == ContentFilterType.PRUNING:
                    result = await self._generate_pruned_markdown(url, default_options)
                elif strategy_filter == ContentFilterType.BM25:
                    result = await self._generate_bm25_markdown(url, user_query, default_options)
                elif strategy_filter == ContentFilterType.COMBINED:
                    result = await self._generate_combined_markdown(url, user_query, default_options)
                else:
                    continue
                
                # Calculate quality metrics
                markdown_text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
                quality_score = self._calculate_content_quality(markdown_text)
                word_count = len(markdown_text.split())
                
                logger.info(f"📊 {strategy_name} result: Quality {quality_score:.1f}%, Words {word_count}")
                
                # Check if this result meets minimum quality threshold
                if quality_score >= 60.0 and word_count >= 50:
                    logger.info(f"✅ {strategy_name} achieved target quality!")
                    best_result = result
                    best_quality = quality_score
                    break
                
                # Track best result so far
                if quality_score > best_quality:
                    best_result = result
                    best_quality = quality_score
                
            except Exception as e:
                logger.warning(f"⚠️ {strategy_name} failed: {e}")
                continue
        
        # Use best result found
        if best_result is None:
            logger.error("❌ All fallback strategies failed")
            processing_time = time.time() - start_time
            return MarkdownGenerationResult(
                raw_markdown="",
                filtered_markdown="",
                references_markdown="",
                content_quality_score=0.0,
                processing_time=processing_time,
                filter_type=filter_type,
                word_count=0,
                metadata={
                    'url': url,
                    'success': False,
                    'error_message': 'All extraction strategies failed'
                }
            )
        
        # Final quality check and metadata
        markdown_text = best_result.markdown if isinstance(best_result.markdown, str) else str(best_result.markdown)
        final_quality = self._calculate_content_quality(markdown_text)
        word_count = len(markdown_text.split())
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Final markdown generation complete in {processing_time:.2f}s - "
                   f"Quality: {final_quality:.1f}%, Words: {word_count}")
        
        return MarkdownGenerationResult(
            raw_markdown=markdown_text,
            filtered_markdown=markdown_text,
            references_markdown=getattr(best_result.markdown, 'references_markdown', ''),
            content_quality_score=final_quality,
            processing_time=processing_time,
            filter_type=filter_type,
            word_count=word_count,
            metadata={
                'url': url,
                'success': best_result.success,
                'error_message': getattr(best_result, 'error_message', None),
                'fallback_used': best_quality < 60.0 or word_count < 50,
                'best_quality_achieved': best_quality
            }
        )

    async def _adaptive_markdown_from_html(
        self,
        raw_html: str,
        url: str,
        filter_type: ContentFilterType = ContentFilterType.PRUNING,
        user_query: Optional[str] = None,
        markdown_options: Optional[Dict[str, Any]] = None
    ) -> MarkdownGenerationResult:
        """
        Adaptive HTML strategy: progressively escalate extraction strength until
        acceptable quality is reached or tiers are exhausted.
        """
        start_time = time.time()

        # Target thresholds tuned for LinkedIn profiles
        target_quality = 60.0
        target_words = 100

        # Prepare options
        default_options = {
            "ignore_links": False,
            "ignore_images": True,
            "escape_html": True,
            "body_width": 0,
            "skip_internal_links": True,
            "include_sup_sub": True,
            "mark_code": True,
            "handle_code_in_pre": True,
            "strip_empty_lines": True,
            "normalize_whitespace": True,
            "preserve_tables": True
        }
        if markdown_options:
            default_options.update(markdown_options)

        best_result: Optional[Any] = None
        best_quality = 0.0

        # Tier 1: Pruning (fast)
        try:
            result = await self._generate_pruned_markdown_from_html(raw_html, default_options)
            text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
            q = self._calculate_content_quality(text)
            w = len(text.split())
            if q > best_quality:
                best_result, best_quality = result, q
            if q >= target_quality and w >= target_words:
                processing_time = time.time() - start_time
                return MarkdownGenerationResult(text, text, getattr(result.markdown, 'references_markdown', ''), q, processing_time, filter_type, w, { 'url': url, 'success': True, 'method': 'html_pruning' })
        except Exception as e:
            logger.warning(f"Adaptive T1 failed: {e}")

        # Tier 2: BM25 or Combined when user_query is available
        if user_query:
            try:
                # Prefer COMBINED for stronger filtering
                result = await self._generate_combined_markdown_from_html(raw_html, user_query, default_options)
                text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
                q = self._calculate_content_quality(text)
                w = len(text.split())
                if q > best_quality:
                    best_result, best_quality = result, q
                if q >= target_quality and w >= target_words:
                    processing_time = time.time() - start_time
                    return MarkdownGenerationResult(text, text, getattr(result.markdown, 'references_markdown', ''), q, processing_time, ContentFilterType.COMBINED, w, { 'url': url, 'success': True, 'method': 'html_combined' })
            except Exception as e:
                logger.warning(f"Adaptive T2 failed: {e}")
        else:
            try:
                # If no query, attempt a stronger basic pass
                result = await self._generate_basic_markdown_from_html(raw_html, default_options)
                text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
                q = self._calculate_content_quality(text)
                w = len(text.split())
                if q > best_quality:
                    best_result, best_quality = result, q
                if q >= target_quality and w >= target_words:
                    processing_time = time.time() - start_time
                    return MarkdownGenerationResult(text, text, getattr(result.markdown, 'references_markdown', ''), q, processing_time, ContentFilterType.NONE, w, { 'url': url, 'success': True, 'method': 'html_basic' })
            except Exception as e:
                logger.warning(f"Adaptive basic failed: {e}")

        # Tier 3: BM25 (even if combined failed) as last attempt when query present
        if user_query:
            try:
                result = await self._generate_bm25_markdown_from_html(raw_html, user_query, default_options)
                text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
                q = self._calculate_content_quality(text)
                w = len(text.split())
                if q > best_quality:
                    best_result, best_quality = result, q
                if q >= target_quality and w >= target_words:
                    processing_time = time.time() - start_time
                    return MarkdownGenerationResult(text, text, getattr(result.markdown, 'references_markdown', ''), q, processing_time, ContentFilterType.BM25, w, { 'url': url, 'success': True, 'method': 'html_bm25' })
            except Exception as e:
                logger.warning(f"Adaptive T3 failed: {e}")

        # Fallback: return best we have
        processing_time = time.time() - start_time
        if best_result is not None:
            text = best_result.markdown if isinstance(best_result.markdown, str) else str(best_result.markdown)
            w = len(text.split())
            return MarkdownGenerationResult(
                raw_markdown=text,
                filtered_markdown=text,
                references_markdown=getattr(best_result.markdown, 'references_markdown', ''),
                content_quality_score=best_quality,
                processing_time=processing_time,
                filter_type=filter_type,
                word_count=w,
                metadata={ 'url': url, 'success': True, 'method': 'html_best_effort', 'best_quality_achieved': best_quality }
            )
        # Nothing worked
        return MarkdownGenerationResult(
            raw_markdown="",
            filtered_markdown="",
            references_markdown="",
            content_quality_score=0.0,
            processing_time=processing_time,
            filter_type=filter_type,
            word_count=0,
            metadata={ 'url': url, 'success': False, 'error_message': 'Adaptive HTML strategies failed' }
        )
    async def _generate_markdown_from_html(
        self,
        raw_html: str,
        url: str,
        filter_type: ContentFilterType = ContentFilterType.PRUNING,
        user_query: Optional[str] = None,
        markdown_options: Optional[Dict[str, Any]] = None
    ) -> MarkdownGenerationResult:
        """
        Generate markdown from raw HTML content instead of navigating to URL.
        
        Args:
            raw_html: Raw HTML content from the profile page
            url: Original URL (for metadata)
            filter_type: Type of content filtering to apply
            user_query: Query for BM25 filtering
            markdown_options: Custom options for markdown generation
            
        Returns:
            MarkdownGenerationResult with processed content
        """
        start_time = time.time()
        
        # Default markdown options optimized for LinkedIn profiles
        default_options = {
            "ignore_links": False,  # Keep links for LinkedIn profiles
            "ignore_images": True,  # Remove images to focus on text
            "escape_html": True,
            "body_width": 0,  # No line wrapping
            "skip_internal_links": True,
            "include_sup_sub": True,
            "mark_code": True,
            "handle_code_in_pre": True
        }
        
        if markdown_options:
            default_options.update(markdown_options)
        
        # Progressive fallback strategy
        fallback_strategies = []
        
        if filter_type == ContentFilterType.COMBINED:
            if not user_query:
                raise ValueError("user_query is required for combined filtering")
            fallback_strategies = [
                (ContentFilterType.COMBINED, "Hybrid filtering (pruning + BM25)"),
                (ContentFilterType.PRUNING, "Pruning filter fallback"),
                (ContentFilterType.BM25, "BM25 filter fallback"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        elif filter_type == ContentFilterType.BM25:
            if not user_query:
                raise ValueError("user_query is required for BM25 filtering")
            fallback_strategies = [
                (ContentFilterType.BM25, "BM25 filtering"),
                (ContentFilterType.PRUNING, "Pruning filter fallback"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        elif filter_type == ContentFilterType.PRUNING:
            fallback_strategies = [
                (ContentFilterType.PRUNING, "Pruning filtering"),
                (ContentFilterType.NONE, "Basic extraction fallback")
            ]
        else:
            fallback_strategies = [
                (ContentFilterType.NONE, "Basic extraction")
            ]
        
        best_result = None
        best_quality = 0.0
        
        for strategy_filter, strategy_name in fallback_strategies:
            try:
                logger.info(f"🔄 Trying {strategy_name} from HTML...")
                
                # Generate markdown with current strategy using HTML
                if strategy_filter == ContentFilterType.NONE:
                    result = await self._generate_basic_markdown_from_html(raw_html, default_options)
                elif strategy_filter == ContentFilterType.PRUNING:
                    result = await self._generate_pruned_markdown_from_html(raw_html, default_options)
                elif strategy_filter == ContentFilterType.BM25:
                    result = await self._generate_bm25_markdown_from_html(raw_html, user_query, default_options)
                elif strategy_filter == ContentFilterType.COMBINED:
                    result = await self._generate_combined_markdown_from_html(raw_html, user_query, default_options)
                else:
                    continue
                
                # Calculate quality metrics
                markdown_text = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
                quality_score = self._calculate_content_quality(markdown_text)
                word_count = len(markdown_text.split())
                
                logger.info(f"📊 {strategy_name} result: Quality {quality_score:.1f}%, Words {word_count}")
                
                # Check if this result meets minimum quality threshold
                if quality_score >= 60.0 and word_count >= 50:
                    logger.info(f"✅ {strategy_name} achieved target quality!")
                    best_result = result
                    best_quality = quality_score
                    break
                
                # Track best result so far
                if quality_score > best_quality:
                    best_result = result
                    best_quality = quality_score
                
            except Exception as e:
                logger.warning(f"⚠️ {strategy_name} failed: {e}")
                continue
        
        # Use best result found
        if best_result is None:
            logger.error("❌ All HTML-based strategies failed")
            processing_time = time.time() - start_time
            return MarkdownGenerationResult(
                raw_markdown="",
                filtered_markdown="",
                references_markdown="",
                content_quality_score=0.0,
                processing_time=processing_time,
                filter_type=filter_type,
                word_count=0,
                metadata={
                    'url': url,
                    'success': False,
                    'error_message': 'All HTML extraction strategies failed'
                }
            )
        
        # Final quality check and metadata
        markdown_text = best_result.markdown if isinstance(best_result.markdown, str) else str(best_result.markdown)
        final_quality = self._calculate_content_quality(markdown_text)
        word_count = len(markdown_text.split())
        processing_time = time.time() - start_time
        
        logger.info(f"✅ HTML-based markdown generation complete in {processing_time:.2f}s - "
                   f"Quality: {final_quality:.1f}%, Words: {word_count}")
        
        return MarkdownGenerationResult(
            raw_markdown=markdown_text,
            filtered_markdown=markdown_text,
            references_markdown=getattr(best_result.markdown, 'references_markdown', ''),
            content_quality_score=final_quality,
            processing_time=processing_time,
            filter_type=filter_type,
            word_count=word_count,
            metadata={
                'url': url,
                'success': best_result.success,
                'error_message': getattr(best_result, 'error_message', None),
                'fallback_used': best_quality < 60.0 or word_count < 50,
                'best_quality_achieved': best_quality,
                'method': 'html_based'
            }
        )
        """Generate basic markdown without content filtering with enhanced loading"""
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options=options
        )
        
        # Import optimized configuration
        from .optimized_crawl4ai_config import create_optimized_crawler_config
        
        # Use optimized configuration with URL for page-specific selectors  
        logger.info(f"🔧 Creating optimized config for URL: {url}")
        config, _ = create_optimized_crawler_config("fast", url=url)
        
        # Override with markdown-specific settings
        config.markdown_generator = md_generator
        config.page_timeout = 120000  # Keep longer timeout for markdown generation
        config.wait_until = "networkidle"  # Wait for full page load
        config.delay_before_return_html = 2.0  # Wait for dynamic content
        config.js_code = [
            # Scroll to trigger lazy loading
            "window.scrollTo(0, document.body.scrollHeight);",
            "await new Promise(resolve => setTimeout(resolve, 1000));",
            "window.scrollTo(0, 0);",
            "await new Promise(resolve => setTimeout(resolve, 1000));"
        ]
        
        return await self.crawler.arun(url, config=config)
    
    async def _generate_pruned_markdown(self, url: str, options: Dict[str, Any]) -> Any:
        """Generate markdown with enhanced pruning filter using Fit Markdown features"""
        # Enhanced pruning filter for LinkedIn profiles
        pruning_filter = PruningContentFilter(
            threshold=0.45,  # Lower threshold for more content retention
            min_word_threshold=10  # Lower minimum for profile content
        )
        
        # Create markdown generator with pruning filter
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options=options
        )
        
        # Import optimized configuration
        from .optimized_crawl4ai_config import create_optimized_crawler_config
        
        # Use optimized configuration with URL for page-specific selectors
        logger.info(f"🔧 Creating optimized config for URL: {url}")
        config, _ = create_optimized_crawler_config("fast", url=url)
        
        # Override with pruned markdown-specific settings
        config.markdown_generator = md_generator
        config.word_count_threshold = 20
        config.page_timeout = 30000  # 30 seconds timeout (much faster)
        config.wait_until = "domcontentloaded"  # Faster loading
        config.delay_before_return_html = 1.0  # Faster processing
        config.js_code = [
            # Scroll to trigger lazy loading
            "window.scrollTo(0, document.body.scrollHeight);",
            "await new Promise(resolve => setTimeout(resolve, 1500));",
            "window.scrollTo(0, 0);",
            "await new Promise(resolve => setTimeout(resolve, 1500));",
            # Expand "Show more" sections
            "document.querySelectorAll('[aria-expanded=\\\"false\\\"]').forEach(btn => btn.click());",
            "await new Promise(resolve => setTimeout(resolve, 1000));"
        ]
        
        result = await self.crawler.arun(url, config=config)
        
        if not result.success or not result.html:
            return result
        
        # Apply Fit Markdown pruning filter
        try:
            pruned_chunks = pruning_filter.filter_content(result.html)
            if pruned_chunks:
                pruned_html = "\n".join(pruned_chunks)
                
                # Generate markdown from pruned HTML
                md_generator_filtered = DefaultMarkdownGenerator(
                    content_source="raw_html",
                    options=options
                )
                
                result.html = pruned_html
                result.markdown = md_generator_filtered.generate_markdown(pruned_html)
                
                # Add Fit Markdown metadata
                if hasattr(result, 'markdown'):
                    result.markdown.fit_markdown = pruned_html
            
        except Exception as e:
            logger.warning(f"Pruning filter failed, using original content: {e}")
        
        return result
    
    async def _generate_bm25_markdown(self, url: str, user_query: str, options: Dict[str, Any]) -> Any:
        """Generate markdown with enhanced BM25 filtering for job relevance"""
        # Enhanced BM25 filter for job-specific relevance
        bm25_filter = BM25ContentFilter(
            user_query=user_query,
            bm25_threshold=1.2,  # Balanced relevance threshold
            language="english"
        )
        
        # Create markdown generator
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options=options
        )
        
        # Enhanced configuration for job-relevant content
        config = CrawlerRunConfig(
            markdown_generator=md_generator,
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=20,
            page_timeout=120000,
            wait_until="networkidle",
            # LinkedIn-specific optimizations
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            exclude_external_links=True,
            # Enhanced loading strategy
            delay_before_return_html=1.0,  # Faster processing
            js_code=[
                # Scroll to trigger lazy loading
                "window.scrollTo(0, document.body.scrollHeight);",
                "await new Promise(resolve => setTimeout(resolve, 1500));",
                "window.scrollTo(0, 0);",
                "await new Promise(resolve => setTimeout(resolve, 1500));",
                # Expand profile sections
                "document.querySelectorAll('[aria-expanded=\\\"false\\\"]').forEach(btn => btn.click());",
                "await new Promise(resolve => setTimeout(resolve, 1000));"
            ]
        )
        
        result = await self.crawler.arun(url, config=config)
        
        if not result.success or not result.html:
            return result
        
        # Apply BM25 filter for job relevance
        try:
            bm25_chunks = bm25_filter.filter_content(result.html)
            
            if not bm25_chunks:
                logger.warning("No content matched BM25 query, using pruned content")
                # Fallback to pruning filter
                pruning_filter = PruningContentFilter(
                    threshold=0.45,
                    min_word_threshold=10
                )
                bm25_chunks = pruning_filter.filter_content(result.html)
            
            if bm25_chunks:
                filtered_html = "\n---\n".join(bm25_chunks)
                
                # Generate markdown from filtered HTML
                md_generator_filtered = DefaultMarkdownGenerator(
                    content_source="raw_html",
                    options=options
                )
                
                result.html = filtered_html
                result.markdown = md_generator_filtered.generate_markdown(filtered_html)
                
                # Add Fit Markdown metadata
                if hasattr(result, 'markdown'):
                    result.markdown.fit_markdown = filtered_html
            
        except Exception as e:
            logger.warning(f"BM25 filter failed, using original content: {e}")
        
        return result
    
    async def _generate_combined_markdown(self, url: str, user_query: str, options: Dict[str, Any]) -> Any:
        """Generate markdown with enhanced hybrid filtering strategy"""
        # Step 1: Enhanced pruning filter
        pruning_filter = PruningContentFilter(
            threshold=0.45,  # Lower threshold for more content retention
            min_word_threshold=10  # Lower minimum for profile content
        )
        
        # Step 2: Enhanced BM25 filter for job relevance
        bm25_filter = BM25ContentFilter(
            user_query=user_query,
            bm25_threshold=1.2,
            language="english"
        )
        
        # Create markdown generator
        md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options=options
        )
        
        # Enhanced configuration with hybrid filtering
        config = CrawlerRunConfig(
            markdown_generator=md_generator,
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=20,
            page_timeout=30000,  # 30 seconds timeout (much faster)
            wait_until="domcontentloaded",  # Faster loading
            # LinkedIn-specific optimizations
            excluded_tags=["nav", "footer", "header", "aside", "script", "style", "noscript"],
            exclude_external_links=True,
            # Enhanced loading strategy
            delay_before_return_html=1.0,  # Faster processing
            js_code=[
                # Scroll to trigger lazy loading
                "window.scrollTo(0, document.body.scrollHeight);",
                "await new Promise(resolve => setTimeout(resolve, 1500));",
                "window.scrollTo(0, 0);",
                "await new Promise(resolve => setTimeout(resolve, 1500));",
                # Expand all profile sections
                "document.querySelectorAll('[aria-expanded=\\\"false\\\"]').forEach(btn => btn.click());",
                "await new Promise(resolve => setTimeout(resolve, 1000));",
                # Trigger any "Show more" buttons
                "document.querySelectorAll('[data-test-id=\\\"show-more\\\"]').forEach(btn => btn.click());",
                "await new Promise(resolve => setTimeout(resolve, 1000));"
            ]
        )
        
        result = await self.crawler.arun(url, config=config)
        
        if not result.success or not result.html:
            return result
        
        try:
            # Step 1: Apply pruning filter
            pruned_chunks = pruning_filter.filter_content(result.html)
            if not pruned_chunks:
                logger.warning("Pruning filter returned no content, using original")
                pruned_html = result.html
            else:
                pruned_html = "\n".join(pruned_chunks)
            
            # Step 2: Apply BM25 filter to pruned content
            bm25_chunks = bm25_filter.filter_content(pruned_html)
            
            if not bm25_chunks:
                logger.warning("No content matched BM25 query after pruning, using pruned content")
                filtered_html = pruned_html
            else:
                filtered_html = "\n---\n".join(bm25_chunks)
            
            # Generate markdown from filtered HTML
            md_generator_filtered = DefaultMarkdownGenerator(
                content_source="raw_html",
                options=options
            )
            
            result.html = filtered_html
            result.markdown = md_generator_filtered.generate_markdown(filtered_html)
            
            # Add Fit Markdown metadata
            if hasattr(result, 'markdown'):
                result.markdown.fit_markdown = filtered_html
            
            logger.info(f"✅ Hybrid filtering applied: {len(pruned_chunks) if pruned_chunks else 0} pruned chunks, {len(bm25_chunks) if bm25_chunks else 0} BM25 chunks")
            
        except Exception as e:
            logger.warning(f"Hybrid filtering failed, using original content: {e}")
        
        return result
    
    async def _generate_basic_markdown_from_html(self, raw_html: str, options: Dict[str, Any]) -> Any:
        """Generate basic markdown from raw HTML content instead of navigating to URL"""
        md_generator = DefaultMarkdownGenerator(
            content_source="raw_html",
            options=options
        )
        
        # Create a mock result object with the HTML
        class MockResult:
            def __init__(self, html_content):
                self.success = True
                self.html = html_content
                self.markdown = md_generator.generate_markdown(html_content)
        
        return MockResult(raw_html)
    
    async def _generate_pruned_markdown_from_html(self, raw_html: str, options: Dict[str, Any]) -> Any:
        """Generate pruned markdown from raw HTML content"""
        # Enhanced pruning filter for LinkedIn profiles
        pruning_filter = PruningContentFilter(
            threshold=0.45,  # Lower threshold for more content retention
            min_word_threshold=10  # Lower minimum for profile content
        )
        
        # Create markdown generator
        md_generator = DefaultMarkdownGenerator(
            content_source="raw_html",
            options=options
        )
        
        # Create a mock result object
        class MockResult:
            def __init__(self, html_content):
                self.success = True
                self.html = html_content
                self.markdown = md_generator.generate_markdown(html_content)
        
        result = MockResult(raw_html)
        
        # Apply Fit Markdown pruning filter
        try:
            pruned_chunks = pruning_filter.filter_content(result.html)
            if pruned_chunks:
                pruned_html = "\n".join(pruned_chunks)
                
                # Generate markdown from pruned HTML
                md_generator_filtered = DefaultMarkdownGenerator(
                    content_source="raw_html",
                    options=options
                )
                
                result.html = pruned_html
                result.markdown = md_generator_filtered.generate_markdown(pruned_html)
                
                # Add Fit Markdown metadata
                if hasattr(result, 'markdown'):
                    result.markdown.fit_markdown = pruned_html
            
        except Exception as e:
            logger.warning(f"Pruning filter failed, using original content: {e}")
        
        return result
    
    async def _generate_bm25_markdown_from_html(self, raw_html: str, user_query: str, options: Dict[str, Any]) -> Any:
        """Generate BM25-filtered markdown from raw HTML content"""
        # Enhanced BM25 filter for job-specific relevance
        bm25_filter = BM25ContentFilter(
            user_query=user_query,
            bm25_threshold=1.2,  # Balanced relevance threshold
            language="english"
        )
        
        # Create markdown generator
        md_generator = DefaultMarkdownGenerator(
            content_source="raw_html",
            options=options
        )
        
        # Create a mock result object
        class MockResult:
            def __init__(self, html_content):
                self.success = True
                self.html = html_content
                self.markdown = md_generator.generate_markdown(html_content)
        
        result = MockResult(raw_html)
        
        # Apply BM25 filter for job relevance
        try:
            bm25_chunks = bm25_filter.filter_content(result.html)
            
            if not bm25_chunks:
                logger.warning("No content matched BM25 query, using pruned content")
                # Fallback to pruning filter
                pruning_filter = PruningContentFilter(
                    threshold=0.45,
                    min_word_threshold=10
                )
                bm25_chunks = pruning_filter.filter_content(result.html)
            
            if bm25_chunks:
                filtered_html = "\n---\n".join(bm25_chunks)
                
                # Generate markdown from filtered HTML
                md_generator_filtered = DefaultMarkdownGenerator(
                    content_source="raw_html",
                    options=options
                )
                
                result.html = filtered_html
                result.markdown = md_generator_filtered.generate_markdown(filtered_html)
                
                # Add Fit Markdown metadata
                if hasattr(result, 'markdown'):
                    result.markdown.fit_markdown = filtered_html
            
        except Exception as e:
            logger.warning(f"BM25 filter failed, using original content: {e}")
        
        return result
    
    async def _generate_combined_markdown_from_html(self, raw_html: str, user_query: str, options: Dict[str, Any]) -> Any:
        """Generate combined-filtered markdown from raw HTML content"""
        # Step 1: Enhanced pruning filter
        pruning_filter = PruningContentFilter(
            threshold=0.45,  # Lower threshold for more content retention
            min_word_threshold=10  # Lower minimum for profile content
        )
        
        # Step 2: Enhanced BM25 filter for job relevance
        bm25_filter = BM25ContentFilter(
            user_query=user_query,
            bm25_threshold=1.2,
            language="english"
        )
        
        # Create markdown generator
        md_generator = DefaultMarkdownGenerator(
            content_source="raw_html",
            options=options
        )
        
        # Create a mock result object
        class MockResult:
            def __init__(self, html_content):
                self.success = True
                self.html = html_content
                self.markdown = md_generator.generate_markdown(html_content)
        
        result = MockResult(raw_html)
        
        try:
            # Step 1: Apply pruning filter
            pruned_chunks = pruning_filter.filter_content(result.html)
            if not pruned_chunks:
                logger.warning("Pruning filter returned no content, using original")
                pruned_html = result.html
            else:
                pruned_html = "\n".join(pruned_chunks)
            
            # Step 2: Apply BM25 filter to pruned content
            bm25_chunks = bm25_filter.filter_content(pruned_html)
            
            if not bm25_chunks:
                logger.warning("No content matched BM25 query after pruning, using pruned content")
                filtered_html = pruned_html
            else:
                filtered_html = "\n---\n".join(bm25_chunks)
            
            # Generate markdown from filtered HTML
            md_generator_filtered = DefaultMarkdownGenerator(
                content_source="raw_html",
                options=options
            )
            
            result.html = filtered_html
            result.markdown = md_generator_filtered.generate_markdown(filtered_html)
            
            # Add Fit Markdown metadata
            if hasattr(result, 'markdown'):
                result.markdown.fit_markdown = filtered_html
            
            logger.info(f"✅ Hybrid filtering applied from HTML: {len(pruned_chunks) if pruned_chunks else 0} pruned chunks, {len(bm25_chunks) if bm25_chunks else 0} BM25 chunks")
            
        except Exception as e:
            logger.warning(f"Hybrid filtering failed, using original content: {e}")
        
        return result
    
    def _calculate_content_quality(self, markdown: str) -> float:
        """
        Calculate enhanced content quality score with LinkedIn-specific metrics.
        
        Args:
            markdown: Generated markdown content
            
        Returns:
            Quality score between 0-100
        """
        if not markdown or not markdown.strip():
            return 0.0
        
        score = 0.0
        word_count = len(markdown.split())
        
        # Enhanced length score (25 points max) - LinkedIn profiles should be substantial
        if word_count > 500:
            score += 25
        elif word_count > 200:
            score += 20
        elif word_count > 100:
            score += 15
        elif word_count > 50:
            score += 10
        elif word_count > 20:
            score += 5
        
        # Enhanced structure score (30 points max)
        if '##' in markdown:  # Has headings
            score += 15
        if '###' in markdown:  # Has subheadings
            score += 5
        if '- ' in markdown or '* ' in markdown:  # Has lists
            score += 10
        
        # Content diversity score (25 points max)
        lines = markdown.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) > 20:
            score += 15
        elif len(non_empty_lines) > 10:
            score += 10
        elif len(non_empty_lines) > 5:
            score += 5
        
        # LinkedIn-specific content indicators (20 points max)
        linkedin_indicators = {
            'experience': 4, 'skills': 4, 'education': 3, 'work': 3,
            'project': 2, 'company': 2, 'university': 2, 'degree': 2,
            'linkedin': 1, 'profile': 1, 'professional': 1
        }
        
        found_indicators = 0
        markdown_lower = markdown.lower()
        for indicator, weight in linkedin_indicators.items():
            if indicator in markdown_lower:
                found_indicators += weight
        
        score += min(found_indicators, 20)
        
        # Penalize very short content or obvious failures
        if word_count < 10:
            score *= 0.3  # Heavy penalty for very short content
        elif word_count < 20:
            score *= 0.6  # Moderate penalty for short content
        
        # Check for common failure patterns
        failure_patterns = ['loading', 'error', 'not found', 'access denied']
        if any(pattern in markdown_lower for pattern in failure_patterns):
            score *= 0.2  # Heavy penalty for error content
        
        return min(max(score, 0.0), 100.0)
    
    async def extract_linkedin_profile_markdown(
        self,
        profile_url: str,
        job_requirements: Optional[Any] = None
    ) -> MarkdownGenerationResult:
        """
        Extract LinkedIn profile with enhanced markdown generation and progressive fallback.
        
        IMPORTANT: This method expects an INDIVIDUAL PROFILE URL, not a search results page.
        For search results, the candidates should be extracted first, then each profile
        URL should be passed to this method individually.
        
        Args:
            profile_url: LinkedIn profile URL (must be /in/ or /talent/profile/)
            job_requirements: Job requirements for relevance filtering
            
        Returns:
            MarkdownGenerationResult optimized for candidate analysis
        """
        # Import page detection
        from .linkedin_page_selectors import detect_linkedin_page_type, LinkedInPageType
        
        # Detect page type and warn if not a profile
        page_type = detect_linkedin_page_type(profile_url)
        
        if page_type == LinkedInPageType.SEARCH_RESULTS:
            logger.warning(f"⚠️ URL appears to be a SEARCH RESULTS page, not a profile: {profile_url}")
            logger.warning("⚠️ This will result in poor extraction quality. Extract profile URLs first!")
        
        # Determine filter type based on job requirements
        if job_requirements:
            # Extract key terms for BM25 filtering
            key_terms = []
            # Handle both dict and JobRequirements object
            if hasattr(job_requirements, 'required_skills'):
                # JobRequirements object
                if job_requirements.required_skills:
                    key_terms.extend(job_requirements.required_skills[:5])  # Top 5 skills
                if hasattr(job_requirements, 'title') and job_requirements.title:
                    key_terms.append(job_requirements.title)
                if hasattr(job_requirements, 'industry') and job_requirements.industry:
                    key_terms.append(job_requirements.industry)
            elif isinstance(job_requirements, dict):
                # Dictionary format
                if job_requirements.get('skills'):
                    key_terms.extend(job_requirements['skills'][:5])  # Top 5 skills
                if job_requirements.get('job_titles'):
                    key_terms.extend(job_requirements['job_titles'][:3])  # Top 3 titles
                if job_requirements.get('industries'):
                    key_terms.extend(job_requirements['industries'][:2])  # Top 2 industries
            
            user_query = " ".join(key_terms) if key_terms else None
            
            if user_query:
                filter_type = ContentFilterType.COMBINED
            else:
                filter_type = ContentFilterType.PRUNING
        else:
            filter_type = ContentFilterType.PRUNING
            user_query = None
        
        # Enhanced LinkedIn-specific markdown options
        linkedin_options = {
            "ignore_links": False,  # Keep LinkedIn links
            "ignore_images": True,  # Remove profile images
            "escape_html": True,
            "body_width": 0,
            "skip_internal_links": True,
            "include_sup_sub": True,
            "mark_code": True,
            "handle_code_in_pre": True,
            # Enhanced options for better LinkedIn extraction
            "strip_empty_lines": True,
            "normalize_whitespace": True,
            "preserve_tables": True
        }
        
        return await self.generate_clean_markdown(
            url=profile_url,
            filter_type=filter_type,
            user_query=user_query,
            markdown_options=linkedin_options
        )

    async def extract_linkedin_profile_markdown_from_html(
        self,
        profile_url: str,
        raw_html: str,
        job_requirements: Optional[Any] = None
    ) -> MarkdownGenerationResult:
        """
        Public helper: generate clean markdown directly from raw HTML while preserving
        the same job-aware filter selection and metadata handling as the URL path.
        """
        # Determine filter type based on job requirements
        if job_requirements:
            key_terms = []
            if hasattr(job_requirements, 'required_skills') and job_requirements.required_skills:
                key_terms.extend(job_requirements.required_skills[:5])
            if hasattr(job_requirements, 'title') and getattr(job_requirements, 'title'):
                key_terms.append(job_requirements.title)
            if hasattr(job_requirements, 'industry') and getattr(job_requirements, 'industry'):
                key_terms.append(job_requirements.industry)
            if isinstance(job_requirements, dict):
                if job_requirements.get('skills'):
                    key_terms.extend(job_requirements['skills'][:5])
                if job_requirements.get('job_titles'):
                    key_terms.extend(job_requirements['job_titles'][:3])
                if job_requirements.get('industries'):
                    key_terms.extend(job_requirements['industries'][:2])
            user_query = " ".join(key_terms) if key_terms else None
            filter_type = ContentFilterType.COMBINED if user_query else ContentFilterType.PRUNING
        else:
            filter_type = ContentFilterType.PRUNING
            user_query = None

        # Enhanced LinkedIn-specific markdown options
        linkedin_options = {
            "ignore_links": False,
            "ignore_images": True,
            "escape_html": True,
            "body_width": 0,
            "skip_internal_links": True,
            "include_sup_sub": True,
            "mark_code": True,
            "handle_code_in_pre": True,
            "strip_empty_lines": True,
            "normalize_whitespace": True,
            "preserve_tables": True
        }

        # Use the internal adaptive HTML pipeline
        return await self._generate_markdown_from_html(
            raw_html=raw_html,
            url=profile_url,
            filter_type=filter_type,
            user_query=user_query,
            markdown_options=linkedin_options
        )

# Enhanced utility functions for improved integration

def create_enhanced_crawl4ai_service(api_key: Optional[str] = None, use_authenticated_session: bool = False) -> Crawl4AIMarkdownService:
    """
    Create an enhanced Crawl4AI service with optimized configuration.
    
    Args:
        api_key: API key for LLM services
        use_authenticated_session: Whether to use authenticated LinkedIn session (disabled to avoid new browsers)
        
    Returns:
        Configured Crawl4AIMarkdownService instance
    """
    # Always use default configuration to avoid creating new browsers
    logger.info("🔐 Using default configuration to maintain existing browser session")
    
    return Crawl4AIMarkdownService(
        api_key=api_key,
        browser_config=None  # Always None to avoid new browser creation
    )

# Utility functions for easy integration
async def get_clean_markdown_for_llm(
    url: str,
    job_requirements: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to get clean markdown for LLM analysis.
    
    Args:
        url: URL to extract markdown from
        job_requirements: Job requirements for relevance filtering
        api_key: API key for LLM services
        
    Returns:
        Clean markdown content optimized for LLM analysis
    """
    async with Crawl4AIMarkdownService(api_key=api_key) as service:
        result = await service.extract_linkedin_profile_markdown(url, job_requirements)
        return result.filtered_markdown

async def batch_extract_markdown(
    urls: List[str],
    job_requirements: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    max_concurrent: int = 3
) -> List[MarkdownGenerationResult]:
    """
    Extract markdown from multiple URLs concurrently.
    
    Args:
        urls: List of URLs to extract markdown from
        job_requirements: Job requirements for relevance filtering
        api_key: API key for LLM services
        max_concurrent: Maximum concurrent extractions
        
    Returns:
        List of MarkdownGenerationResult objects
    """
    async with Crawl4AIMarkdownService(api_key=api_key) as service:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_single(url: str) -> MarkdownGenerationResult:
            async with semaphore:
                return await service.extract_linkedin_profile_markdown(url, job_requirements)
        
        tasks = [extract_single(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
