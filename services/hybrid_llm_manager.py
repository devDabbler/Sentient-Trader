"""
Hybrid LLM Manager - Intelligent switching between local Ollama and cloud APIs
Optimizes for speed (cloud) vs cost (local) based on use case
"""

from loguru import logger
from typing import Optional, Literal
import os
from services.llm_strategy_analyzer import LLMStrategyAnalyzer


class HybridLLMManager:
    """
    Manages LLM selection strategy:
    - Use LOCAL Ollama for: single queries, interactive analysis, development
    - Use CLOUD API for: bulk operations, time-sensitive tasks, production
    """
    
    def __init__(self):
        """Initialize both local and cloud LLM analyzers"""
        self.local_analyzer = None
        self.cloud_analyzer = None
        self.mode = os.getenv('LLM_MODE', 'hybrid')  # 'hybrid', 'local_only', 'cloud_only'
        
        # Initialize analyzers
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize both local and cloud analyzers"""
        try:
            # Try to initialize local Ollama
            local_model = os.getenv('AI_ANALYZER_MODEL', 'ollama/qwen2.5:7b')
            if local_model.startswith('ollama/') or local_model.startswith('ollama:'):
                try:
                    self.local_analyzer = LLMStrategyAnalyzer(
                        provider='ollama',
                        model=local_model.replace('ollama/', '').replace('ollama:', '')
                    )
                    logger.success("✅ Local Ollama analyzer initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize Ollama: {e}")
                    self.local_analyzer = None
        except Exception as e:
            logger.warning(f"⚠️ Local analyzer initialization failed: {e}")
            self.local_analyzer = None
        
        try:
            # Initialize cloud fallback with fast model
            cloud_model = os.getenv('CLOUD_TRADING_MODEL', 'google/gemini-2.0-flash-exp:free')
            api_key = os.getenv('OPENROUTER_API_KEY')
            
            if api_key:
                self.cloud_analyzer = LLMStrategyAnalyzer(
                    provider='openrouter',
                    model=cloud_model,
                    api_key=api_key
                )
                logger.success(f"✅ Cloud analyzer initialized with {cloud_model}")
            else:
                logger.warning("⚠️ OPENROUTER_API_KEY not found, cloud fallback disabled")
                self.cloud_analyzer = None
        except Exception as e:
            logger.warning(f"⚠️ Cloud analyzer initialization failed: {e}")
            self.cloud_analyzer = None
    
    def get_analyzer(
        self,
        use_case: Literal['interactive', 'bulk', 'auto'] = 'interactive',
        force_cloud: bool = False
    ) -> Optional[LLMStrategyAnalyzer]:
        """
        Get appropriate LLM analyzer based on use case
        
        Args:
            use_case: 
                - 'interactive': Single analysis, user waiting (prefer local if fast enough)
                - 'bulk': Multiple analyses, batch processing (prefer cloud for speed)
                - 'auto': Automated background task (prefer local to save costs)
            force_cloud: Force cloud API regardless of use case
            
        Returns:
            LLMStrategyAnalyzer instance or None
        """
        
        # Force cloud mode
        if force_cloud or self.mode == 'cloud_only':
            if self.cloud_analyzer:
                logger.info("☁️ Using cloud API (forced or cloud-only mode)")
                return self.cloud_analyzer
            else:
                logger.warning("⚠️ Cloud forced but not available, trying local...")
                return self.local_analyzer
        
        # Force local mode
        if self.mode == 'local_only':
            if self.local_analyzer:
                logger.info("🏠 Using local Ollama (local-only mode)")
                return self.local_analyzer
            else:
                logger.warning("⚠️ Local forced but not available, trying cloud...")
                return self.cloud_analyzer
        
        # Hybrid mode - intelligent selection
        if use_case == 'bulk':
            # Bulk operations: prefer cloud for speed
            if self.cloud_analyzer:
                logger.info("☁️ Using cloud API for bulk operation (speed priority)")
                return self.cloud_analyzer
            elif self.local_analyzer:
                logger.warning("⚠️ Cloud not available for bulk, using local (may be slow)")
                return self.local_analyzer
            
        elif use_case == 'interactive':
            # Interactive: prefer local if available, cloud as fallback
            if self.local_analyzer:
                logger.info("🏠 Using local Ollama for interactive analysis")
                return self.local_analyzer
            elif self.cloud_analyzer:
                logger.info("☁️ Local not available, using cloud for interactive")
                return self.cloud_analyzer
                
        elif use_case == 'auto':
            # Auto/background: prefer local to save API costs
            if self.local_analyzer:
                logger.info("🏠 Using local Ollama for background task")
                return self.local_analyzer
            elif self.cloud_analyzer:
                logger.info("☁️ Local not available, using cloud for background task")
                return self.cloud_analyzer
        
        logger.error("❌ No LLM analyzer available!")
        return None
    
    def analyze_with_llm(
        self,
        prompt: str,
        use_case: Literal['interactive', 'bulk', 'auto'] = 'interactive',
        force_cloud: bool = False
    ) -> str:
        """
        Convenience method to analyze with appropriate LLM
        
        Args:
            prompt: The prompt to analyze
            use_case: Type of analysis operation
            force_cloud: Force cloud API
            
        Returns:
            LLM response text or empty string on failure
        """
        analyzer = self.get_analyzer(use_case=use_case, force_cloud=force_cloud)
        if analyzer:
            try:
                return analyzer.analyze_with_llm(prompt)
            except Exception as e:
                logger.error(f"❌ LLM analysis failed: {e}")
                return ""
        else:
            logger.error("❌ No LLM analyzer available")
            return ""
    
    def get_status(self) -> dict:
        """Get status of available analyzers"""
        return {
            'mode': self.mode,
            'local_available': self.local_analyzer is not None,
            'cloud_available': self.cloud_analyzer is not None,
            'local_model': self.local_analyzer.model if self.local_analyzer else None,
            'cloud_model': self.cloud_analyzer.model if self.cloud_analyzer else None
        }


# Global instance for easy access
_hybrid_manager = None

def get_hybrid_llm_manager() -> HybridLLMManager:
    """Get or create global hybrid LLM manager"""
    global _hybrid_manager
    if _hybrid_manager is None:
        _hybrid_manager = HybridLLMManager()
    return _hybrid_manager


def get_llm_for_bulk_operations() -> Optional[LLMStrategyAnalyzer]:
    """Convenience function to get LLM optimized for bulk operations"""
    manager = get_hybrid_llm_manager()
    return manager.get_analyzer(use_case='bulk', force_cloud=True)


def get_llm_for_interactive() -> Optional[LLMStrategyAnalyzer]:
    """Convenience function to get LLM for interactive single queries"""
    manager = get_hybrid_llm_manager()
    return manager.get_analyzer(use_case='interactive')


def get_llm_for_background() -> Optional[LLMStrategyAnalyzer]:
    """Convenience function to get LLM for background/automated tasks"""
    manager = get_hybrid_llm_manager()
    return manager.get_analyzer(use_case='auto')
