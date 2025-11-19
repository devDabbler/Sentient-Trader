"""
OCR utilities for extracting Option Alpha bot configuration from screenshots.

This module provides functionality to extract bot configuration data from
screenshots of Option Alpha bot interfaces.
"""

from loguru import logger
from typing import Dict, Tuple, Optional
import io


def extract_bot_config_from_screenshot(image_bytes: bytes) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extract bot configuration from a screenshot image.
    
    Args:
        image_bytes: Raw image bytes from uploaded screenshot
        
    Returns:
        Tuple of (config_dict, error_message)
        - config_dict: Extracted bot configuration if successful, None otherwise
        - error_message: Error message if extraction failed, None otherwise
    """
    try:
        # For now, use the LLM-based extraction from llm_strategy_analyzer
        # This is a placeholder - full OCR implementation would use:
        # - pytesseract or easyocr for text extraction
        # - OpenCV for image preprocessing
        # - LLM for structured data extraction
        
        from services.llm_strategy_analyzer import extract_bot_config_from_screenshot as llm_extract
        
        # Call the LLM-based extraction (currently returns a dict, not a tuple)
        config = llm_extract()
        
        if config:
            return config, None
        else:
            return None, "Failed to extract bot configuration from screenshot"
            
    except ImportError:
        logger.debug("LLM strategy analyzer not available for OCR extraction")
        return None, "OCR extraction requires LLM strategy analyzer module"
    except Exception as e:
        logger.error(f"Error extracting bot config from screenshot: {e}")
        return None, f"OCR extraction failed: {str(e)}"


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract raw text from an image using OCR.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Extracted text string
    """
    try:
        # Try using pytesseract if available
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            logger.debug("pytesseract not available, OCR text extraction disabled")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

