"""
Human Behavior Simulation for LinkedIn Scraping
================================================

This module provides utilities to make browser automation appear more human-like,
reducing the risk of detection by LinkedIn's anti-bot systems.

Based on best practices from: https://www.reddit.com/r/SaaS/comments/1mibl3u/how_to_build_a_linkedin_scraper_that_actually/

Key Principles:
1. Random delays instead of fixed timing
2. Mouse movement before clicks
3. Variable scroll patterns
4. Human-like typing speed
"""

import random
import time
import logging
from typing import Optional
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


def human_delay(min_sec: float = 0.5, max_sec: float = 2.0):
    """
    Sleep for a random duration to mimic human behavior.
    
    Real humans don't have consistent timing - they pause, think, get distracted.
    This function adds realistic variability to automation timing.
    
    Args:
        min_sec: Minimum delay in seconds
        max_sec: Maximum delay in seconds
    
    Example:
        human_delay(0.3, 0.8)  # Random delay between 300-800ms
        human_delay(1.0, 2.5)  # Random delay between 1-2.5 seconds
    """
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


def human_click(driver: WebDriver, element: WebElement, use_js_fallback: bool = True):
    """
    Click element with human-like mouse movement.
    
    Real users move their mouse to an element before clicking. This function
    simulates that behavior with slight randomness in positioning.
    
    Args:
        driver: Selenium WebDriver instance
        element: Element to click
        use_js_fallback: If True, falls back to JavaScript click if ActionChains fails
    
    Example:
        button = driver.find_element(By.ID, "submit")
        human_click(driver, button)
    """
    try:
        actions = ActionChains(driver)
        
        # Move to element with slight random offset (mimics imperfect mouse control)
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        actions.move_to_element_with_offset(element, offset_x, offset_y)
        
        # Random pause before click (humans don't click instantly)
        pause_duration = random.uniform(0.1, 0.3)
        actions.pause(pause_duration)
        
        # Perform the click
        actions.click()
        actions.perform()
        
        logger.debug(f"Human click performed with offset ({offset_x}, {offset_y}) and {pause_duration:.2f}s pause")
        
        # Small delay after click (processing time)
        human_delay(0.1, 0.3)
        
    except Exception as e:
        logger.warning(f"ActionChains click failed: {e}")
        
        if use_js_fallback:
            logger.info("Falling back to JavaScript click")
            try:
                driver.execute_script("arguments[0].click();", element)
                human_delay(0.1, 0.3)
            except Exception as js_error:
                logger.error(f"JavaScript click also failed: {js_error}")
                raise
        else:
            raise


def human_scroll(driver: WebDriver, num_scrolls: Optional[int] = None, direction: str = 'down'):
    """
    Scroll page with human-like patterns.
    
    Real users don't scroll uniformly - they scroll varying distances at varying speeds,
    sometimes scroll back up to re-read something, and pause at different intervals.
    
    Args:
        driver: Selenium WebDriver instance
        num_scrolls: Number of scroll actions (random if None)
        direction: 'down', 'up', or 'mixed' for varied scrolling
    
    Example:
        human_scroll(driver)  # Random natural scrolling
        human_scroll(driver, num_scrolls=3, direction='down')  # 3 downward scrolls
    """
    if num_scrolls is None:
        num_scrolls = random.randint(2, 5)
    
    for i in range(num_scrolls):
        # Random scroll distance (not always to bottom)
        if direction == 'down':
            scroll_distance = random.randint(300, 800)
        elif direction == 'up':
            scroll_distance = -random.randint(300, 800)
        else:  # mixed
            scroll_distance = random.randint(300, 800) * random.choice([1, -1])
        
        driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
        logger.debug(f"Scrolled {scroll_distance}px")
        
        # Variable pause between scrolls (humans read as they scroll)
        human_delay(0.3, 1.2)
        
        # Sometimes scroll back up a bit (30% chance - humans do this to re-read)
        if random.random() < 0.3 and direction != 'up':
            back_distance = random.randint(50, 200)
            driver.execute_script(f"window.scrollBy(0, -{back_distance});")
            logger.debug(f"Scrolled back up {back_distance}px (re-reading)")
            human_delay(0.2, 0.5)


def human_scroll_to_element(driver: WebDriver, element: WebElement, behavior: str = 'smooth'):
    """
    Scroll to element with human-like behavior.
    
    Args:
        driver: Selenium WebDriver instance
        element: Element to scroll to
        behavior: 'smooth' or 'auto' scroll behavior
    
    Example:
        target = driver.find_element(By.ID, "target")
        human_scroll_to_element(driver, target)
    """
    try:
        # Scroll with smooth behavior (more human-like)
        driver.execute_script(
            f"arguments[0].scrollIntoView({{block: 'center', behavior: '{behavior}'}});", 
            element
        )
        
        # Wait for scroll animation to complete
        human_delay(0.3, 0.6)
        
        logger.debug(f"Scrolled to element with {behavior} behavior")
        
    except Exception as e:
        logger.warning(f"Failed to scroll to element: {e}")


def human_type(element: WebElement, text: str, typing_speed: str = 'normal'):
    """
    Type text with human-like speed variations.
    
    Real users don't type at constant speed - they vary based on the character,
    occasionally make mistakes, and pause to think.
    
    Args:
        element: Input element to type into
        text: Text to type
        typing_speed: 'fast' (50-100ms), 'normal' (50-150ms), 'slow' (100-250ms)
    
    Example:
        search_box = driver.find_element(By.ID, "search")
        human_type(search_box, "Software Engineer", typing_speed='normal')
    """
    speed_ranges = {
        'fast': (0.05, 0.10),
        'normal': (0.05, 0.15),
        'slow': (0.10, 0.25)
    }
    
    min_delay, max_delay = speed_ranges.get(typing_speed, speed_ranges['normal'])
    
    for i, char in enumerate(text):
        element.send_keys(char)
        
        # Variable typing speed per character
        char_delay = random.uniform(min_delay, max_delay)
        
        # Longer pause after spaces (thinking/reading)
        if char == ' ':
            char_delay *= random.uniform(1.5, 2.5)
        
        # Occasional longer pause (thinking, 10% chance)
        if random.random() < 0.1:
            char_delay *= random.uniform(2.0, 4.0)
        
        time.sleep(char_delay)
    
    logger.debug(f"Typed '{text}' with {typing_speed} speed")
    
    # Small pause after typing (processing/reviewing)
    human_delay(0.2, 0.5)


def human_page_load_wait(driver: WebDriver, min_wait: float = 1.0, max_wait: float = 3.0):
    """
    Wait for page load with human-like timing.
    
    Humans take time to visually process a new page before interacting.
    This adds realistic delay after page loads.
    
    Args:
        driver: Selenium WebDriver instance
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
    """
    # Wait for page to be ready
    try:
        driver.execute_script("return document.readyState") == "complete"
    except:
        pass
    
    # Human processing time
    human_delay(min_wait, max_wait)
    logger.debug("Page load wait completed")


def random_mouse_movement(driver: WebDriver, num_movements: int = 3):
    """
    Perform random mouse movements to simulate human browsing.
    
    Real users move their mouse around while reading/thinking.
    This can help avoid detection during wait periods.
    
    Args:
        driver: Selenium WebDriver instance
        num_movements: Number of random movements to perform
    """
    try:
        actions = ActionChains(driver)
        
        for _ in range(num_movements):
            # Random coordinates within viewport
            x = random.randint(100, 1000)
            y = random.randint(100, 700)
            
            # Move to random position
            actions.move_by_offset(x, y)
            actions.pause(random.uniform(0.1, 0.3))
            
            # Reset offset for next movement
            actions.move_by_offset(-x, -y)
        
        actions.perform()
        logger.debug(f"Performed {num_movements} random mouse movements")
        
    except Exception as e:
        logger.debug(f"Random mouse movement failed (non-critical): {e}")


def simulate_reading_pause(min_sec: float = 2.0, max_sec: float = 5.0):
    """
    Simulate time spent reading content.
    
    Humans spend time reading before taking action. This adds realistic
    pauses that suggest content consumption.
    
    Args:
        min_sec: Minimum reading time in seconds
        max_sec: Maximum reading time in seconds
    """
    reading_time = random.uniform(min_sec, max_sec)
    logger.debug(f"Simulating reading for {reading_time:.2f}s")
    time.sleep(reading_time)


# Convenience function for common use case
def wait_and_click(driver: WebDriver, element: WebElement, 
                   pre_click_delay: tuple = (0.3, 0.8),
                   use_human_click: bool = True):
    """
    Wait briefly then click element (common pattern).
    
    Args:
        driver: Selenium WebDriver instance
        element: Element to click
        pre_click_delay: (min, max) delay before clicking
        use_human_click: Use human_click() vs regular click()
    """
    human_delay(*pre_click_delay)
    
    if use_human_click:
        human_click(driver, element)
    else:
        element.click()
        human_delay(0.1, 0.3)


# Export all functions
__all__ = [
    'human_delay',
    'human_click',
    'human_scroll',
    'human_scroll_to_element',
    'human_type',
    'human_page_load_wait',
    'random_mouse_movement',
    'simulate_reading_pause',
    'wait_and_click'
]
