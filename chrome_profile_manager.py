"""
Chrome Profile Manager for persistent LinkedIn Recruiter sessions.
This module provides two approaches for maintaining browser sessions:
1. ChromeProfileManager - Uses your actual Chrome profile with session persistence
2. ChromeRemoteDebugger - Alternative approach using Chrome Remote Debugging

Key Features:
- Uses your actual Chrome profile with all saved cookies and sessions
- Maintains session between runs - browser stays open after script ends
- Manual login flow preserved - you log in once, stay logged in
- Handles contract selection - waits for you to manually select the right contract
- Session persistence - tries to reuse existing browser sessions
"""

import os
import sys
import json
import time
import psutil
import subprocess
import pickle
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, NoSuchWindowException
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ChromeProfileManager:
    """Manages Chrome browser sessions using existing user profile"""
    
    def __init__(self):
        self.driver = None
        self.session_file = Path("chrome_session.pkl")
        self.chrome_user_data_dir = self._get_chrome_user_data_dir()
        
    def _get_chrome_user_data_dir(self):
        """Get the Chrome user data directory based on the operating system"""
        home = Path.home()
        
        if sys.platform == "darwin":  # macOS
            base_dir = home / "Library" / "Application Support" / "Google" / "Chrome"
        elif sys.platform == "win32":  # Windows
            base_dir = home / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
        else:  # Linux
            base_dir = home / ".config" / "google-chrome"
        
        # Use a separate profile directory for automation to avoid conflicts
        automation_dir = base_dir.parent / "Chrome_Automation"
        automation_dir.mkdir(exist_ok=True)
        
        return automation_dir
    
    def _is_chrome_running_with_profile(self):
        """Check if Chrome is already running with the automation profile"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    # Check if it's running with our specific automation profile
                    if any(str(self.chrome_user_data_dir) in str(arg) for arg in cmdline):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def setup_chrome_options(self, headless=False, use_existing_session=True):
        """Configure Chrome options to use existing profile"""
        options = Options()
        
        # Use separate automation profile to avoid conflicts with main browser
        options.add_argument(f"user-data-dir={self.chrome_user_data_dir}")
        
        # Use a specific automation profile directory
        options.add_argument("profile-directory=AutomationProfile")
        
        # Disable automation flags to appear more like regular browsing
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Additional options to avoid detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Keep browser open after script ends (for session persistence)
        if use_existing_session:
            options.add_experimental_option("detach", True)
        
        # Window size
        options.add_argument("--window-size=1920,1080")
        
        # Realistic user agent
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
        options.add_argument("--accept-language=en-US,en;q=0.9")
        
        # Additional stealth options
        options.add_argument("--disable-extensions")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        
        # Suppress logging
        options.add_argument("--log-level=3")
        options.add_argument("--silent")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        if headless:
            options.add_argument("--headless")
        
        return options
    
    def connect_to_existing_or_create_session(self):
        """Try to connect to existing Chrome session or create new one"""
        
        # First, check if there's an existing Chrome session we can connect to
        if self._is_chrome_running_with_profile():
            logger.warning("⚠️  Chrome is already running with your profile.")
            logger.info("Please close all Chrome windows and try again, or use remote debugging.")
            return None
        
        try:
            # Try to restore previous session
            if self.session_file.exists():
                logger.info("📌 Attempting to restore previous session...")
                with open(self.session_file, 'rb') as f:
                    session_data = pickle.load(f)
                    
                # Check if the previous driver session is still valid
                try:
                    # This will fail if the session is dead
                    self.driver = webdriver.Chrome(
                        service=Service(),
                        options=self.setup_chrome_options()
                    )
                    return self.driver
                except:
                    logger.info("Previous session expired, creating new one...")
        except:
            pass
        
        # Create new session
        logger.info("🚀 Starting new Chrome session with your profile...")
        options = self.setup_chrome_options()
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(),
                options=options
            )
            
            # Save session for potential reuse
            with open(self.session_file, 'wb') as f:
                pickle.dump({'session_id': self.driver.session_id}, f)
            
            logger.info("✅ Chrome session started successfully")
            return self.driver
            
        except WebDriverException as e:
            logger.error(f"❌ Error starting Chrome: {e}")
            return None
    
    def navigate_to_linkedin_recruiter(self):
        """Navigate to LinkedIn Recruiter with optimized flow"""
        if not self.driver:
            logger.error("No driver session available")
            return False
        
        try:
            current_url = self.driver.current_url
            
            # Check if already on LinkedIn Recruiter
            if "talent" in current_url and "search" in current_url:
                logger.info("✅ Already on LinkedIn Recruiter search page")
                return True
            
            # Check if already on LinkedIn Recruiter home
            if "talent" in current_url:
                logger.info("✅ Already on LinkedIn Recruiter, navigating to search...")
                self.driver.get("https://www.linkedin.com/talent/search")
                time.sleep(0.3)  # PERFORMANCE: Reduced from 3s - page loads async
                return True
            
            # Check if already logged into LinkedIn
            if "feed" in current_url or "home" in current_url:
                logger.info("✅ Already logged into LinkedIn, navigating to Recruiter...")
                self.driver.get("https://www.linkedin.com/talent/search")
                # PERFORMANCE FIX: Minimal wait for page to start loading
                time.sleep(0.2)

                # Check if contract selection is needed
                if "contract-chooser" in self.driver.current_url or "login-cap" in self.driver.current_url:
                    logger.info("📋 Contract selection required - please select manually...")
                    return self._wait_for_contract_selection()

                return True

            # Need to login first
            logger.info("📍 Navigating to LinkedIn...")
            self.driver.get("https://www.linkedin.com")

            # PERFORMANCE FIX: Minimal wait for page to start loading
            time.sleep(0.2)

            # Check if we need to login
            if "login" in self.driver.current_url:
                logger.info("🔐 Please log in to LinkedIn manually...")
                logger.info("   Waiting for login completion...")

                # PERFORMANCE FIX: Reduced login wait timeout from 300s (5 min) to 120s (2 min)
                # Most users complete login within 30-60 seconds
                WebDriverWait(self.driver, 120).until(
                    lambda driver: "feed" in driver.current_url or "home" in driver.current_url
                )
                logger.info("✅ Login successful!")

                # Attempt to auto-navigate to Recruiter after regular login.
                # Many users log into linkedin.com; try to jump to recruiter and auto-select a contract.
                try:
                    logger.info("🔄 Attempting automatic navigation to LinkedIn Recruiter after regular login...")
                    self.driver.get("https://www.linkedin.com/talent/search")
                    time.sleep(0.3)  # PERFORMANCE: Reduced from 2s

                    # If contract chooser appears, try to auto-select first available contract
                    if "contract-chooser" in self.driver.current_url or "login-cap" in self.driver.current_url:
                        logger.info("📍 Contract chooser detected — attempting automatic contract selection...")
                        try:
                            # Look for common contract chooser selectors and click the first enabled option
                            contract_selectors = [
                                "button[data-test-contract-option]",
                                "button[data-control-name='select_contract']",
                                "button[role='button']",
                                "div.contract-option button"
                            ]
                            selected = False
                            for sel in contract_selectors:
                                try:
                                    elems = self.driver.find_elements(By.CSS_SELECTOR, sel)
                                    if elems:
                                        for e in elems:
                                            try:
                                                if e.is_displayed() and e.is_enabled():
                                                    e.click()
                                                    logger.info("✅ Clicked contract element using selector: %s", sel)
                                                    selected = True
                                                    break
                                            except Exception:
                                                continue
                                    if selected:
                                        break
                                except Exception:
                                    continue

                            # Also try selecting links if buttons are not present
                            if not selected:
                                link_elems = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='contract']")
                                for le in link_elems:
                                    try:
                                        if le.is_displayed():
                                            le.click()
                                            logger.info("✅ Clicked contract link to select contract")
                                            selected = True
                                            break
                                    except Exception:
                                        continue

                            if selected:
                                # PERFORMANCE FIX: Reduced wait time for recruiter to load
                                time.sleep(1)
                                logger.info("✅ Automatic contract selection succeeded — now on Recruiter")
                                return True
                            else:
                                logger.info("⚠️ Automatic contract selection not available; falling back to manual selection prompt")
                        except Exception as e:
                            logger.debug("Automatic contract selection attempt failed: %s", e)
                            # fall through to manual selection wait
                except Exception as e:
                    logger.debug("Automatic navigation attempt failed: %s", e)
                    # fall through to manual navigation
            
            # Now navigate to LinkedIn Recruiter
            logger.info("📍 Navigating to LinkedIn Recruiter...")
            self.driver.get("https://www.linkedin.com/talent/search")

            # PERFORMANCE FIX: Reduced wait time after navigation
            time.sleep(1)

            # Check if we need to select a contract
            if "contract-chooser" in self.driver.current_url or "login-cap" in self.driver.current_url:
                logger.info("📋 Contract selection required - please select manually...")
                return self._wait_for_contract_selection()
            
            logger.info("✅ Successfully loaded LinkedIn Recruiter")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navigating to LinkedIn Recruiter: {e}")
            return False
    
    def _wait_for_contract_selection(self, timeout_seconds: int = 300):
        """Wait for manual contract selection with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                current_url = self.driver.current_url
                if "talent" in current_url and "login" not in current_url and "contract-chooser" not in current_url:
                    logger.info("✅ Contract selected successfully!")
                    return True
                    
                remaining = timeout_seconds - (time.time() - start_time)
                if remaining > 0:
                    logger.info("⏰ Still waiting for contract selection... %ds remaining", int(remaining))
                    
                time.sleep(30)
                
            except Exception as e:
                logger.debug("Error checking contract selection: %s", e)
                time.sleep(5)
        
        logger.error("Contract selection timeout after %d seconds", timeout_seconds)
        return False
    
    def keep_session_alive(self):
        """Keep the browser session alive between script runs"""
        if self.driver:
            logger.info("\n🔄 Keeping browser session alive...")
            logger.info("   The browser will remain open for your next run.")
            logger.info("   To close it manually, use the close_session() method.")
            # Don't quit the driver - keep it running
            # self.driver = None  # COMMENTED OUT: This was breaking chat searches!
    
    def close_session(self):
        """Explicitly close the browser session"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🛑 Browser session closed")
            except:
                pass
        
        # Clean up session file
        if self.session_file.exists():
            self.session_file.unlink()


class ChromeRemoteDebugger:
    """Alternative approach using Chrome Remote Debugging for even better persistence"""
    
    def __init__(self, debugging_port=9222):
        self.debugging_port = debugging_port
        self.driver = None
    
    def start_chrome_with_debugging(self):
        """Start Chrome with remote debugging enabled"""
        chrome_path = self._get_chrome_executable()
        user_data_dir = ChromeProfileManager()._get_chrome_user_data_dir()
        
        # Start Chrome with debugging port
        cmd = [
            chrome_path,
            f'--remote-debugging-port={self.debugging_port}',
            f'--user-data-dir={user_data_dir}',
            '--profile-directory=AutomationProfile'
        ]
        
        logger.info(f"Starting Chrome with debugging on port {self.debugging_port}...")
        subprocess.Popen(cmd)
        time.sleep(3)  # Give Chrome time to start
        
        logger.info(f"Chrome started. You can now connect to it on port {self.debugging_port}")
    
    def _get_chrome_executable(self):
        """Get Chrome executable path based on OS"""
        if sys.platform == "darwin":  # macOS
            return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif sys.platform == "win32":  # Windows
            return "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        else:  # Linux
            return "/usr/bin/google-chrome"
    
    def connect_to_existing_chrome(self):
        """Connect to the Chrome instance with debugging enabled"""
        options = Options()
        options.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.debugging_port}")
        
        try:
            self.driver = webdriver.Chrome(options=options)
            logger.info(f"✅ Connected to Chrome on port {self.debugging_port}")
            return self.driver
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            logger.info(f"   Make sure Chrome is running with --remote-debugging-port={self.debugging_port}")
            return None


class LinkedInRecruiterApp:
    """Main application that uses the Chrome Profile Manager"""
    
    def __init__(self):
        self.browser_manager = ChromeProfileManager()
        self.driver = None
    
    def initialize(self):
        """Initialize the browser and navigate to LinkedIn Recruiter"""
        self.driver = self.browser_manager.connect_to_existing_or_create_session()
        
        if not self.driver:
            logger.error("Failed to initialize browser")
            return False
        
        # Navigate to LinkedIn Recruiter with manual login/contract selection
        return self.browser_manager.navigate_to_linkedin_recruiter()
    
    def run_your_automation(self):
        """Your main automation logic goes here"""
        if not self.driver:
            logger.error("Driver not initialized")
            return
        
        logger.info("\n🤖 Running your automation tasks...")
        
        # Add your LinkedIn Recruiter automation code here
        # Example:
        try:
            # Wait for page to be ready
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Your automation tasks...
            logger.info("   Performing automated tasks...")
            time.sleep(2)  # Placeholder for your actual automation
            
            logger.info("✅ Automation completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during automation: {e}")
    
    def finish(self, keep_alive=True):
        """Finish the session"""
        if keep_alive:
            self.browser_manager.keep_session_alive()
        else:
            self.browser_manager.close_session()


# Example usage
if __name__ == "__main__":
    # Method 1: Using Chrome Profile Manager (Recommended)
    logger.info("=== LinkedIn Recruiter Automation ===\n")
    
    app = LinkedInRecruiterApp()
    
    # Initialize and navigate to LinkedIn Recruiter
    if app.initialize():
        # Run your automation
        app.run_your_automation()
        
        # Keep session alive for next run
        app.finish(keep_alive=True)
    else:
        logger.error("Failed to initialize the application")
    
    
    # Method 2: Using Remote Debugging (Alternative for better persistence)
    # Uncomment below to use this method instead
    """
    debugger = ChromeRemoteDebugger()
    
    # First time: Start Chrome with debugging
    # debugger.start_chrome_with_debugging()
    
    # Connect to existing Chrome
    driver = debugger.connect_to_existing_chrome()
    if driver:
        # Use the driver for your automation
        driver.get("https://www.linkedin.com/talent/home")
        # ... your automation code ...
    """
