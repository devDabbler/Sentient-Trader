"""
Process Lock Utility - Ensures only one instance of a service runs at a time

Usage:
    from utils.process_lock import ProcessLock
    
    # At the start of your service:
    lock = ProcessLock("discord_approval_bot")
    if not lock.acquire():
        print("Another instance is already running!")
        sys.exit(1)
    
    # Your service code here...
    
    # On shutdown (optional - handled automatically on clean exit):
    lock.release()
"""

import os
import sys
import atexit
import signal
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ProcessLock:
    """
    Cross-platform process lock using PID files.
    Prevents multiple instances of the same service from running.
    """
    
    def __init__(self, service_name: str, lock_dir: Optional[str] = None):
        """
        Initialize process lock.
        
        Args:
            service_name: Unique name for this service (e.g., 'discord_approval_bot')
            lock_dir: Directory for lock files (defaults to project_root/locks/)
        """
        self.service_name = service_name
        self._locked = False
        
        # Determine lock directory
        if lock_dir:
            self.lock_dir = Path(lock_dir)
        else:
            # Default to project_root/locks/
            project_root = Path(__file__).parent.parent
            self.lock_dir = project_root / "locks"
        
        # Ensure lock directory exists
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # PID file path
        self.pid_file = self.lock_dir / f"{service_name}.pid"
        
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            if sys.platform == 'win32':
                # Windows: Use tasklist
                import subprocess
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                    capture_output=True,
                    text=True
                )
                return str(pid) in result.stdout
            else:
                # Unix: Send signal 0 (doesn't kill, just checks)
                os.kill(pid, 0)
                return True
        except (OSError, ProcessLookupError, PermissionError):
            return False
        except Exception as e:
            logger.debug(f"Error checking process {pid}: {e}")
            return False
    
    def _read_pid_file(self) -> Optional[int]:
        """Read PID from lock file."""
        try:
            if self.pid_file.exists():
                content = self.pid_file.read_text().strip()
                if content:
                    # Format: PID|timestamp|service_name
                    parts = content.split('|')
                    return int(parts[0])
        except (ValueError, IOError) as e:
            logger.debug(f"Error reading PID file: {e}")
        return None
    
    def _write_pid_file(self):
        """Write current PID to lock file."""
        pid = os.getpid()
        timestamp = datetime.now().isoformat()
        content = f"{pid}|{timestamp}|{self.service_name}"
        self.pid_file.write_text(content)
        logger.info(f"🔒 Process lock acquired: {self.service_name} (PID: {pid})")
    
    def _remove_pid_file(self):
        """Remove PID file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                logger.info(f"🔓 Process lock released: {self.service_name}")
        except Exception as e:
            logger.debug(f"Error removing PID file: {e}")
    
    def acquire(self, force: bool = False) -> bool:
        """
        Attempt to acquire the process lock.
        
        Args:
            force: If True, kill existing process and take over lock
            
        Returns:
            True if lock acquired, False if another instance is running
        """
        existing_pid = self._read_pid_file()
        
        if existing_pid:
            if self._is_process_running(existing_pid):
                if force:
                    logger.warning(f"⚠️ Force-killing existing {self.service_name} (PID: {existing_pid})")
                    try:
                        if sys.platform == 'win32':
                            os.system(f'taskkill /F /PID {existing_pid}')
                        else:
                            os.kill(existing_pid, signal.SIGTERM)
                        import time
                        time.sleep(1)  # Give it time to die
                    except Exception as e:
                        logger.error(f"Failed to kill existing process: {e}")
                        return False
                else:
                    logger.error(
                        f"❌ Another instance of {self.service_name} is already running (PID: {existing_pid})\n"
                        f"   If this is incorrect, delete: {self.pid_file}"
                    )
                    return False
            else:
                # Stale PID file - process is dead
                logger.info(f"🧹 Cleaning up stale lock file for {self.service_name} (old PID: {existing_pid})")
                self._remove_pid_file()
        
        # Write our PID
        self._write_pid_file()
        self._locked = True
        
        # Register cleanup handlers
        atexit.register(self.release)
        
        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, releasing lock...")
            self.release()
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)
        except Exception as e:
            logger.debug(f"Could not register signal handlers: {e}")
        
        return True
    
    def release(self):
        """Release the process lock."""
        if self._locked:
            self._remove_pid_file()
            self._locked = False
    
    def is_locked(self) -> bool:
        """Check if we hold the lock."""
        return self._locked
    
    @classmethod
    def is_service_running(cls, service_name: str, lock_dir: Optional[str] = None) -> bool:
        """
        Static method to check if a service is running.
        
        Args:
            service_name: Name of the service to check
            lock_dir: Directory for lock files
            
        Returns:
            True if service is running, False otherwise
        """
        lock = cls(service_name, lock_dir)
        existing_pid = lock._read_pid_file()
        if existing_pid:
            return lock._is_process_running(existing_pid)
        return False
    
    @classmethod
    def get_running_services(cls, lock_dir: Optional[str] = None) -> dict:
        """
        Get all running services with their PIDs.
        
        Returns:
            Dict mapping service_name -> PID
        """
        if lock_dir:
            locks_path = Path(lock_dir)
        else:
            project_root = Path(__file__).parent.parent
            locks_path = project_root / "locks"
        
        running = {}
        if locks_path.exists():
            for pid_file in locks_path.glob("*.pid"):
                service_name = pid_file.stem
                lock = cls(service_name, str(locks_path))
                pid = lock._read_pid_file()
                if pid and lock._is_process_running(pid):
                    running[service_name] = pid
        
        return running


def ensure_single_instance(service_name: str, force: bool = False) -> ProcessLock:
    """
    Convenience function to ensure single instance.
    Exits the program if another instance is running.
    
    Args:
        service_name: Unique name for this service
        force: If True, kill existing instance
        
    Returns:
        ProcessLock object (lock is held)
        
    Raises:
        SystemExit if another instance is running and force=False
    """
    lock = ProcessLock(service_name)
    if not lock.acquire(force=force):
        print(f"\n❌ ERROR: Another instance of '{service_name}' is already running!")
        print(f"   To force restart, use: --force or delete {lock.pid_file}\n")
        sys.exit(1)
    return lock

