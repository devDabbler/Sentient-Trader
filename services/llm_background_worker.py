"""
LLM Background Worker
Processes async LLM requests from queue in background thread
"""

import time
import logging
import threading
from queue import Empty
from typing import Optional

from services.llm_request_manager import get_llm_manager


logger = logging.getLogger(__name__)


class LLMBackgroundWorker:
    """
    Background worker for processing async LLM requests
    
    Features:
    - Runs in separate thread
    - Processes queued requests asynchronously
    - Respects priority ordering
    - Graceful shutdown
    """
    
    def __init__(self, poll_interval: float = 0.5):
        """
        Initialize background worker
        
        Args:
            poll_interval: How often to check queue (seconds)
        """
        self.manager = get_llm_manager()
        self.poll_interval = poll_interval
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.processed_count = 0
        self.error_count = 0
        
        logger.info("LLM Background Worker initialized")
    
    def start(self):
        """Start the background worker thread"""
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="LLM-Worker"
        )
        self.worker_thread.start()
        logger.info("LLM Background Worker started")
    
    def stop(self):
        """Stop the background worker thread"""
        if not self.is_running:
            logger.warning("Worker not running")
            return
        
        self.is_running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info(
            f"LLM Background Worker stopped "
            f"(Processed: {self.processed_count}, Errors: {self.error_count})"
        )
    
    def _worker_loop(self):
        """Main worker loop"""
        logger.info("Worker loop started")
        
        while self.is_running:
            try:
                # Try to get request from queue
                try:
                    priority_value, timestamp, llm_request = self.manager.queue.get(
                        block=True,
                        timeout=self.poll_interval
                    )
                    
                    logger.debug(
                        f"Processing queued request: {llm_request.request_id} "
                        f"(Priority: {llm_request.priority.name})"
                    )
                    
                    # Process the request
                    response = self.manager._process_request(llm_request)
                    
                    if response:
                        self.processed_count += 1
                        logger.info(
                            f"Processed async request {llm_request.request_id} "
                            f"(Total: {self.processed_count})"
                        )
                    else:
                        self.error_count += 1
                        logger.error(
                            f"Failed to process request {llm_request.request_id} "
                            f"(Total errors: {self.error_count})"
                        )
                    
                    # Mark task as done
                    self.manager.queue.task_done()
                
                except Empty:
                    # No requests in queue, continue polling
                    continue
            
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self.error_count += 1
                time.sleep(1.0)  # Back off on error
        
        logger.info("Worker loop ended")
    
    def get_status(self) -> dict:
        """Get worker status"""
        return {
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'queue_size': self.manager.queue.qsize(),
            'thread_alive': self.worker_thread.is_alive() if self.worker_thread else False
        }


# Global worker instance
_worker_instance: Optional[LLMBackgroundWorker] = None
_worker_lock = threading.Lock()


def get_llm_worker() -> LLMBackgroundWorker:
    """Get singleton instance of LLM Background Worker"""
    global _worker_instance
    
    if _worker_instance is None:
        with _worker_lock:
            if _worker_instance is None:
                _worker_instance = LLMBackgroundWorker()
    
    return _worker_instance


def start_llm_worker():
    """Start the background worker"""
    worker = get_llm_worker()
    worker.start()
    return worker


def stop_llm_worker():
    """Stop the background worker"""
    global _worker_instance
    
    if _worker_instance:
        _worker_instance.stop()
        _worker_instance = None


if __name__ == "__main__":
    # Test the worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start worker
    worker = start_llm_worker()
    
    # Submit some test requests
    from services.llm_helper import llm_request
    
    logger.info("Submitting test requests...")
    
    for i in range(5):
        llm_request(
            prompt=f"Test request {i}",
            service_name="test_service",
            priority="LOW"
        )
        logger.info(f"Submitted request {i}")
        time.sleep(0.5)
    
    # Wait for processing
    logger.info("Waiting for queue to empty...")
    time.sleep(10)
    
    # Check status
    status = worker.get_status()
    logger.info(f"Worker status: {status}")
    
    # Stop worker
    stop_llm_worker()
    logger.info("Test complete")
