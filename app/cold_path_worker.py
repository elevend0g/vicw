"""Background worker for processing offload queue"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from offload_queue import OffloadQueue
from semantic_manager import SemanticManager
from config import COLD_PATH_BATCH_SIZE, COLD_PATH_WORKERS

logger = logging.getLogger(__name__)


class ColdPathWorker:
    """
    Background worker that continuously processes the offload queue.
    Runs independently from the hot path.
    """
    
    def __init__(self, offload_queue: OffloadQueue, semantic_manager: SemanticManager):
        self.offload_queue = offload_queue
        self.semantic_manager = semantic_manager
        self.is_running = False
        self.is_paused = False
        self.processed_count = 0
        self.failed_count = 0
        self.worker_task: asyncio.Task = None
        
        # Dedicated thread pool for cold path CPU-bound operations
        self.executor = ThreadPoolExecutor(
            max_workers=COLD_PATH_WORKERS,
            thread_name_prefix='cold_path'
        )
        
        # Inject executor into semantic manager
        self.semantic_manager.executor = self.executor
        
        logger.info(f"ColdPathWorker initialized with {COLD_PATH_WORKERS} workers")
    
    async def start(self):
        """Start the background worker"""
        if self.is_running:
            logger.warning("ColdPathWorker already running")
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("ColdPathWorker started")
    
    async def stop(self):
        """Stop the background worker gracefully"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor and wait for pending tasks
        self.executor.shutdown(wait=True)
        logger.info("ColdPathWorker stopped")
    
    async def pause(self):
        """Pause processing (useful during LLM generation)"""
        self.is_paused = True
        logger.debug("ColdPathWorker paused")
    
    async def resume(self):
        """Resume processing"""
        self.is_paused = False
        logger.debug("ColdPathWorker resumed")
    
    async def _worker_loop(self):
        """
        Main worker loop: continuously fetch and process offload jobs.
        Runs independently from the hot path.
        """
        logger.info("ColdPathWorker loop started")
        
        while self.is_running:
            try:
                # Skip processing if paused
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Fetch batch of jobs
                batch = await self.offload_queue.dequeue_batch(COLD_PATH_BATCH_SIZE)
                
                if batch:
                    logger.info(f"Processing batch of {len(batch)} offload jobs")
                    await self._process_batch(batch)
                else:
                    # Sleep briefly to avoid busy-waiting when queue is empty
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                logger.info("ColdPathWorker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cold path worker loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _process_batch(self, batch: list):
        """Process a batch of offload jobs concurrently"""
        # Process jobs concurrently
        tasks = [self.semantic_manager.process_job(job) for job in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        for result in results:
            if isinstance(result, Exception):
                self.failed_count += 1
                logger.error(f"Job processing failed: {result}")
            elif result and result.success:
                self.processed_count += 1
            else:
                self.failed_count += 1
        
        logger.info(
            f"Batch complete. Total processed: {self.processed_count}, "
            f"failed: {self.failed_count}"
        )
    
    async def process_batch(self):
        """
        Public method to process a single batch.
        Useful for manual triggering or testing.
        """
        batch = await self.offload_queue.dequeue_batch(COLD_PATH_BATCH_SIZE)
        if batch:
            await self._process_batch(batch)
            return len(batch)
        return 0
    
    async def shutdown(self):
        """Alias for stop() for consistency"""
        await self.stop()
    
    def get_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0
                else 0
            )
        }
