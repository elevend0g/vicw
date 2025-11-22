"""Thread-safe async queue for offload jobs"""

import logging
import asyncio
from collections import deque
from typing import Optional
from data_models import OffloadJob

logger = logging.getLogger(__name__)


class OffloadQueue:
    """
    Thread-safe queue for offload jobs.
    Decouples hot path (enqueue) from cold path (process).
    """
    
    def __init__(self, max_size: int = 100):
        self.queue = deque()
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.enqueued_count = 0
        self.processed_count = 0
        self.dropped_count = 0
    
    async def enqueue(self, job: OffloadJob) -> bool:
        """
        Non-blocking enqueue. Returns False if queue is full.
        This is called from the hot path and should be very fast.
        """
        async with self.lock:
            if len(self.queue) >= self.max_size:
                logger.warning(
                    f"Offload queue is full ({self.max_size}). "
                    f"Dropping oldest job to make room."
                )
                self.queue.popleft()
                self.dropped_count += 1
            
            self.queue.append(job)
            self.enqueued_count += 1
            
            logger.debug(
                f"Queued offload job {job.job_id}. "
                f"Queue size: {len(self.queue)}"
            )
            return True
    
    async def dequeue(self) -> Optional[OffloadJob]:
        """
        Retrieve a single job from the queue.
        Returns None if queue is empty.
        """
        async with self.lock:
            if self.queue:
                job = self.queue.popleft()
                self.processed_count += 1
                return job
            return None
    
    async def dequeue_batch(self, batch_size: int) -> list[OffloadJob]:
        """
        Retrieve up to batch_size jobs from the queue.
        Called by the cold path worker.
        """
        async with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.queue))):
                job = self.queue.popleft()
                batch.append(job)
                self.processed_count += 1
            
            if batch:
                logger.debug(f"Dequeued batch of {len(batch)} jobs")
            
            return batch
    
    async def peek(self) -> Optional[OffloadJob]:
        """Peek at the next job without removing it"""
        async with self.lock:
            if self.queue:
                return self.queue[0]
            return None
    
    async def get_queue_size(self) -> int:
        """Get current queue size"""
        async with self.lock:
            return len(self.queue)
    
    async def is_empty(self) -> bool:
        """Check if queue is empty"""
        async with self.lock:
            return len(self.queue) == 0
    
    async def clear(self):
        """Clear all jobs from the queue"""
        async with self.lock:
            count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {count} jobs from queue")
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            "current_size": len(self.queue),
            "max_size": self.max_size,
            "enqueued_total": self.enqueued_count,
            "processed_total": self.processed_count,
            "dropped_total": self.dropped_count,
            "pending": self.enqueued_count - self.processed_count
        }
    
    def get_processed_count(self) -> int:
        """Get total number of processed jobs"""
        return self.processed_count
    
    def get_enqueued_count(self) -> int:
        """Get total number of enqueued jobs"""
        return self.enqueued_count
