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
        self.sleep_cycle_task = asyncio.create_task(self._sleep_cycle_loop())
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
        
        if self.sleep_cycle_task:
            self.sleep_cycle_task.cancel()
            try:
                await self.sleep_cycle_task
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

    async def _sleep_cycle_loop(self):
        """
        Background maintenance loop (Sleep Cycle).
        Consolidates old events into MacroEvents.
        """
        logger.info("Sleep Cycle loop started")
        import uuid
        import time
        
        while self.is_running:
            try:
                # Run every 60 seconds (for testing/demo purposes, usually 1 hour)
                await asyncio.sleep(60)
                
                if self.is_paused:
                    continue
                
                # 1. Find old events (e.g., older than 1 hour)
                # For testing, we might want to consolidate even recent ones if we want to verify logic.
                # Let's stick to 1 hour default but maybe configurable.
                events = await self.semantic_manager.neo4j_graph.get_old_events(hours=1, limit=10)
                
                if not events:
                    continue
                
                logger.info(f"Sleep Cycle: Found {len(events)} old events to consolidate")
                
                # Group by flow_id (simple strategy: just take batch)
                # We'll just take the first 5 events and consolidate them for now.
                batch_size = 5
                for i in range(0, len(events), batch_size):
                    chunk = events[i:i+batch_size]
                    if len(chunk) < 2: # Need at least 2 events to consolidate
                        continue
                        
                    # 2. Generate Summary for Macro-Event
                    descriptions = [e.get('description', '') for e in chunk]
                    combined_text = "\n".join(descriptions)
                    
                    summary = f"Consolidated sequence of {len(chunk)} events."
                    if self.semantic_manager.llm_client:
                        try:
                            prompt = f"Summarize these events into a single Macro-Event description:\n{combined_text}"
                            summary = await self.semantic_manager.llm_client.generate(
                                context=[{"role": "user", "content": prompt}]
                            )
                        except Exception as e:
                            logger.warning(f"Failed to generate macro summary: {e}")
                    
                    # 3. Create Macro-Event
                    macro_uid = str(uuid.uuid4())
                    macro_data = {
                        "uid": macro_uid,
                        "name": f"Macro-Event {int(time.time())}",
                        "description": summary,
                        "type": "MacroEvent",
                        "event_count": len(chunk)
                    }
                    
                    await self.semantic_manager.neo4j_graph.create_macro_event(macro_data)
                    
                    # 4. Consolidate (Link)
                    event_uids = [e['uid'] for e in chunk]
                    await self.semantic_manager.neo4j_graph.consolidate_events(event_uids, macro_uid)
                    
                    # 5. Embed Macro-Event
                    embedding = await self.semantic_manager.generate_embedding(summary)
                    if embedding:
                        await self.semantic_manager.qdrant_db.upsert_vector(
                            f"vec_{macro_uid}",
                            embedding,
                            {
                                "domain": "consolidated",
                                "node_id": macro_uid,
                                "type": "MacroEvent",
                                "name": macro_data["name"]
                            }
                        )
                        
                    logger.info(f"Sleep Cycle: Consolidated {len(chunk)} events into {macro_uid}")
                    
            except asyncio.CancelledError:
                logger.info("Sleep Cycle loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in Sleep Cycle loop: {e}")
                await asyncio.sleep(60)

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
