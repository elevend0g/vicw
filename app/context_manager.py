"""Hot path context management with deterministic pressure control"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional

from data_models import OffloadJob, PinnedHeader
from offload_queue import OffloadQueue
from semantic_manager import SemanticManager
from config import (
    MAX_CONTEXT_TOKENS,
    OFFLOAD_THRESHOLD,
    TARGET_AFTER_RELIEF,
    HYSTERESIS_THRESHOLD
)

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger('vicw.metrics')


class ContextManager:
    """
    Manages hot path with context pressure handling.
    
    Key responsibilities:
    - Track token counts in working context
    - Trigger offload when threshold is reached
    - Maintain pinned state that never gets offloaded
    - Support RAG injection for retrieved memories
    """
    
    def __init__(
        self,
        max_context: int,
        offload_queue: OffloadQueue,
        embedding_model: Any,
        semantic_manager: Optional[SemanticManager] = None
    ):
        self.max_context = max_context
        self.working_context: List[Dict[str, str]] = []
        self.offload_queue = offload_queue
        self.embedding_model = embedding_model
        self.semantic_manager = semantic_manager
        self.tokenizer = None
        self.offload_job_count = 0
        self.placeholder_markers: Dict[str, int] = {}
        self.last_relief_tokens = 0  # For hysteresis
        
        # Pinned state header (never offloaded)
        self.pinned_header = PinnedHeader()
        
        logger.info(f"ContextManager initialized (max_context={max_context})")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~0.75 tokens per word
        # In production, use actual tokenizer
        return int(len(text.split()) / 0.75)
    
    def _token_count(self) -> int:
        """Calculate total token count in working context"""
        total = 0
        for msg in self.working_context:
            msg_text = f"{msg['role']}: {msg['content']}"
            total += self._estimate_tokens(msg_text)
        return total
    
    async def _create_placeholder_card(self, job_id: str, token_count: int, message_count: int) -> Dict[str, str]:
        """
        Create a lightweight reference card for offloaded content.
        This is inserted in place of the removed messages.
        """
        return {
            "role": "system",
            "content": f"[ARCHIVED mem_id:{job_id} tokens:{token_count} msgs:{message_count}]"
        }
    
    async def _relieve_pressure(self):
        """
        HOT PATH: Extract and QUEUE oldest context for async processing.
        Returns immediately without waiting for semantic operations.
        
        Implements:
        - Shed-to-target: Drop to TARGET_AFTER_RELIEF instead of just barely fitting
        - Hysteresis: Don't re-trigger until HYSTERESIS_THRESHOLD is crossed
        """
        relief_start_time = time.time()
        tokens_before = self._token_count()
        
        logger.info("=" * 60)
        logger.info("PRESSURE RELIEF TRIGGERED (HOT PATH)")
        logger.info(f"Context tokens before relief: {tokens_before}/{self.max_context}")
        logger.info("=" * 60)
        
        # Calculate target tokens
        target_tokens = int(self.max_context * TARGET_AFTER_RELIEF)
        tokens_to_extract = tokens_before - target_tokens
        extracted_tokens = 0
        extracted_messages = []
        
        # Extract messages until we've removed enough tokens
        # Never extract pinned header or system messages
        while extracted_tokens < tokens_to_extract and len(self.working_context) > 1:
            # Skip if first message is system message or pinned
            if self.working_context[0]['role'] == 'system':
                break
            
            msg = self.working_context.pop(0)
            extracted_messages.append(msg)
            
            msg_text = f"{msg['role']}: {msg['content']}"
            extracted_tokens += self._estimate_tokens(msg_text)
        
        # Convert extracted messages to chunk text
        chunk_text = "\n".join([f"{m['role']}: {m['content']}" for m in extracted_messages])
        
        logger.info(f"Extracted {len(extracted_messages)} messages (~{extracted_tokens} tokens)")
        
        # Create offload job (NOT processed yet, just queued)
        self.offload_job_count += 1
        job = OffloadJob.create(
            chunk_text=chunk_text,
            token_count=extracted_tokens,
            message_count=len(extracted_messages),
            metadata={"relief_num": self.offload_job_count}
        )
        
        # HOT PATH: Enqueue for async processing (non-blocking)
        await self.offload_queue.enqueue(job)
        
        # Insert lightweight placeholder in context
        placeholder = await self._create_placeholder_card(
            job.job_id,
            extracted_tokens,
            len(extracted_messages)
        )
        self.working_context.insert(0, placeholder)
        self.placeholder_markers[job.job_id] = 0
        
        tokens_after = self._token_count()
        self.last_relief_tokens = tokens_after
        relief_time = (time.time() - relief_start_time) * 1000
        
        logger.info(f"Pressure relief complete in {relief_time:.2f}ms (HOT PATH)")
        logger.info(f"Context tokens after relief: {tokens_after}/{self.max_context}")
        logger.info(f"Offload queued: job_id={job.job_id}")
        logger.info("=" * 60)
        
        # Log metrics
        metrics_logger.info(
            f"PRESSURE_RELIEF_HOT_PATH | "
            f"tokens_before={tokens_before} | "
            f"tokens_after={tokens_after} | "
            f"job_id={job.job_id} | "
            f"relief_time_ms={relief_time:.2f}"
        )
    
    async def add_message(self, role: str, content: str):
        """
        Main entry point: add message and trigger pressure relief if needed.
        This is the HOT PATH and should be as fast as possible.
        
        Implements hysteresis to prevent thrashing.
        """
        # Add the new message
        self.working_context.append({"role": role, "content": content})
        
        # Check context pressure
        current_tokens = self._token_count()
        pressure_threshold = int(self.max_context * OFFLOAD_THRESHOLD)
        hysteresis_threshold = int(self.max_context * HYSTERESIS_THRESHOLD)
        pressure_percentage = (current_tokens / self.max_context) * 100
        
        logger.info(
            f"Context pressure: {current_tokens}/{self.max_context} tokens "
            f"({pressure_percentage:.1f}%)"
        )
        
        metrics_logger.info(
            f"CONTEXT_PRESSURE | "
            f"tokens={current_tokens} | "
            f"max={self.max_context} | "
            f"percentage={pressure_percentage:.1f} | "
            f"message_role={role}"
        )
        
        # Trigger relief if needed with hysteresis
        # Only trigger if we exceed threshold AND we're above hysteresis point
        if current_tokens > pressure_threshold:
            # Check hysteresis: has enough new content accumulated?
            if self.last_relief_tokens == 0 or current_tokens > hysteresis_threshold:
                logger.info(
                    f"TRIGGER: Token count ({current_tokens}) exceeds threshold ({pressure_threshold})"
                )
                await self._relieve_pressure()
            else:
                logger.debug(
                    f"Hysteresis: Not triggering relief yet "
                    f"(current={current_tokens}, hysteresis={hysteresis_threshold})"
                )
    
    async def augment_context_with_memory(self, query_text: str, top_k_semantic: int = 2, top_k_relational: int = 5) -> int:
        """
        Performs semantic (Qdrant/Redis) and relational (Neo4j) search and injects results.
        Returns total number of items injected.
        
        This is called during generation to enhance context with relevant memories.
        """
        if not self.semantic_manager:
            logger.warning("No semantic manager available for RAG")
            return 0
        
        rag_start_time = time.time()
        
        try:
            # 1. Generate embedding for query
            query_embedding = await self.semantic_manager.generate_embedding(query_text)
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding for RAG")
                return 0
            
            # 2. Query memory systems (hybrid retrieval)
            rag_result = await self.semantic_manager.query_memory(
                query_embedding,
                query_text,
                top_k_semantic=top_k_semantic,
                top_k_relational=top_k_relational
            )
            
            if rag_result.is_empty():
                logger.info("RAG skipped: No relevant memories found")
                return 0
            
            # 3. Convert to context message and inject
            rag_message = rag_result.to_context_message()
            
            if rag_message:
                # Inject before the last user message if possible
                if self.working_context and self.working_context[-1]['role'] == 'user':
                    self.working_context.insert(-1, rag_message)
                else:
                    self.working_context.append(rag_message)
                
                rag_time = (time.time() - rag_start_time) * 1000
                logger.info(
                    f"RAG complete: Injected {rag_result.total_items} items in {rag_time:.2f}ms"
                )
                
                metrics_logger.info(
                    f"RAG_INJECTION | "
                    f"semantic={len(rag_result.semantic_chunks)} | "
                    f"relational={len(rag_result.relational_facts)} | "
                    f"total_time_ms={rag_time:.2f}"
                )
                
                return rag_result.total_items
            
            return 0
            
        except Exception as e:
            logger.error(f"Error during RAG augmentation: {e}")
            return 0
    
    def get_context_window(self) -> List[Dict[str, str]]:
        """
        Get the current context window for LLM generation.
        Optionally includes pinned header.
        """
        context = []
        
        # Add pinned header if it has content
        pinned_msg = self.pinned_header.to_context_message()
        if pinned_msg:
            context.append(pinned_msg)
        
        # Add working context
        context.extend(self.working_context)
        
        return context
    
    def update_pinned_header(self, **kwargs):
        """Update pinned header fields"""
        for key, value in kwargs.items():
            if hasattr(self.pinned_header, key):
                setattr(self.pinned_header, key, value)
                logger.debug(f"Updated pinned header: {key}={value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics"""
        return {
            "current_tokens": self._token_count(),
            "max_tokens": self.max_context,
            "message_count": len(self.working_context),
            "offload_count": self.offload_job_count,
            "pressure_percentage": (self._token_count() / self.max_context) * 100,
            "pinned_header": self.pinned_header.to_dict()
        }
