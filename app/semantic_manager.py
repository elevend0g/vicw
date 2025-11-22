"""Semantic manager for cold path processing"""

import uuid
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from data_models import OffloadJob, OffloadResult, RAGResult
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from config import RAG_TOP_K_SEMANTIC, RAG_TOP_K_RELATIONAL

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger('vicw.metrics')


class SemanticManager:
    """
    Cold path: processes offload jobs asynchronously.
    Handles embeddings, summarization, and knowledge graph updates.
    """
    
    def __init__(
        self,
        embedding_model: Any,
        redis_storage: RedisStorage,
        qdrant_db: QdrantVectorDB,
        neo4j_graph: Neo4jKnowledgeGraph,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        self.embedding_model = embedding_model
        self.redis_storage = redis_storage
        self.qdrant_db = qdrant_db
        self.neo4j_graph = neo4j_graph
        self.executor = executor  # For CPU-bound operations like embedding
        
        logger.info("SemanticManager initialized")
    
    def _summarize_sync(self, text: str) -> str:
        """
        Simple extractive summarization without LLM (CPU-bound operation).
        In production, you might use a dedicated summarization model.
        """
        if len(text) < 100:
            return text
        
        try:
            lines = text.split('\n')
            
            # Keep first 3 and last 3 lines for context
            if len(lines) <= 6:
                summary = text[:500]
            else:
                first_part = '\n'.join(lines[:3])
                last_part = '\n'.join(lines[-3:])
                summary = f"{first_part}\n[...]\n{last_part}"
            
            # Truncate to reasonable length
            if len(summary) > 500:
                summary = summary[:500] + "..."
            
            logger.debug(f"Created extractive summary ({len(summary)} chars from {len(text)} chars)")
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return text[:200] + "..."
    
    def _embed_sync(self, text: str) -> np.ndarray:
        """
        Synchronous embedding (CPU-bound operation).
        Runs in thread pool to avoid blocking event loop.
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            # Return zero vector on error
            return np.zeros(384)
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text asynchronously.
        Returns embedding as list of floats.
        """
        if not self.executor:
            logger.warning("No executor available for embedding generation")
            return None
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(self.executor, self._embed_sync, text)
        
        if embedding is not None:
            return embedding.tolist()
        return None
    
    async def process_job(self, job: OffloadJob) -> Optional[OffloadResult]:
        """
        Process a single offload job asynchronously.
        Steps:
        1. Generate summary (CPU-bound, in executor)
        2. Generate embedding (CPU-bound, in executor)
        3. Store in Redis (async I/O)
        4. Store in Qdrant (async I/O)
        5. Update Neo4j (async I/O)
        """
        job_start_time = time.time()
        logger.info(f"Processing offload job {job.job_id} ({job.token_count} tokens)")
        
        try:
            # Run CPU-bound tasks in parallel using executor
            loop = asyncio.get_event_loop()
            
            # Generate summary and embedding in parallel
            if self.executor:
                summary_task = loop.run_in_executor(self.executor, self._summarize_sync, job.chunk_text)
                embedding_task = loop.run_in_executor(self.executor, self._embed_sync, job.chunk_text)
                summary, embedding_np = await asyncio.gather(summary_task, embedding_task)
                embedding = embedding_np.tolist()
            else:
                summary = self._summarize_sync(job.chunk_text)
                embedding_np = self._embed_sync(job.chunk_text)
                embedding = embedding_np.tolist()
            
            # Store in Redis (async I/O)
            job.summary = summary
            redis_success = await self.redis_storage.store_chunk(job, summary)
            
            if not redis_success:
                logger.warning(f"Failed to store job {job.job_id} in Redis")
            
            # Store in Qdrant (async I/O)
            metadata = {
                "job_id": job.job_id,
                "summary": summary,
                "token_count": job.token_count,
                "timestamp": job.timestamp
            }
            await self.qdrant_db.upsert_vector(job.job_id, embedding, metadata)
            
            # Update Neo4j knowledge graph (async I/O)
            await self.neo4j_graph.update_graph_from_context(job.job_id, summary)
            
            process_time = (time.time() - job_start_time) * 1000
            logger.info(f"Completed offload job {job.job_id} in {process_time:.2f}ms")
            
            metrics_logger.info(
                f"OFFLOAD_JOB_COMPLETE | "
                f"job_id={job.job_id} | "
                f"time_ms={process_time:.2f} | "
                f"tokens={job.token_count}"
            )
            
            return OffloadResult(
                job_id=job.job_id,
                summary=summary,
                embedding=embedding,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing offload job {job.job_id}: {e}")
            return OffloadResult(
                job_id=job.job_id,
                summary="",
                embedding=[],
                success=False,
                error=str(e)
            )
    
    async def query_summaries(self, query_embedding: List[float], top_k: int = None) -> List[str]:
        """
        Retrieve summaries using semantic search.
        1. Search Qdrant for similar vectors
        2. Fetch summaries from Redis
        """
        if top_k is None:
            top_k = RAG_TOP_K_SEMANTIC
        
        start_time = time.time()
        
        try:
            # Search Qdrant for similar vectors
            search_results = await self.qdrant_db.search(query_embedding, top_k=top_k)
            
            if not search_results:
                logger.info("No semantic search results found")
                return []
            
            # Extract job_ids
            job_ids = [result['job_id'] for result in search_results]
            
            # Fetch summaries from Redis
            chunks = await self.redis_storage.get_chunks_by_ids(job_ids, fields=['summary'])
            summaries = [chunk.get('summary', '') for chunk in chunks if chunk.get('summary')]
            
            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(summaries)} summaries in {retrieval_time:.2f}ms")
            
            metrics_logger.info(
                f"SEMANTIC_RETRIEVAL | "
                f"summaries={len(summaries)} | "
                f"latency_ms={retrieval_time:.2f} | "
                f"top_k={top_k}"
            )
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error querying summaries: {e}")
            return []
    
    async def query_memory(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k_semantic: int = None,
        top_k_relational: int = None
    ) -> RAGResult:
        """
        Hybrid retrieval: semantic (Qdrant/Redis) + relational (Neo4j).
        Returns RAGResult with both types of retrieved information.
        """
        start_time = time.time()
        
        if top_k_semantic is None:
            top_k_semantic = RAG_TOP_K_SEMANTIC
        if top_k_relational is None:
            top_k_relational = RAG_TOP_K_RELATIONAL
        
        try:
            # Run semantic and relational queries in parallel
            semantic_task = self.query_summaries(query_embedding, top_k=top_k_semantic)
            relational_task = self.neo4j_graph.relational_query(query_text, limit=top_k_relational)
            
            semantic_summaries, relational_facts = await asyncio.gather(
                semantic_task,
                relational_task
            )
            
            retrieval_time = (time.time() - start_time) * 1000
            
            result = RAGResult(
                semantic_chunks=semantic_summaries,
                relational_facts=relational_facts,
                retrieval_time_ms=retrieval_time
            )
            
            logger.info(
                f"Hybrid retrieval complete: "
                f"{len(semantic_summaries)} semantic + {len(relational_facts)} relational "
                f"in {retrieval_time:.2f}ms"
            )
            
            metrics_logger.info(
                f"HYBRID_RETRIEVAL | "
                f"semantic={len(semantic_summaries)} | "
                f"relational={len(relational_facts)} | "
                f"latency_ms={retrieval_time:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return RAGResult()
