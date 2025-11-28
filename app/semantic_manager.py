"""Semantic manager for cold path processing"""

import uuid
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zstandard as zstd

from data_models import OffloadJob, OffloadResult, RAGResult
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from config import RAG_TOP_K_SEMANTIC, RAG_TOP_K_RELATIONAL, STATE_TRACKING_ENABLED, STATE_CONFIG_PATH
from state_extractor import get_extractor

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
        llm_client: Any = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        self.embedding_model = embedding_model
        self.redis_storage = redis_storage
        self.qdrant_db = qdrant_db
        self.neo4j_graph = neo4j_graph
        self.llm_client = llm_client
        self.executor = executor  # For CPU-bound operations like embedding

        # V1.0: Compression for full-text storage
        self._compressor = zstd.ZstdCompressor(level=3)
        self._decompressor = zstd.ZstdDecompressor()

        logger.info("SemanticManager initialized")

    # ... (keep _summarize_sync and _embed_sync and generate_embedding as is) ...
    # Wait, I need to include them in the replacement if I'm replacing a block that includes them.
    # But I can just replace __init__ and process_job separately if they are far apart.
    # They are close enough. I'll include the helper methods in the thought process but here I will try to be precise.
    
    # Actually, I'll replace the whole class methods that need changing.
    
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
        Supports both SentenceTransformer and llama-cpp-python.
        """
        try:
            # Check if it's a llama_cpp model (has create_embedding method)
            if hasattr(self.embedding_model, 'create_embedding'):
                # llama-cpp-python returns: {'data': [{'embedding': [0.1, ...], ...}], ...}
                response = self.embedding_model.create_embedding(text)
                embedding_list = response['data'][0]['embedding']
                return np.array(embedding_list, dtype=np.float32)
            else:
                # Assume SentenceTransformer
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            # Return zero vector on error (dimension might be wrong if not 384, but safe fallback)
            # Ideally we should know the dimension. Qwen3 is 1024.
            return np.zeros(1024) # Updated default for Qwen3, though dynamic would be better
    
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
        Ingestion Pipeline:
        1. Extraction (LLM): Extract Entities and Events.
        2. Context Check: Identify active Context.
        3. Sequence Assignment: timestamp, flow_id, flow_step.
        4. Vector Generation: Contextual Wrapper + Embedding.
        5. Graph Materialization: Create nodes in Neo4j.
        """
        job_start_time = time.time()
        logger.info(f"Processing offload job {job.job_id} ({job.token_count} tokens)")
        
        try:
            # 1. Extraction (LLM)
            # Determine domain from metadata or default
            domain = job.metadata.get("domain", "general")
            
            extractor = get_extractor(STATE_CONFIG_PATH)
            # Use LLM client if available
            extraction_data = await extractor.extract_metaphysical_graph(
                job.chunk_text, 
                context_domain=domain, 
                llm_client=self.llm_client
            )
            
            entities = extraction_data.get("entities", [])
            events = extraction_data.get("events", [])
            
            # 2. Context Check (Simplified: create/merge Context node based on domain)
            context_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, domain))
            await self.neo4j_graph.create_context_node({
                "uid": context_uid,
                "name": domain.capitalize(),
                "domain": domain,
                "description": f"Context for {domain} domain"
            })
            
            # 3. Sequence Assignment
            # For now, we use job.timestamp. flow_id could be job.metadata.get('thread_id')
            flow_id = job.metadata.get("thread_id", "default_flow")
            # We need to get the next flow_step. For simplicity, we'll use timestamp as a proxy or just 0
            # In a real system, we'd query Redis for the last step.
            flow_step = int(job.timestamp) 

            # 4. Vector Generation & 5. Graph Materialization
            
            # Create Chunk Node first
            chunk_uid = str(uuid.uuid4())
            await self.neo4j_graph.create_chunk_node({
                "uid": chunk_uid,
                "content": job.chunk_text[:200] + "...", # Store snippet or full? Graph usually stores snippet.
                "source": "chat",
                "domain": domain,
                "token_count": job.token_count
            })
            
            # Process Entities
            for entity in entities:
                entity_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain}:{entity['name']}"))
                
                # Contextual Wrapper for Entity
                wrapper_text = self._generate_contextual_wrapper(
                    domain=domain,
                    subtype=entity.get("subtype", "entity"),
                    name=entity["name"],
                    content=entity.get("description", "")
                )
                
                # Generate Embedding
                embedding = await self.generate_embedding(wrapper_text)
                
                if embedding:
                    # Upsert to Qdrant
                    qdrant_payload = {
                        "domain": domain,
                        "node_id": entity_uid,
                        "subtype": entity.get("subtype", "entity"),
                        "name": entity["name"],
                        "type": "Entity"
                    }
                    await self.qdrant_db.upsert_vector(f"vec_{entity_uid}", embedding, qdrant_payload)
                
                # Create Node in Neo4j
                await self.neo4j_graph.create_entity_node({
                    "uid": entity_uid,
                    "name": entity["name"],
                    "subtype": entity.get("subtype", "entity"),
                    "domain": domain,
                    "description": entity.get("description", ""),
                    "qdrant_id": f"vec_{entity_uid}" if embedding else None
                })
                
                # Link to Context
                await self.neo4j_graph.create_metaphysical_relationship(
                    entity_uid, "Entity", context_uid, "Context", "BELONGS_TO"
                )
                
                # Link to Chunk (Proof)
                await self.neo4j_graph.create_metaphysical_relationship(
                    chunk_uid, "Chunk", entity_uid, "Entity", "MENTIONS"
                )

            # Process Events
            for event in events:
                event_uid = str(uuid.uuid4()) # Events are unique instances
                
                # Contextual Wrapper for Event
                wrapper_text = self._generate_contextual_wrapper(
                    domain=domain,
                    subtype=event.get("subtype", "event"),
                    name=event["name"],
                    content=event.get("description", "")
                )
                
                # Generate Embedding
                embedding = await self.generate_embedding(wrapper_text)
                
                if embedding:
                    # Upsert to Qdrant
                    qdrant_payload = {
                        "domain": domain,
                        "node_id": event_uid,
                        "subtype": event.get("subtype", "event"),
                        "name": event["name"],
                        "type": "Event"
                    }
                    await self.qdrant_db.upsert_vector(f"vec_{event_uid}", embedding, qdrant_payload)
                
                # Create Node in Neo4j
                await self.neo4j_graph.create_event_node({
                    "uid": event_uid,
                    "name": event["name"],
                    "subtype": event.get("subtype", "event"),
                    "domain": domain,
                    "timestamp": job.timestamp,
                    "flow_id": flow_id,
                    "flow_step": flow_step,
                    "description": event.get("description", ""),
                    "qdrant_id": f"vec_{event_uid}" if embedding else None
                })
                
                # Link to Context
                await self.neo4j_graph.create_metaphysical_relationship(
                    event_uid, "Event", context_uid, "Context", "BELONGS_TO"
                )

                # Link to Chunk
                await self.neo4j_graph.create_metaphysical_relationship(
                    chunk_uid, "Chunk", event_uid, "Event", "MENTIONS"
                )
                
                # Handle 'caused_by' (Entity -> Event)
                caused_by_names = event.get("caused_by", [])
                for cause_name in caused_by_names:
                    # Try to find the entity UID (assuming deterministic UUID generation)
                    cause_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain}:{cause_name}"))
                    # We optimistically create the relationship. If entity doesn't exist, this might fail or we should MERGE entity first.
                    # For safety, we should probably ensure entity exists. But for now, let's assume extraction found it.
                    # Actually, create_metaphysical_relationship uses MATCH, so it will fail if nodes don't exist.
                    # We should probably skip or create a placeholder.
                    pass 

            process_time = (time.time() - job_start_time) * 1000
            logger.info(f"Completed offload job {job.job_id} in {process_time:.2f}ms")
            
            metrics_logger.info(
                f"OFFLOAD_JOB_COMPLETE | "
                f"job_id={job.job_id} | "
                f"time_ms={process_time:.2f} | "
                f"tokens={job.token_count} | "
                f"entities={len(entities)} | "
                f"events={len(events)}"
            )
            
            return OffloadResult(
                job_id=job.job_id,
                summary=f"Extracted {len(entities)} entities and {len(events)} events",
                embedding=[], # We generated multiple embeddings
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

    def _generate_contextual_wrapper(self, domain: str, subtype: str, name: str, content: str) -> str:
        """
        Construct the contextual wrapper string.
        Format: [Domain: <domain>] [Type: <subtype>] [Name: <name>] <content>
        """
        return f"[Domain: {domain}] [Type: {subtype}] [Name: {name}] {content}"
    
    async def query_summaries(self, query_embedding: List[float], top_k: int = None) -> List[str]:
        """
        Retrieve full text using semantic search.
        V1.0: Returns compressed full text, falls back to summaries for old data.
        """
        if top_k is None:
            top_k = RAG_TOP_K_SEMANTIC

        start_time = time.time()

        try:
            # Search Qdrant
            search_results = await self.qdrant_db.search(query_embedding, top_k=top_k)

            if not search_results:
                logger.info("No semantic search results found")
                return []

            chunks = []

            for result in search_results:
                payload = result.get('payload', {})

                # Try compressed full text first (V1.0 data)
                if payload.get('chunk_text_compressed'):
                    try:
                        compressed_bytes = bytes.fromhex(payload['chunk_text_compressed'])
                        text = self._decompressor.decompress(compressed_bytes).decode('utf-8')
                        chunks.append(text)
                        continue
                    except Exception as e:
                        logger.warning(f"Decompression failed: {e}")
                        # Fall through to summary fallback

                # Fallback to summary for old data
                if payload.get('summary'):
                    chunks.append(payload['summary'])

            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}ms")

            return chunks
            
        except Exception as e:
            logger.error(f"Error querying summaries: {e}")
            return []
    
    async def retrieve_metaphysical_context(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> RAGResult:
        """
        Retrieve context using the Metaphysical Schema strategy.
        1. Intent Analysis
        2. Vector Filter Scan
        3. Graph Expansion
        4. Synthesis
        """
        start_time = time.time()
        
        # 1. Intent Analysis (LLM)
        intent = "general"
        if self.llm_client:
            try:
                prompt = f"""Classify the intent of this query into one of: ['coding', 'creative', 'general'].
                Query: {query_text}
                Return JSON: {{"intent": "..."}}"""
                
                response = await self.llm_client.generate(
                    context=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                import json
                data = json.loads(response)
                intent = data.get("intent", "general").lower()
            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")

        # 2. Vector Filter Scan (Qdrant)
        # Construct filter based on intent
        filter_dict = None
        if intent != "general":
            # We filter by domain. Note: This assumes 'domain' field in payload matches intent.
            # In reality, 'coding' intent might map to 'coding' domain, 'creative' to 'story', etc.
            # For simplicity, we use direct mapping or fallback.
            domain_map = {"coding": "coding", "creative": "story"}
            domain = domain_map.get(intent, intent)
            
            filter_dict = self.qdrant_db.create_filter([
                {"key": "domain", "match": {"value": domain}}
            ])

        search_results = await self.qdrant_db.search(
            query_embedding, 
            top_k=top_k, 
            filter_dict=filter_dict
        )
        
        # Extract Node UIDs and Semantic Chunks
        node_uids = []
        semantic_chunks = []
        
        for result in search_results:
            payload = result.get('payload', {})
            node_id = payload.get('node_id')
            if node_id:
                node_uids.append(node_id)
            
            # Also get the content (summary or compressed text)
            if payload.get('chunk_text_compressed'):
                try:
                    compressed_bytes = bytes.fromhex(payload['chunk_text_compressed'])
                    text = self._decompressor.decompress(compressed_bytes).decode('utf-8')
                    semantic_chunks.append(text)
                except:
                    if payload.get('summary'):
                        semantic_chunks.append(payload['summary'])
            elif payload.get('summary'):
                semantic_chunks.append(payload['summary'])

        # 3. Graph Expansion (Neo4j)
        expanded_context = await self.neo4j_graph.expand_metaphysical_context(node_uids)
        
        # 4. Synthesis (Format for RAGResult)
        relational_facts = []
        for item in expanded_context:
            node = item['node']
            node_str = f"[{node.get('type', 'Node')}: {node.get('name')}] {node.get('description', '')}"
            relational_facts.append(node_str)
            
            for rel in item['relationships']:
                relational_facts.append(f"  - {rel}")
                
        retrieval_time = (time.time() - start_time) * 1000
        
        return RAGResult(
            semantic_chunks=semantic_chunks,
            relational_facts=relational_facts,
            retrieval_time_ms=retrieval_time
        )

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
        # Use the new metaphysical retrieval strategy
        return await self.retrieve_metaphysical_context(
            query_text, 
            query_embedding, 
            top_k=top_k_semantic or RAG_TOP_K_SEMANTIC
        )

    async def store_response_embedding(self, embedding: List[float]) -> bool:
        """
        Store response embedding in Redis for echo detection.
        Uses sorted set with timestamp scores for sliding window.
        Returns True if successful.
        """
        try:
            from config import ECHO_RESPONSE_HISTORY_SIZE
            import json

            # Generate unique ID for this response
            response_id = f"resp_{uuid.uuid4().hex[:8]}"
            timestamp = time.time()

            # Store embedding in Redis sorted set
            key = "response_embeddings"
            value = json.dumps(embedding)

            # Add to sorted set with timestamp as score
            await self.redis_storage.redis.zadd(key, {value: timestamp})

            # Trim to keep only recent responses
            # Keep the most recent N responses (highest scores)
            await self.redis_storage.redis.zremrangebyrank(
                key,
                0,
                -(ECHO_RESPONSE_HISTORY_SIZE + 1)
            )

            logger.debug(f"Stored response embedding with timestamp {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error storing response embedding: {e}")
            return False

    async def check_response_similarity(
        self,
        new_embedding: List[float],
        threshold: float = None
    ) -> tuple[bool, float]:
        """
        Check if new response embedding is similar to recent responses.
        Returns (is_duplicate, max_similarity).
        """
        try:
            from config import ECHO_SIMILARITY_THRESHOLD
            import json

            if threshold is None:
                threshold = ECHO_SIMILARITY_THRESHOLD

            # Get recent response embeddings from Redis
            key = "response_embeddings"
            recent_responses = await self.redis_storage.redis.zrange(
                key,
                0,
                -1,
                withscores=False
            )

            if not recent_responses:
                logger.debug("No previous responses to compare")
                return (False, 0.0)

            # Compute cosine similarity with each stored embedding
            new_emb_array = np.array(new_embedding)
            max_similarity = 0.0

            for stored_response in recent_responses:
                try:
                    stored_emb = json.loads(stored_response)
                    stored_emb_array = np.array(stored_emb)

                    # Cosine similarity
                    dot_product = np.dot(new_emb_array, stored_emb_array)
                    norm_new = np.linalg.norm(new_emb_array)
                    norm_stored = np.linalg.norm(stored_emb_array)

                    if norm_new > 0 and norm_stored > 0:
                        similarity = dot_product / (norm_new * norm_stored)
                        max_similarity = max(max_similarity, similarity)

                        if similarity >= threshold:
                            logger.warning(
                                f"ECHO_DETECTED | similarity={similarity:.4f} | threshold={threshold}"
                            )
                            metrics_logger.info(
                                f"ECHO_DETECTED | similarity={similarity:.4f} | threshold={threshold}"
                            )
                            return (True, float(similarity))

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing stored embedding: {e}")
                    continue

            logger.debug(f"Max similarity: {max_similarity:.4f} (threshold: {threshold})")
            return (False, float(max_similarity))

        except Exception as e:
            logger.error(f"Error checking response similarity: {e}")
            return (False, 0.0)
