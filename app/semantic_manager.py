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
        0. Store raw chunk in Redis (FIRST - preserve data even if processing fails)
        1. Extraction (LLM): Extract Entities and Events.
        2. Context Check: Identify active Context.
        3. Sequence Assignment: timestamp, flow_id, flow_step.
        4. Vector Generation: Contextual Wrapper + Embedding.
        5. Graph Materialization: Create nodes in Neo4j.
        """
        job_start_time = time.time()
        logger.info(f"Processing offload job {job.job_id} ({job.token_count} tokens)")

        try:
            # 0. CRITICAL: Store raw chunk in Redis FIRST to prevent data loss
            # If extraction fails, at least we have the raw text preserved
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(self.executor, self._summarize_sync, job.chunk_text)
            stored = await self.redis_storage.store_chunk(job, summary)
            if stored:
                logger.info(f"Stored chunk {job.job_id} in Redis (raw text preserved)")
            else:
                logger.warning(f"Failed to store chunk {job.job_id} in Redis - continuing anyway")

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

            # Log extraction results
            if not entities and not events:
                logger.warning(f"Extraction returned no entities or events for job {job.job_id} (chunk stored in Redis)")
            else:
                logger.info(f"Extracted {len(entities)} entities and {len(events)} events from job {job.job_id}")
            
            # 2. Context Check (Simplified: create/merge Context node based on domain)
            context_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, domain))
            await self.neo4j_graph.create_context_node({
                "uid": context_uid,
                "name": domain.capitalize(),
                "domain": domain,
                "description": f"Context for {domain} domain"
            })
            
            # 3. Sequence Assignment
            # flow_id represents a logical thread/conversation
            flow_id = job.metadata.get("thread_id", "default_flow")

            # For flow_step, we assign sequential integers to events within this batch
            # In a production system with multiple batches per flow, we'd query Neo4j/Redis
            # for the max flow_step and increment from there. For now, we start from 0
            # within each batch and track events for NEXT edge creation.
            base_flow_step = 0  # Could query Redis: f"flow_step:{flow_id}" for global counter 

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

            # Track event UIDs for NEXT edge creation
            # Format: [(event_uid, flow_step, flow_id), ...]
            processed_events = []

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
            for event_idx, event in enumerate(events):
                event_uid = str(uuid.uuid4()) # Events are unique instances

                # Assign sequential flow_step
                event_flow_step = base_flow_step + event_idx

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

                # Create Node in Neo4j with correct flow_step
                await self.neo4j_graph.create_event_node({
                    "uid": event_uid,
                    "name": event["name"],
                    "subtype": event.get("subtype", "event"),
                    "domain": domain,
                    "timestamp": job.timestamp,
                    "flow_id": flow_id,
                    "flow_step": event_flow_step,
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
                
                # Handle 'caused_by' (Entity -> Event via INITIATED edge)
                caused_by_names = event.get("caused_by", [])
                for cause_name in caused_by_names:
                    # Compute entity UID (deterministic UUID based on domain:name)
                    cause_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain}:{cause_name}"))

                    try:
                        # Create INITIATED edge: Entity -> Event
                        await self.neo4j_graph.create_metaphysical_relationship(
                            cause_uid, "Entity",
                            event_uid, "Event",
                            "INITIATED"
                        )
                        logger.debug(f"Created INITIATED edge: {cause_name} -> {event['name']}")
                    except Exception as e:
                        # Entity might not exist if extraction found it in event but not in entities list
                        logger.warning(f"Failed to create INITIATED edge for {cause_name} -> {event['name']}: {e}")

                # TODO: Implement CAUSED edges (Event -> Event/Entity consequences)
                # Currently, the extraction format doesn't provide explicit causality data.
                # To implement this, we would need to:
                # 1. Enhance the extraction prompt in state_extractor.py to ask for "consequences"
                # 2. Extract consequence relationships here
                # 3. Create CAUSED edges with certainty scores
                # For now, temporal sequencing is captured via NEXT edges (see below)

                # Track this event for NEXT edge creation
                processed_events.append((event_uid, event_flow_step, flow_id))

            # Create NEXT edges between consecutive events in the same flow_id
            # Sort by flow_step to ensure correct ordering
            processed_events.sort(key=lambda x: x[1])  # Sort by flow_step

            for i in range(len(processed_events) - 1):
                current_event = processed_events[i]
                next_event = processed_events[i + 1]

                # Only create NEXT edge if both events are in the same flow
                if current_event[2] == next_event[2]:  # Same flow_id
                    try:
                        await self.neo4j_graph.create_metaphysical_relationship(
                            current_event[0], "Event",  # current event UID
                            next_event[0], "Event",     # next event UID
                            "NEXT"
                        )
                        logger.debug(f"Created NEXT edge: {current_event[0][:8]} -> {next_event[0][:8]} (flow_step {current_event[1]} -> {next_event[1]})")
                    except Exception as e:
                        logger.warning(f"Failed to create NEXT edge: {e}")

            if processed_events:
                logger.info(f"Created {len(processed_events)-1} NEXT edges for {len(processed_events)} events in flow '{flow_id}'")

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

    async def _analyze_intent_robust(self, query_text: str) -> str:
        """
        Robust intent analysis with retry logic and keyword fallback.
        Returns one of: ['coding', 'creative', 'general']

        Strategy:
        1. Try LLM-based classification (with retry)
        2. Fall back to keyword detection
        3. Default to 'general' if all else fails
        """
        import json
        import asyncio

        # 1. Try LLM-based classification with retry
        if self.llm_client:
            max_retries = 2
            retry_delay = 0.5  # seconds

            for attempt in range(max_retries):
                try:
                    prompt = f"""Classify the intent of this query into one of: ['coding', 'creative', 'general'].

Query: {query_text}

Return JSON: {{"intent": "coding"}} or {{"intent": "creative"}} or {{"intent": "general"}}"""

                    response = await self.llm_client.generate(
                        context=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        max_tokens=50,  # Keep it short
                        temperature=0.0  # Deterministic for classification
                    )

                    # Handle empty response
                    if not response or not response.strip():
                        logger.warning(f"Intent analysis attempt {attempt + 1}: Empty LLM response")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        raise ValueError("Empty LLM response")

                    # Parse JSON with better error handling
                    try:
                        # Strip markdown code fences if present
                        cleaned_response = response.strip()
                        if cleaned_response.startswith("```"):
                            # Remove opening fence (```json or ```)
                            lines = cleaned_response.split('\n')
                            lines = lines[1:]  # Skip first line with ```
                            # Remove closing fence
                            if lines and lines[-1].strip().startswith("```"):
                                lines = lines[:-1]
                            cleaned_response = '\n'.join(lines).strip()

                        data = json.loads(cleaned_response)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Intent analysis attempt {attempt + 1}: Malformed JSON: {response[:100]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        raise

                    # Validate intent value
                    intent = data.get("intent", "").lower()
                    valid_intents = ["coding", "creative", "general"]

                    if intent not in valid_intents:
                        logger.warning(f"Intent analysis attempt {attempt + 1}: Invalid intent '{intent}'")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        raise ValueError(f"Invalid intent: {intent}")

                    logger.info(f"Intent analysis succeeded: '{intent}' (attempt {attempt + 1})")
                    return intent

                except Exception as e:
                    logger.warning(f"Intent analysis attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Intent analysis failed after {max_retries} attempts, falling back to keyword detection")

        # 2. Keyword-based fallback detection
        query_lower = query_text.lower()

        # Coding keywords
        coding_keywords = [
            'function', 'class', 'method', 'variable', 'code', 'debug', 'error',
            'python', 'javascript', 'java', 'rust', 'go', 'c++', 'typescript',
            'api', 'database', 'algorithm', 'bug', 'refactor', 'test', 'compile',
            'import', 'package', 'module', 'library', 'framework'
        ]

        # Creative keywords
        creative_keywords = [
            'story', 'character', 'plot', 'scene', 'chapter', 'narrative',
            'protagonist', 'antagonist', 'dialogue', 'setting', 'theme',
            'novel', 'fiction', 'fantasy', 'sci-fi', 'adventure', 'mystery',
            'write', 'writing', 'creative', 'imagination', 'world-building'
        ]

        # Count keyword matches
        coding_matches = sum(1 for kw in coding_keywords if kw in query_lower)
        creative_matches = sum(1 for kw in creative_keywords if kw in query_lower)

        if coding_matches > creative_matches and coding_matches >= 1:
            logger.info(f"Intent detected via keyword fallback: 'coding' ({coding_matches} matches)")
            return "coding"
        elif creative_matches > coding_matches and creative_matches >= 1:
            logger.info(f"Intent detected via keyword fallback: 'creative' ({creative_matches} matches)")
            return "creative"

        # 3. Default to general
        logger.info("Intent defaulted to 'general' (no LLM or keyword match)")
        return "general"

    async def retrieve_metaphysical_context(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> RAGResult:
        """
        Retrieve context using the Metaphysical Schema strategy.
        1. Intent Analysis (with robust fallback)
        2. Vector Filter Scan (with score threshold)
        3. Graph Expansion
        4. Synthesis
        """
        from config import RAG_SCORE_THRESHOLD

        start_time = time.time()

        logger.info("=" * 60)
        logger.info("RAG RETRIEVAL START")
        logger.info(f"Query: {query_text[:100]}...")
        logger.info("=" * 60)

        # 1. Intent Analysis (Robust with fallback)
        intent_start = time.time()
        intent = await self._analyze_intent_robust(query_text)
        intent_time = (time.time() - intent_start) * 1000

        logger.info(f"Intent Analysis: '{intent}' ({intent_time:.2f}ms)")

        # 2. Vector Filter Scan (Qdrant with score threshold)
        # Construct filter based on intent
        # Strategy: Search specific domain + "general" as fallback
        filter_dict = None
        if intent != "general":
            # Map intent to domain
            domain_map = {"coding": "coding", "creative": "story"}
            domain = domain_map.get(intent, intent)

            # Use OR filter: search both specific domain AND general domain
            filter_dict = self.qdrant_db.create_domain_filter(domain)
            logger.info(f"Applying domain filter: {domain} OR general")
        else:
            logger.info("Using general intent (no domain filter)")

        # Search with score threshold
        search_start = time.time()
        search_results = await self.qdrant_db.search(
            query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
            score_threshold=RAG_SCORE_THRESHOLD  # NEW
        )
        search_time = (time.time() - search_start) * 1000

        # Log search results
        logger.info(f"Qdrant Search: {len(search_results)} results above threshold {RAG_SCORE_THRESHOLD} ({search_time:.2f}ms)")
        if search_results:
            scores = [r.get('score', 0) for r in search_results]
            logger.info(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        else:
            logger.warning("  No results passed score threshold - RAG will be skipped")

        # Extract Node UIDs and Semantic Chunks
        node_uids = []
        semantic_chunks = []

        for idx, result in enumerate(search_results, 1):
            payload = result.get('payload', {})
            score = result.get('score', 0)
            node_id = payload.get('node_id')

            logger.debug(f"  Result {idx}: score={score:.3f}, node_id={node_id}, type={payload.get('type')}")

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
        if node_uids:
            graph_start = time.time()
            expanded_context = await self.neo4j_graph.expand_metaphysical_context(node_uids)
            graph_time = (time.time() - graph_start) * 1000
            logger.info(f"Neo4j Graph Expansion: {len(expanded_context)} nodes expanded ({graph_time:.2f}ms)")
        else:
            expanded_context = []
            logger.info("Neo4j Graph Expansion: Skipped (no node UIDs from Qdrant)")

        # 4. Synthesis (Format for RAGResult)
        relational_facts = []
        for item in expanded_context:
            node = item['node']
            node_str = f"[{node.get('type', 'Node')}: {node.get('name')}] {node.get('description', '')}"
            relational_facts.append(node_str)

            for rel in item['relationships']:
                relational_facts.append(f"  - {rel}")

        retrieval_time = (time.time() - start_time) * 1000

        logger.info("=" * 60)
        logger.info("RAG RETRIEVAL COMPLETE")
        logger.info(f"Total Time: {retrieval_time:.2f}ms")
        logger.info(f"Results: {len(semantic_chunks)} semantic chunks, {len(relational_facts)} relational facts")
        logger.info("=" * 60)

        # Log to metrics logger
        metrics_logger.info(
            f"SEMANTIC_RETRIEVAL | "
            f"intent={intent} | "
            f"query_length={len(query_text)} | "
            f"qdrant_results={len(search_results)} | "
            f"semantic_chunks={len(semantic_chunks)} | "
            f"relational_facts={len(relational_facts)} | "
            f"total_time_ms={retrieval_time:.2f}"
        )

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

            # Run synchronous Redis operations in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def _store_sync():
                # Add to sorted set with timestamp as score
                self.redis_storage.redis.zadd(key, {value: timestamp})

                # Trim to keep only recent responses
                # Keep the most recent N responses (highest scores)
                self.redis_storage.redis.zremrangebyrank(
                    key,
                    0,
                    -(ECHO_RESPONSE_HISTORY_SIZE + 1)
                )

            await loop.run_in_executor(None, _store_sync)

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

            # Run synchronous Redis operation in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def _get_recent_responses():
                # Get recent response embeddings from Redis
                key = "response_embeddings"
                return self.redis_storage.redis.zrange(
                    key,
                    0,
                    -1,
                    withscores=False
                )

            recent_responses = await loop.run_in_executor(None, _get_recent_responses)

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
