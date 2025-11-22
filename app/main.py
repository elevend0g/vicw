"""Main orchestrator for VICW Phase 2 - CLI mode"""

import os
import logging
import asyncio
from pathlib import Path

# Set thread limits BEFORE imports
from config import (
    apply_thread_config,
    MAX_CONTEXT_TOKENS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EXTERNAL_API_URL,
    EXTERNAL_API_KEY,
    EXTERNAL_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    LOG_LEVEL
)

apply_thread_config()

# Now import heavy libraries
from sentence_transformers import SentenceTransformer

from context_manager import ContextManager
from offload_queue import OffloadQueue
from cold_path_worker import ColdPathWorker
from semantic_manager import SemanticManager
from llm_inference import ExternalLLMInference
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics logging
metrics_logger = logging.getLogger('vicw.metrics')
metrics_handler = logging.FileHandler('vicw_metrics.log')
metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)


async def main():
    """Main async orchestrator for CLI mode"""
    
    print("=" * 60)
    print("VICW Phase 2 - Virtual Infinite Context Window")
    print("Type 'exit' to quit, 'stats' for statistics")
    print("=" * 60)
    print()
    
    logger.info("=" * 60)
    logger.info("VICW SESSION STARTED (PHASE 2)")
    logger.info("=" * 60)
    
    # Initialize components
    try:
        # Initialize Redis storage
        logger.info("Initializing Redis storage...")
        redis_storage = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        await redis_storage.init()
        
        # Initialize Qdrant vector database
        logger.info("Initializing Qdrant vector database...")
        qdrant_db = QdrantVectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION,
            dimension=EMBEDDING_DIMENSION
        )
        await qdrant_db.init()
        
        # Initialize Neo4j knowledge graph
        logger.info("Initializing Neo4j knowledge graph...")
        neo4j_graph = Neo4jKnowledgeGraph(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        await neo4j_graph.init()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize offload queue
        offload_queue = OffloadQueue()
        
        # Initialize semantic manager
        logger.info("Initializing semantic manager...")
        semantic_manager = SemanticManager(
            embedding_model=embedding_model,
            redis_storage=redis_storage,
            qdrant_db=qdrant_db,
            neo4j_graph=neo4j_graph
        )
        
        # Initialize context manager
        logger.info("Initializing context manager...")
        context_manager = ContextManager(
            max_context=MAX_CONTEXT_TOKENS,
            offload_queue=offload_queue,
            embedding_model=embedding_model,
            semantic_manager=semantic_manager
        )
        
        # Initialize external LLM
        logger.info(f"Initializing external LLM: {EXTERNAL_MODEL_NAME}...")
        if not EXTERNAL_API_KEY:
            raise ValueError("VICW_LLM_API_KEY environment variable must be set")
        
        llm = ExternalLLMInference(
            api_url=EXTERNAL_API_URL,
            api_key=EXTERNAL_API_KEY,
            model_name=EXTERNAL_MODEL_NAME
        )
        await llm.init()
        
        # Initialize and start cold path worker
        logger.info("Starting cold path worker...")
        cold_path_worker = ColdPathWorker(offload_queue, semantic_manager)
        await cold_path_worker.start()
        
        # Load system prompt if available
        system_prompt_path = Path("system_prompt.txt")
        if system_prompt_path.exists():
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read().strip()
            await context_manager.add_message("system", system_prompt)
            logger.info("System prompt loaded and added to context")
        else:
            logger.warning(f"System prompt not found at {system_prompt_path}")
        
        logger.info("=" * 60)
        logger.info("VICW Phase 2 system ready")
        logger.info(f"LLM: {EXTERNAL_MODEL_NAME}")
        logger.info(f"Max context: {MAX_CONTEXT_TOKENS} tokens")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize VICW system: {e}")
        print(f"\nERROR: Failed to initialize system: {e}")
        return
    
    # Main conversation loop
    turn_count = 0
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                # Show statistics
                stats = context_manager.get_stats()
                queue_stats = offload_queue.get_stats()
                worker_stats = cold_path_worker.get_stats()
                
                print("\n" + "=" * 60)
                print("SYSTEM STATISTICS")
                print("=" * 60)
                print(f"Context: {stats['current_tokens']}/{stats['max_tokens']} tokens ({stats['pressure_percentage']:.1f}%)")
                print(f"Messages in context: {stats['message_count']}")
                print(f"Offload jobs: {stats['offload_count']}")
                print(f"Queue: {queue_stats['current_size']}/{queue_stats['max_size']}")
                print(f"Processed: {queue_stats['processed_total']}, Dropped: {queue_stats['dropped_total']}")
                print(f"Worker: Processed={worker_stats['processed_count']}, Failed={worker_stats['failed_count']}")
                print("=" * 60)
                continue
            
            if not user_input:
                continue
            
            turn_count += 1
            turn_start_time = asyncio.get_event_loop().time()
            
            logger.info(f"--- Turn {turn_count} START ---")
            metrics_logger.info(
                f"TURN_START | turn={turn_count} | queue_size={await offload_queue.get_queue_size()}"
            )
            
            # Add user message (triggers async offload if needed)
            await context_manager.add_message("user", user_input)
            
            # Perform RAG
            rag_items = await context_manager.augment_context_with_memory(user_input)
            
            if rag_items > 0:
                print(f"[Retrieved {rag_items} items from long-term memory]")
            
            # Pause cold path during generation
            await cold_path_worker.pause()
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            try:
                response = await llm.generate(context_manager.get_context_window())
                print(response)
            except Exception as e:
                print(f"ERROR: {e}")
                logger.error(f"Generation error: {e}")
                await cold_path_worker.resume()
                continue
            
            # Resume cold path
            await cold_path_worker.resume()
            
            # Add assistant response
            await context_manager.add_message("assistant", response)
            
            turn_time = (asyncio.get_event_loop().time() - turn_start_time) * 1000
            queue_size = await offload_queue.get_queue_size()
            
            logger.info(f"--- Turn {turn_count} END ({turn_time:.2f}ms, queue_size={queue_size}) ---")
            metrics_logger.info(
                f"TURN_END | "
                f"turn={turn_count} | "
                f"total_time_ms={turn_time:.2f} | "
                f"queue_size={queue_size}"
            )
    
    finally:
        # Cleanup
        logger.info("=" * 60)
        logger.info(f"VICW SESSION ENDED - Total turns: {turn_count}")
        logger.info(f"Offload jobs processed: {cold_path_worker.processed_count}")
        logger.info("=" * 60)
        
        print("\nShutting down...")
        
        await cold_path_worker.shutdown()
        await llm.shutdown()
        await redis_storage.shutdown()
        await qdrant_db.shutdown()
        await neo4j_graph.close()
        
        print("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFATAL ERROR: {e}")
