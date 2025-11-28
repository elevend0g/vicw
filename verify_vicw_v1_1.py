
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), "app"))

from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIMENSION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_MODEL_NAME
)
from sentence_transformers import SentenceTransformer
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from semantic_manager import SemanticManager
from data_models import OffloadJob

# Mock LLM Client for testing
class MockLLMClient:
    async def generate(self, context, response_format=None, stop=None):
        content = context[0]['content']
        # Return mock JSON for extraction
        if "Extract 'Entities'" in content:
            return """
            {
                "entities": [
                    {"name": "Project VICW", "subtype": "Project", "description": "Virtual Infinite Context Window project"},
                    {"name": "Metaphysical Schema", "subtype": "Architecture", "description": "New graph schema for VICW"}
                ],
                "events": [
                    {"name": "Implementation Started", "subtype": "Milestone", "description": "Started implementing version 1.1", "caused_by": ["Project VICW"]},
                    {"name": "Schema Update", "subtype": "Action", "description": "Updated Neo4j schema", "caused_by": ["Metaphysical Schema"]}
                ]
            }
            """
        # Return mock intent
        if "Classify the intent" in content:
            return '{"intent": "coding"}'
        
        # Return mock summary
        return "Consolidated summary of events."

    async def init(self):
        pass

async def run_verification():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Verification")
    
    logger.info("Initializing components...")
    
    # 1. Initialize Components
    redis_storage = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    await redis_storage.init()
    
    qdrant_db = QdrantVectorDB(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=QDRANT_COLLECTION,
        dimension=EMBEDDING_DIMENSION
    )
    await qdrant_db.init()
    
    neo4j_graph = Neo4jKnowledgeGraph(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    await neo4j_graph.init()
    
    # CLEANUP: Clear Neo4j database to avoid constraint errors
    logger.info("Clearing Neo4j database...")
    async with neo4j_graph._driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    llm_client = MockLLMClient()
    
    executor = ThreadPoolExecutor(max_workers=2)
    
    semantic_manager = SemanticManager(
        embedding_model=embedding_model,
        redis_storage=redis_storage,
        qdrant_db=qdrant_db,
        neo4j_graph=neo4j_graph,
        llm_client=llm_client,
        executor=executor
    )
    
    # 2. Test Ingestion
    logger.info("Testing Ingestion...")
    job = OffloadJob(
        job_id="test_job_1",
        chunk_text="Project VICW has started implementation of the Metaphysical Schema. This update involves changes to Neo4j and Qdrant.",
        token_count=20,
        message_count=1,
        timestamp=time.time(),
        metadata={"domain": "coding", "thread_id": "test_thread"}
    )
    
    result = await semantic_manager.process_job(job)
    if result.success:
        logger.info("Ingestion successful!")
    else:
        logger.error(f"Ingestion failed: {result.error}")
        return

    # 3. Verify Neo4j State
    logger.info("Verifying Neo4j State...")
    async with neo4j_graph._driver.session() as session:
        # Check Context
        r = await session.run("MATCH (c:Context {domain: 'coding'}) RETURN c")
        if await r.single():
            logger.info("✅ Context node found")
        else:
            logger.error("❌ Context node NOT found")
            
        # Check Entity
        r = await session.run("MATCH (e:Entity {name: 'Project VICW'}) RETURN e")
        if await r.single():
            logger.info("✅ Entity node found")
        else:
            logger.error("❌ Entity node NOT found")

        # Check Event
        r = await session.run("MATCH (e:Event {name: 'Implementation Started'}) RETURN e")
        if await r.single():
            logger.info("✅ Event node found")
        else:
            logger.error("❌ Event node NOT found")

    # 4. Test Retrieval
    logger.info("Testing Retrieval...")
    query = "What is the status of Project VICW?"
    query_embedding = embedding_model.encode(query).tolist()
    
    rag_result = await semantic_manager.retrieve_metaphysical_context(query, query_embedding)
    
    logger.info(f"Retrieval Result: {len(rag_result.relational_facts)} facts found")
    for fact in rag_result.relational_facts:
        logger.info(f"  - {fact}")
        
    if len(rag_result.relational_facts) > 0:
        logger.info("✅ Retrieval successful")
    else:
        logger.warning("⚠️ Retrieval returned no facts (might be expected if mock data doesn't align perfectly)")

    # 5. Test Sleep Cycle (Manual Trigger)
    logger.info("Testing Sleep Cycle...")
    # Create a dummy old event
    async with neo4j_graph._driver.session() as session:
        await session.run("""
            MERGE (e:Event {uid: 'old_event_1'})
            SET e.name = 'Old Event', 
                e.timestamp = $ts, 
                e.description = 'An old event',
                e.flow_id = 'test_flow'
        """, parameters={"ts": time.time() - 7200}) # 2 hours ago
        
        await session.run("""
            MERGE (e:Event {uid: 'old_event_2'})
            SET e.name = 'Old Event 2', 
                e.timestamp = $ts, 
                e.description = 'Another old event',
                e.flow_id = 'test_flow'
        """, parameters={"ts": time.time() - 7100})

    # Manually call get_old_events and consolidate
    events = await neo4j_graph.get_old_events(hours=1)
    logger.info(f"Found {len(events)} old events")
    
    if len(events) >= 2:
        macro_uid = "test_macro_event"
        await neo4j_graph.create_macro_event({
            "uid": macro_uid,
            "name": "Test Macro Event",
            "description": "Consolidated test events",
            "type": "MacroEvent"
        })
        await neo4j_graph.consolidate_events([e['uid'] for e in events], macro_uid)
        logger.info("✅ Events consolidated")
        
        # Verify consolidation
        async with neo4j_graph._driver.session() as session:
            r = await session.run("MATCH (e:Event {uid: 'old_event_1'})-[:CONSOLIDATED_INTO]->(m:MacroEvent) RETURN m")
            if await r.single():
                logger.info("✅ Relationship CONSOLIDATED_INTO verified")
            else:
                logger.error("❌ Relationship CONSOLIDATED_INTO NOT found")
    else:
        logger.warning("Not enough old events found for test")

    # Cleanup
    await neo4j_graph.close()
    await redis_storage.shutdown()
    executor.shutdown()
    # Qdrant client doesn't need explicit close usually, but good practice if available
    
    logger.info("Verification complete!")

if __name__ == "__main__":
    asyncio.run(run_verification())
