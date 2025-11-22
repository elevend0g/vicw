"""Neo4j-based knowledge graph for relational tracking"""

import logging
import asyncio
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraph:
    """Manages relational data (entities, goals, relationships) using Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: AsyncDriver = None
        logger.info(f"Neo4j configured for {uri}")
    
    async def init(self):
        """Initialize Neo4j driver and constraints"""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                connection_timeout=10
            )
            
            # Verify connectivity
            await self._driver.verify_connectivity()
            
            # Initialize constraints
            await self.initialize_constraints()
            
            logger.info("Neo4j driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise
    
    async def initialize_constraints(self):
        """Create uniqueness constraints and indexes"""
        async with self._driver.session() as session:
            try:
                # Entity uniqueness
                await session.run(
                    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
                
                # Chunk uniqueness
                await session.run(
                    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                    "FOR (c:Chunk) REQUIRE c.job_id IS UNIQUE"
                )
                
                # Create indexes for performance
                await session.run(
                    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )
                
                await session.run(
                    "CREATE INDEX chunk_timestamp_idx IF NOT EXISTS FOR (c:Chunk) ON (c.processed_at)"
                )
                
                logger.info("Neo4j constraints and indexes initialized")
            except Exception as e:
                logger.warning(f"Error creating constraints (may already exist): {e}")
    
    async def update_chunk_node(self, job_id: str, summary: str, token_count: int) -> None:
        """
        COLD PATH: Create/update a Chunk node with metadata.
        In a production system, you would parse the summary to extract entities and relationships.
        """
        cql_query = """
        MERGE (c:Chunk {job_id: $job_id})
        SET c.summary = $summary,
            c.token_count = $token_count,
            c.processed_at = timestamp()
        RETURN c
        """
        
        async with self._driver.session() as session:
            try:
                await session.run(
                    cql_query,
                    job_id=job_id,
                    summary=summary,
                    token_count=token_count
                )
                logger.debug(f"Updated Neo4j chunk node for job_id={job_id}")
            except Exception as e:
                logger.error(f"Error updating Neo4j chunk {job_id}: {e}")
    
    async def update_graph_from_context(self, job_id: str, summary: str) -> None:
        """
        COLD PATH: Extract entities and relationships from summary and update graph.
        This is a simplified implementation. In production, you would use NLP or LLM
        to extract structured information.
        """
        async with self._driver.session() as session:
            try:
                # For now, just create a chunk node
                # In production, you would:
                # 1. Parse summary for entities (using spaCy or LLM)
                # 2. Extract relationships
                # 3. Create entity nodes and relationship edges
                
                await session.run("""
                    MERGE (c:Chunk {job_id: $job_id})
                    SET c.summary = $summary,
                        c.created_at = timestamp()
                """, job_id=job_id, summary=summary)
                
                logger.debug(f"Updated knowledge graph for job_id={job_id}")
                
            except Exception as e:
                logger.error(f"Error updating knowledge graph for {job_id}: {e}")
    
    async def create_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None) -> None:
        """Create or update an entity node"""
        async with self._driver.session() as session:
            try:
                props = properties or {}
                props['name'] = name
                props['type'] = entity_type
                props['updated_at'] = "timestamp()"
                
                # Build property string
                prop_str = ", ".join([f"e.{k} = ${k}" for k in props.keys()])
                
                cql = f"MERGE (e:Entity {{name: $name}}) SET {prop_str}"
                
                await session.run(cql, **props)
                logger.debug(f"Created/updated entity: {name} ({entity_type})")
                
            except Exception as e:
                logger.error(f"Error creating entity {name}: {e}")
    
    async def create_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> None:
        """Create a relationship between two entities"""
        async with self._driver.session() as session:
            try:
                props = properties or {}
                props['created_at'] = "timestamp()"
                
                cql = f"""
                MATCH (a:Entity {{name: $from_entity}})
                MATCH (b:Entity {{name: $to_entity}})
                MERGE (a)-[r:{relationship_type}]->(b)
                SET r += $props
                """
                
                await session.run(
                    cql,
                    from_entity=from_entity,
                    to_entity=to_entity,
                    props=props
                )
                logger.debug(f"Created relationship: ({from_entity})-[:{relationship_type}]->({to_entity})")
                
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
    
    async def relational_query(self, query_text: str, limit: int = 5) -> List[str]:
        """
        HOT PATH: Query the graph for relevant relationships.
        Returns formatted relationship strings for context injection.
        """
        # Simple query that searches for entities and their relationships
        cql_query = """
        MATCH (n)-[r]->(m)
        WHERE n.name CONTAINS $query 
           OR m.name CONTAINS $query 
           OR n.summary CONTAINS $query
           OR m.summary CONTAINS $query
        RETURN n, type(r) AS relationship, m
        LIMIT $limit
        """
        
        async with self._driver.session() as session:
            try:
                result = await session.run(cql_query, query=query_text, limit=limit)
                
                structured_data = []
                async for record in result:
                    # Extract node names
                    n = record['n']
                    m = record['m']
                    r_type = record['relationship']
                    
                    # Get node names or labels
                    n_name = n.get('name', list(n.labels)[0] if n.labels else 'Node')
                    m_name = m.get('name', list(m.labels)[0] if m.labels else 'Node')
                    
                    # Format as relationship triple
                    structured_data.append(f"({n_name})-[:{r_type}]->({m_name})")
                
                logger.debug(f"Relational query returned {len(structured_data)} facts")
                return structured_data
                
            except Exception as e:
                logger.error(f"Error in relational query: {e}")
                return []
    
    async def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get all relationships and properties for an entity"""
        cql_query = """
        MATCH (e:Entity {name: $entity_name})
        OPTIONAL MATCH (e)-[r]->(other)
        RETURN e, collect({rel: type(r), target: other.name}) AS relationships
        """
        
        async with self._driver.session() as session:
            try:
                result = await session.run(cql_query, entity_name=entity_name)
                record = await result.single()
                
                if record:
                    return {
                        "entity": dict(record['e']),
                        "relationships": record['relationships']
                    }
                return {}
                
            except Exception as e:
                logger.error(f"Error getting entity context: {e}")
                return {}
    
    async def clear_graph(self) -> None:
        """Clear all nodes and relationships (use with caution!)"""
        async with self._driver.session() as session:
            try:
                await session.run("MATCH (n) DETACH DELETE n")
                logger.warning("Cleared all Neo4j graph data")
            except Exception as e:
                logger.error(f"Error clearing graph: {e}")
    
    async def close(self):
        """Close the Neo4j driver connection"""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j driver closed")
