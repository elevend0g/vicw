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

                # State uniqueness
                await session.run(
                    "CREATE CONSTRAINT state_id_unique IF NOT EXISTS "
                    "FOR (s:State) REQUIRE s.id IS UNIQUE"
                )

                # Create indexes for performance
                await session.run(
                    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )

                await session.run(
                    "CREATE INDEX state_type_status_idx IF NOT EXISTS FOR (s:State) ON (s.type, s.status)"
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
                # FIXED: Passed parameters as a dictionary
                await session.run(
                    cql_query,
                    parameters={
                        "job_id": job_id,
                        "summary": summary,
                        "token_count": token_count
                    }
                )
                logger.debug(f"Updated Neo4j chunk node for job_id={job_id}")
            except Exception as e:
                logger.error(f"Error updating Neo4j chunk {job_id}: {e}")
    
    async def update_graph_from_context(self, job_id: str, summary: str) -> None:
        """
        COLD PATH: Extract entities and relationships from summary and update graph.
        """
        async with self._driver.session() as session:
            try:
                # Create/update the Chunk node
                await session.run("""
                    MERGE (c:Chunk {job_id: $job_id})
                    SET c.summary = $summary,
                        c.created_at = timestamp()
                """, parameters={"job_id": job_id, "summary": summary})

                # Extract entities using simple rule-based approach
                entities = self._extract_entities(summary)

                # Create Entity nodes and relationships to the Chunk
                for entity_name, entity_type in entities:
                    # Create/update entity node
                    await session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.updated_at = timestamp()
                    """, parameters={"name": entity_name, "type": entity_type})

                    # Create relationship from Chunk to Entity
                    await session.run("""
                        MATCH (c:Chunk {job_id: $job_id})
                        MATCH (e:Entity {name: $name})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, parameters={"job_id": job_id, "name": entity_name})

                # Create relationships between entities mentioned together
                if len(entities) > 1:
                    for i in range(len(entities) - 1):
                        for j in range(i + 1, len(entities)):
                            await session.run("""
                                MATCH (e1:Entity {name: $name1})
                                MATCH (e2:Entity {name: $name2})
                                MERGE (e1)-[:RELATED_TO]-(e2)
                            """, parameters={"name1": entities[i][0], "name2": entities[j][0]})

                if entities:
                    logger.debug(f"Updated knowledge graph for job_id={job_id} with {len(entities)} entities")
                else:
                    logger.debug(f"Updated knowledge graph for job_id={job_id} (no entities extracted)")

            except Exception as e:
                logger.error(f"Error updating knowledge graph for {job_id}: {e}")

    def _extract_entities(self, text: str) -> List[tuple]:
        """
        Simple rule-based entity extraction.
        Returns list of (entity_name, entity_type) tuples.
        """
        import re

        entities = []

        # Extract capitalized phrases (2-4 words) as potential entities
        # Matches patterns like "John Smith", "New York City", etc.
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
        matches = re.findall(pattern, text)

        # Deduplicate and filter common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'What', 'Why', 'How', 'Who'}
        seen = set()

        for match in matches:
            # Skip if starts with common word or already seen
            if match.split()[0] in common_words or match in seen:
                continue

            seen.add(match)

            # Simple type inference based on context
            entity_type = "UNKNOWN"
            if any(word in text.lower() for word in ['goal', 'objective', 'aim']):
                entity_type = "GOAL"
            elif any(word in text.lower() for word in ['task', 'action', 'do', 'implement']):
                entity_type = "TASK"
            elif any(word in text.lower() for word in ['fact', 'is', 'are', 'was', 'were']):
                entity_type = "FACT"

            entities.append((match, entity_type))

        # Limit to top 10 entities per chunk to avoid overload
        return entities[:10]
    
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
                
                # FIXED: props is already a dictionary, passed to parameters
                await session.run(cql, parameters=props)
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
                
                # FIXED: Explicit parameters dictionary
                await session.run(
                    cql,
                    parameters={
                        "from_entity": from_entity,
                        "to_entity": to_entity,
                        "props": props
                    }
                )
                logger.debug(f"Created relationship: ({from_entity})-[:{relationship_type}]->({to_entity})")
                
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
    
    async def relational_query(self, query_text: str, limit: int = 5) -> List[str]:
        """
        HOT PATH: Query the graph for relevant relationships.
        Returns formatted relationship strings for context injection.
        """
        # Extract potential keywords from query (capitalized words)
        import re
        keywords = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query_text)

        # Also include significant lowercase words
        significant_words = [word for word in query_text.split()
                           if len(word) > 3 and word.lower() not in
                           {'about', 'tell', 'what', 'when', 'where', 'how', 'why', 'who', 'the', 'this', 'that'}]

        # Combine keywords and significant words
        search_terms = keywords + significant_words

        if not search_terms:
            # Fallback to using the whole query if no keywords extracted
            search_terms = [query_text]

        async with self._driver.session() as session:
            try:
                all_results = []

                # Search for each term
                for term in search_terms[:3]:  # Limit to first 3 terms to avoid too many queries
                    cql_query = """
                    MATCH (n)-[r]->(m)
                    WHERE toLower(coalesce(n.name, '')) CONTAINS toLower($query)
                       OR toLower(coalesce(m.name, '')) CONTAINS toLower($query)
                       OR toLower(coalesce(n.summary, '')) CONTAINS toLower($query)
                       OR toLower(coalesce(m.summary, '')) CONTAINS toLower($query)
                    RETURN n, type(r) AS relationship, m
                    LIMIT $limit
                    """

                    result = await session.run(
                        cql_query,
                        parameters={"query": term, "limit": limit}
                    )

                    async for record in result:
                        # Extract node names
                        n = record['n']
                        m = record['m']
                        r_type = record['relationship']

                        # Get node names or labels
                        n_name = n.get('name', n.get('summary', list(n.labels)[0] if n.labels else 'Node'))
                        m_name = m.get('name', m.get('summary', list(m.labels)[0] if m.labels else 'Node'))

                        # Truncate long summaries
                        if len(n_name) > 50:
                            n_name = n_name[:47] + "..."
                        if len(m_name) > 50:
                            m_name = m_name[:47] + "..."

                        # Format as relationship triple
                        fact = f"({n_name})-[:{r_type}]->({m_name})"
                        if fact not in all_results:  # Avoid duplicates
                            all_results.append(fact)

                # Limit total results
                structured_data = all_results[:limit]

                logger.info(f"Relational query for '{query_text}' (terms: {search_terms[:3]}) returned {len(structured_data)} facts")
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
                # FIXED: Explicit parameters
                result = await session.run(cql_query, parameters={"entity_name": entity_name})
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
    
    async def create_state(self, state_type: str, description: str, status: str = "active") -> str:
        """
        COLD PATH: Create a new state node.
        Returns the state_id.
        """
        from data_models import State

        state = State.create(state_type, description, status)

        async with self._driver.session() as session:
            try:
                await session.run("""
                    CREATE (s:State {
                        id: $id,
                        type: $type,
                        desc: $desc,
                        status: $status,
                        created: $created,
                        updated: $updated
                    })
                """, parameters={
                    "id": state.id,
                    "type": state.type,
                    "desc": state.desc,
                    "status": state.status,
                    "created": int(state.created),
                    "updated": int(state.updated)
                })
                logger.debug(f"Created state: {state_type}/{description} ({status})")
                return state.id
            except Exception as e:
                logger.error(f"Error creating state: {e}")
                return None

    async def find_similar_state(self, state_type: str, description: str) -> Dict[str, Any]:
        """
        COLD PATH: Find existing state by type and fuzzy match on description.
        Returns state dict or None.
        """
        # Simple fuzzy matching: lowercase and check if descriptions are very similar
        desc_lower = description.lower().strip()

        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (s:State {type: $type})
                    WHERE toLower(s.desc) CONTAINS $desc_part
                       OR $desc_part CONTAINS toLower(s.desc)
                    RETURN s
                    LIMIT 1
                """, parameters={
                    "type": state_type,
                    "desc_part": desc_lower[:30]  # Use first 30 chars for matching
                })

                record = await result.single()
                if record:
                    state_node = record['s']
                    return {
                        "id": state_node.get("id"),
                        "type": state_node.get("type"),
                        "desc": state_node.get("desc"),
                        "status": state_node.get("status"),
                        "created": state_node.get("created"),
                        "updated": state_node.get("updated")
                    }
                return None
            except Exception as e:
                logger.error(f"Error finding similar state: {e}")
                return None

    async def update_state_status(self, state_id: str, new_status: str) -> bool:
        """
        COLD PATH: Update the status of an existing state.
        Returns True if successful.
        """
        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (s:State {id: $state_id})
                    SET s.status = $new_status,
                        s.updated = $updated
                    RETURN s
                """, parameters={
                    "state_id": state_id,
                    "new_status": new_status,
                    "updated": int(asyncio.get_event_loop().time())
                })

                record = await result.single()
                if record:
                    logger.debug(f"Updated state {state_id} to {new_status}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error updating state status: {e}")
                return False

    async def get_active_states(self, state_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        HOT PATH: Get active states (status='active').
        If state_type is provided, filter by type.
        """
        async with self._driver.session() as session:
            try:
                if state_type:
                    query = """
                        MATCH (s:State {type: $type, status: 'active'})
                        RETURN s
                        ORDER BY s.created DESC
                        LIMIT $limit
                    """
                    params = {"type": state_type, "limit": limit}
                else:
                    query = """
                        MATCH (s:State {status: 'active'})
                        RETURN s
                        ORDER BY s.created DESC
                        LIMIT $limit
                    """
                    params = {"limit": limit}

                result = await session.run(query, parameters=params)

                states = []
                async for record in result:
                    state_node = record['s']
                    states.append({
                        "id": state_node.get("id"),
                        "type": state_node.get("type"),
                        "desc": state_node.get("desc"),
                        "status": state_node.get("status"),
                        "created": state_node.get("created"),
                        "updated": state_node.get("updated")
                    })

                logger.debug(f"Retrieved {len(states)} active states (type={state_type})")
                return states
            except Exception as e:
                logger.error(f"Error getting active states: {e}")
                return []

    async def get_completed_states(self, state_type: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        HOT PATH: Get completed states (status='completed').
        If state_type is provided, filter by type.
        """
        async with self._driver.session() as session:
            try:
                if state_type:
                    query = """
                        MATCH (s:State {type: $type, status: 'completed'})
                        RETURN s
                        ORDER BY s.updated DESC
                        LIMIT $limit
                    """
                    params = {"type": state_type, "limit": limit}
                else:
                    query = """
                        MATCH (s:State {status: 'completed'})
                        RETURN s
                        ORDER BY s.updated DESC
                        LIMIT $limit
                    """
                    params = {"limit": limit}

                result = await session.run(query, parameters=params)

                states = []
                async for record in result:
                    state_node = record['s']
                    states.append({
                        "id": state_node.get("id"),
                        "type": state_node.get("type"),
                        "desc": state_node.get("desc"),
                        "status": state_node.get("status"),
                        "created": state_node.get("created"),
                        "updated": state_node.get("updated")
                    })

                logger.debug(f"Retrieved {len(states)} completed states (type={state_type})")
                return states
            except Exception as e:
                logger.error(f"Error getting completed states: {e}")
                return []

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