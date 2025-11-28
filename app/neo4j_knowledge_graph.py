"""Neo4j-based knowledge graph for relational tracking"""

import logging
import asyncio
import time
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
                # --- Metaphysical Schema Constraints ---
                
                # Context uniqueness
                await session.run(
                    "CREATE CONSTRAINT context_uid_unique IF NOT EXISTS "
                    "FOR (c:Context) REQUIRE c.uid IS UNIQUE"
                )
                
                # Entity uniqueness
                await session.run(
                    "CREATE CONSTRAINT entity_uid_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.uid IS UNIQUE"
                )
                
                # Event uniqueness
                await session.run(
                    "CREATE CONSTRAINT event_uid_unique IF NOT EXISTS "
                    "FOR (e:Event) REQUIRE e.uid IS UNIQUE"
                )
                
                # Concept uniqueness
                await session.run(
                    "CREATE CONSTRAINT concept_uid_unique IF NOT EXISTS "
                    "FOR (c:Concept) REQUIRE c.uid IS UNIQUE"
                )
                
                # Chunk uniqueness
                await session.run(
                    "CREATE CONSTRAINT chunk_uid_unique IF NOT EXISTS "
                    "FOR (c:Chunk) REQUIRE c.uid IS UNIQUE"
                )

                # Legacy constraints (keeping for safety, though we might migrate away)
                await session.run(
                    "CREATE CONSTRAINT state_id_unique IF NOT EXISTS "
                    "FOR (s:State) REQUIRE s.id IS UNIQUE"
                )

                # --- Indexes for Performance ---
                
                # Name lookups
                await session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                await session.run("CREATE INDEX event_name_idx IF NOT EXISTS FOR (e:Event) ON (e.name)")
                await session.run("CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)")
                
                # Domain filtering
                await session.run("CREATE INDEX entity_domain_idx IF NOT EXISTS FOR (e:Entity) ON (e.domain)")
                await session.run("CREATE INDEX event_domain_idx IF NOT EXISTS FOR (e:Event) ON (e.domain)")
                
                # Time-based lookups
                await session.run("CREATE INDEX event_timestamp_idx IF NOT EXISTS FOR (e:Event) ON (e.timestamp)")
                
                # Flow lookups
                await session.run("CREATE INDEX event_flow_idx IF NOT EXISTS FOR (e:Event) ON (e.flow_id, e.flow_step)")

                logger.info("Neo4j constraints and indexes initialized")
            except Exception as e:
                logger.warning(f"Error creating constraints (may already exist): {e}")

    # --- Metaphysical Node Creation Methods ---

    async def create_context_node(self, node_dict: Dict[str, Any]) -> None:
        """Create a Context node"""
        query = """
        MERGE (c:Context {uid: $uid})
        SET c += $props
        """
        # Exclude uid from props since it's used in MERGE
        props = {k: v for k, v in node_dict.items() if k != 'uid'}
        
        async with self._driver.session() as session:
            await session.run(query, parameters={"uid": node_dict['uid'], "props": props})
            logger.debug(f"Created Context node: {node_dict.get('name')}")

    async def create_entity_node(self, node_dict: Dict[str, Any]) -> None:
        """Create an Entity node"""
        query = """
        MERGE (e:Entity {uid: $uid})
        SET e += $props
        """
        props = {k: v for k, v in node_dict.items() if k != 'uid'}
        
        async with self._driver.session() as session:
            await session.run(query, parameters={"uid": node_dict['uid'], "props": props})
            logger.debug(f"Created Entity node: {node_dict.get('name')}")

    async def create_event_node(self, node_dict: Dict[str, Any]) -> None:
        """Create an Event node"""
        query = """
        MERGE (e:Event {uid: $uid})
        SET e += $props
        """
        props = {k: v for k, v in node_dict.items() if k != 'uid'}
        
        async with self._driver.session() as session:
            await session.run(query, parameters={"uid": node_dict['uid'], "props": props})
            logger.debug(f"Created Event node: {node_dict.get('name')}")

    async def create_concept_node(self, node_dict: Dict[str, Any]) -> None:
        """Create a Concept node"""
        query = """
        MERGE (c:Concept {uid: $uid})
        SET c += $props
        """
        props = {k: v for k, v in node_dict.items() if k != 'uid'}
        
        async with self._driver.session() as session:
            await session.run(query, parameters={"uid": node_dict['uid'], "props": props})
            logger.debug(f"Created Concept node: {node_dict.get('name')}")

    async def create_chunk_node(self, node_dict: Dict[str, Any]) -> None:
        """Create a Chunk node"""
        query = """
        MERGE (c:Chunk {uid: $uid})
        SET c += $props
        """
        props = {k: v for k, v in node_dict.items() if k != 'uid'}
        
        async with self._driver.session() as session:
            await session.run(query, parameters={"uid": node_dict['uid'], "props": props})
            logger.debug(f"Created Chunk node: {node_dict.get('uid')}")

    # --- Metaphysical Relationship Methods ---

    async def create_metaphysical_relationship(
        self, 
        start_uid: str, 
        start_label: str, 
        end_uid: str, 
        end_label: str, 
        rel_type: str, 
        props: Dict[str, Any] = None
    ) -> None:
        """Generic method to create relationships between metaphysical nodes"""
        # Sanitize labels to prevent injection (though internal use only)
        valid_labels = {"Context", "Entity", "Event", "Concept", "Chunk"}
        if start_label not in valid_labels or end_label not in valid_labels:
            raise ValueError(f"Invalid labels: {start_label}, {end_label}")
        
        query = f"""
        MATCH (a:{start_label} {{uid: $start_uid}})
        MATCH (b:{end_label} {{uid: $end_uid}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        
        async with self._driver.session() as session:
            await session.run(
                query, 
                parameters={
                    "start_uid": start_uid, 
                    "end_uid": end_uid, 
                    "props": props or {}
                }
            )
            logger.debug(f"Created relationship: ({start_label})-[{rel_type}]->({end_label})")
    
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
    
    async def expand_metaphysical_context(self, node_uids: List[str]) -> List[Dict[str, Any]]:
        """
        Expand context from a list of starting node UIDs.
        Traverses:
        - 1 Hop outgoing via [:CAUSED] (Consequences)
        - 1 Hop incoming via [:INITIATED] (Agents/Causes)
        - 1 Hop linear via [:NEXT] (Sequence)
        """
        if not node_uids:
            return []

        query = """
        MATCH (start)
        WHERE start.uid IN $uids
        
        // 1. Direct Consequences
        OPTIONAL MATCH (start)-[r1:CAUSED]->(consequence)
        
        // 2. Agents/Causes
        OPTIONAL MATCH (agent)-[r2:INITIATED]->(start)
        
        // 3. Next Steps
        OPTIONAL MATCH (start)-[r3:NEXT]->(next_step)
        
        RETURN 
            start,
            collect(DISTINCT {rel: "CAUSED", target: consequence}) as consequences,
            collect(DISTINCT {rel: "INITIATED_BY", target: agent}) as agents,
            collect(DISTINCT {rel: "NEXT", target: next_step}) as next_steps
        """
        
        async with self._driver.session() as session:
            try:
                result = await session.run(query, parameters={"uids": node_uids})
                
                expanded_context = []
                async for record in result:
                    start_node = dict(record['start'])
                    
                    # Process relationships
                    relationships = []
                    
                    for item in record['consequences']:
                        if item['target']:
                            relationships.append(f"CAUSED -> {item['target'].get('name')} ({item['target'].get('subtype')})")
                            
                    for item in record['agents']:
                        if item['target']:
                            relationships.append(f"INITIATED BY <- {item['target'].get('name')} ({item['target'].get('subtype')})")
                            
                    for item in record['next_steps']:
                        if item['target']:
                            relationships.append(f"NEXT -> {item['target'].get('name')} ({item['target'].get('subtype')})")
                    
                    expanded_context.append({
                        "node": start_node,
                        "relationships": relationships
                    })
                    
                return expanded_context
                
            except Exception as e:
                logger.error(f"Error expanding metaphysical context: {e}")
                return []

    async def get_old_events(self, hours: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find Event nodes older than 'hours' that are not yet consolidated.
        """
        # Calculate cutoff timestamp (assuming timestamp is unix epoch or ISO string)
        # If timestamp is ISO string, we can use datetime.
        # But here we assume timestamp is stored as float/int or string.
        # Let's assume we use the 'created_at' or 'timestamp' property.
        
        query = """
        MATCH (e:Event)
        WHERE e.timestamp < $cutoff_timestamp
          AND NOT (e)-[:CONSOLIDATED_INTO]->(:MacroEvent)
        RETURN e
        ORDER BY e.timestamp ASC
        LIMIT $limit
        """
        
        # Calculate cutoff. Assuming timestamp is float (seconds since epoch)
        cutoff = time.time() - (hours * 3600)
        
        async with self._driver.session() as session:
            try:
                result = await session.run(query, parameters={"cutoff_timestamp": cutoff, "limit": limit})
                events = []
                async for record in result:
                    events.append(dict(record['e']))
                return events
            except Exception as e:
                logger.error(f"Error getting old events: {e}")
                return []

    async def create_macro_event(self, macro_event_data: Dict[str, Any]) -> str:
        """
        Create a MacroEvent node.
        """
        query = """
        MERGE (m:MacroEvent {uid: $uid})
        SET m += $props, m.created_at = timestamp()
        RETURN m.uid
        """
        
        uid = macro_event_data.get("uid")
        if not uid:
            uid = str(uuid.uuid4())
            macro_event_data["uid"] = uid
            
        async with self._driver.session() as session:
            try:
                await session.run(query, parameters={"uid": uid, "props": macro_event_data})
                logger.info(f"Created MacroEvent {uid}")
                return uid
            except Exception as e:
                logger.error(f"Error creating MacroEvent: {e}")
                return None

    async def consolidate_events(self, event_uids: List[str], macro_event_uid: str):
        """
        Link events to MacroEvent and mark them as consolidated.
        Optionally delete them if pruning is enabled (here we just link).
        """
        query = """
        MATCH (m:MacroEvent {uid: $macro_uid})
        MATCH (e:Event)
        WHERE e.uid IN $event_uids
        MERGE (e)-[:CONSOLIDATED_INTO]->(m)
        """
        
        async with self._driver.session() as session:
            try:
                await session.run(query, parameters={"macro_uid": macro_event_uid, "event_uids": event_uids})
                logger.info(f"Consolidated {len(event_uids)} events into {macro_event_uid}")
            except Exception as e:
                logger.error(f"Error consolidating events: {e}")
    
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
                        updated: $updated,
                        visit_count: $visit_count,
                        last_visited: $last_visited
                    })
                """, parameters={
                    "id": state.id,
                    "type": state.type,
                    "desc": state.desc,
                    "status": state.status,
                    "created": int(state.created),
                    "updated": int(state.updated),
                    "visit_count": state.visit_count,
                    "last_visited": state.last_visited
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
                        "updated": state_node.get("updated"),
                        "visit_count": state_node.get("visit_count", 0),
                        "last_visited": state_node.get("last_visited", 0.0)
                    }
                return None
            except Exception as e:
                logger.error(f"Error finding similar state: {e}")
                return None

    async def update_state_status(self, state_id: str, new_status: str) -> bool:
        """
        COLD PATH: Update the status of an existing state.
        Resets visit_count to 0 (progress made = fresh start).
        Returns True if successful.
        """
        import time
        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (s:State {id: $state_id})
                    SET s.status = $new_status,
                        s.updated = $updated,
                        s.visit_count = 0,
                        s.last_visited = 0.0
                    RETURN s
                """, parameters={
                    "state_id": state_id,
                    "new_status": new_status,
                    "updated": int(time.time())
                })

                record = await result.single()
                if record:
                    logger.debug(f"Updated state {state_id} to {new_status} (visit_count reset)")
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
                        "updated": state_node.get("updated"),
                        "visit_count": state_node.get("visit_count", 0),
                        "last_visited": state_node.get("last_visited", 0.0)
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
                        "updated": state_node.get("updated"),
                        "visit_count": state_node.get("visit_count", 0),
                        "last_visited": state_node.get("last_visited", 0.0)
                    })

                logger.debug(f"Retrieved {len(states)} completed states (type={state_type})")
                return states
            except Exception as e:
                logger.error(f"Error getting completed states: {e}")
                return []

    async def increment_state_visits(self, state_ids: List[str]) -> int:
        """
        HOT PATH: Increment visit_count for multiple states (batch update).
        Called when states are injected into context.
        Returns number of states successfully updated.
        """
        if not state_ids:
            return 0

        import time
        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    UNWIND $state_ids AS state_id
                    MATCH (s:State {id: state_id})
                    SET s.visit_count = COALESCE(s.visit_count, 0) + 1,
                        s.last_visited = $timestamp
                    RETURN count(s) AS updated_count
                """, parameters={
                    "state_ids": state_ids,
                    "timestamp": time.time()
                })

                record = await result.single()
                count = record['updated_count'] if record else 0

                if count > 0:
                    logger.debug(f"Incremented visit_count for {count} states")
                return count
            except Exception as e:
                logger.error(f"Error incrementing state visits: {e}")
                return 0

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