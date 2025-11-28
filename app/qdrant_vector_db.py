"""Qdrant-based vector search implementation"""

import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantVectorDB:
    """Async wrapper for Qdrant client, managing vector storage and search"""
    
    def __init__(self, host: str, port: int, collection_name: str, dimension: int):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        # Qdrant client is synchronous, we run it in thread pool via asyncio.to_thread
        self.client: QdrantClient = None
        logger.info(f"QdrantVectorDB configured for {host}:{port}, collection: {collection_name}")
    
    async def init(self):
        """Initialize Qdrant client and ensure collection exists"""
        try:
            # Create client in thread pool
            await asyncio.to_thread(self._init_client)
            logger.info(f"Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def _init_client(self):
        """Synchronous client initialization (runs in thread pool)"""
        self.client = QdrantClient(host=self.host, port=self.port, timeout=10.0)
        
        try:
            # Check if collection exists
            info = self.client.get_collection(collection_name=self.collection_name)
            
            # Verify dimension matches
            if info.config.params.vectors.size != self.dimension:
                logger.warning(
                    f"Collection dimension mismatch: {info.config.params.vectors.size} vs {self.dimension}. "
                    f"Recreating collection."
                )
                self.client.delete_collection(collection_name=self.collection_name)
                self._create_collection()
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' exists (vectors={info.vectors_count})")
                
        except (UnexpectedResponse, ValueError):
            # Collection doesn't exist, create it
            logger.info(f"Collection '{self.collection_name}' not found, creating...")
            self._create_collection()
    
    def _create_collection(self):
        """Create the Qdrant collection (synchronous)"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE
            ),
            on_disk_payload=True  # Store payloads on disk to save memory
        )
        logger.info(f"Created Qdrant collection '{self.collection_name}' (dim={self.dimension})")
    
    async def upsert_vector(self, job_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store or update a vector point"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        # 1. Generate a valid UUID for Qdrant (Fixes the "Format error")
        point_id = str(uuid.uuid4())

        # 2. Inject the original job_id into metadata for traceability
        metadata["_job_id"] = job_id

        def sync_upsert():
            point = PointStruct(
                id=point_id,       # <--- Uses the valid UUID
                vector=embedding,
                payload=metadata   # <--- Contains the original job_id
            )
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point]
            )
            
        # Ensure you are calling this via asyncio.to_thread if wrapping sync code
        await asyncio.to_thread(sync_upsert)
        logger.debug(f"Upserted vector for job_id={job_id} in Qdrant")
    
    async def search(self, query_vector: List[float], top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search Qdrant for nearest neighbors.
        Returns list of dicts with job_id, score, and payload.
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        def sync_search():
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=filter_dict
            )
        
        search_results = await asyncio.to_thread(sync_search)

        results = []
        for hit in search_results:
            # Extract the original job_id from payload (stored as _job_id during upsert)
            payload = hit.payload or {}
            job_id = payload.get("_job_id", str(hit.id))  # Fallback to UUID if _job_id missing
            
            # Also try to get node_id from payload (new schema)
            node_id = payload.get("node_id")

            results.append({
                "job_id": job_id,
                "node_id": node_id,
                "score": hit.score,
                "payload": payload
            })

        return results
    
    def create_filter(self, must_conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Helper to create Qdrant filters.
        Example:
            must_conditions = [
                {"key": "domain", "match": {"value": "coding"}},
                {"key": "subtype", "match": {"value": "method"}}
            ]
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        conditions = []
        for cond in must_conditions:
            key = cond["key"]
            value = cond["match"]["value"]
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=conditions)
    
    async def get_vector(self, job_id: str) -> Optional[Dict]:
        """Retrieve a specific vector point by job_id"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        def sync_retrieve():
            return self.client.retrieve(
                collection_name=self.collection_name,
                ids=[job_id],
                with_payload=True,
                with_vectors=True
            )
        
        results = await asyncio.to_thread(sync_retrieve)
        
        if results:
            point = results[0]
            return {
                "job_id": str(point.id),
                "vector": point.vector,
                "payload": point.payload or {}
            }
        return None
    
    async def delete_vector(self, job_id: str) -> bool:
        """Delete a vector point by job_id"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        def sync_delete():
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[job_id],
                wait=True
            )
        
        try:
            await asyncio.to_thread(sync_delete)
            logger.debug(f"Deleted vector {job_id} from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector {job_id}: {e}")
            return False
    
    async def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        def sync_info():
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status
            }
        
        return await asyncio.to_thread(sync_info)
    
    async def shutdown(self):
        """Close Qdrant client"""
        if self.client:
            # Qdrant client doesn't require explicit closing in current version
            logger.info("Qdrant client shutdown")
