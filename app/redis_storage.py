"""Redis-based persistent storage for offload jobs"""

import logging
import json
from typing import List, Dict, Any, Optional
import redis.asyncio as redis

from data_models import OffloadJob
from config import REDIS_CHUNK_TTL

logger = logging.getLogger(__name__)


class RedisStorage:
    """Async Redis storage for chunk persistence"""
    
    CHUNK_KEY_PREFIX = "chunk:"
    CHUNK_INDEX_KEY = "chunk_index"  # Sorted set for tracking chunks by timestamp
    
    def __init__(self, host: str, port: int, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis: Optional[redis.Redis] = None
    
    async def init(self):
        """Initialize Redis connection"""
        try:
            self.redis = await redis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"Redis connected to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def store_chunk(self, job: OffloadJob, summary: str) -> bool:
        """Store a chunk with summary in Redis"""
        if not self.redis:
            logger.error("Redis not initialized")
            return False
        
        key = self.CHUNK_KEY_PREFIX + job.job_id
        
        try:
            # Prepare chunk data
            chunk_data = {
                "job_id": job.job_id,
                "chunk_text": job.chunk_text,
                "summary": summary,
                "metadata": json.dumps(job.metadata),
                "timestamp": str(job.timestamp),
                "token_count": str(job.token_count),
                "message_count": str(job.message_count)
            }
            
            # Store chunk as hash
            await self.redis.hset(key, mapping=chunk_data)
            
            # Set TTL
            await self.redis.expire(key, REDIS_CHUNK_TTL)
            
            # Add to index (sorted set by timestamp)
            await self.redis.zadd(self.CHUNK_INDEX_KEY, {job.job_id: job.timestamp})
            
            logger.debug(f"Stored chunk {job.job_id} in Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunk {job.job_id}: {e}")
            return False
    
    async def get_chunk_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single chunk by job_id"""
        if not self.redis:
            return None
        
        key = self.CHUNK_KEY_PREFIX + job_id
        
        try:
            chunk = await self.redis.hgetall(key)
            if chunk:
                # Parse metadata
                if 'metadata' in chunk:
                    chunk['metadata'] = json.loads(chunk['metadata'])
                return chunk
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {job_id}: {e}")
            return None
    
    async def get_chunks_by_ids(self, job_ids: List[str], fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve multiple chunks by their job_ids.
        If fields is specified, only retrieve those fields.
        """
        if not self.redis or not job_ids:
            return []
        
        try:
            # Use pipeline for efficient batch retrieval
            pipe = self.redis.pipeline()
            
            for job_id in job_ids:
                key = self.CHUNK_KEY_PREFIX + job_id
                if fields:
                    pipe.hmget(key, fields)
                else:
                    pipe.hgetall(key)
            
            results = await pipe.execute()
            
            output = []
            for i, result in enumerate(results):
                if not result:
                    continue
                
                if fields:
                    # Reconstruct dict from field list
                    chunk = dict(zip(fields, result))
                    # Filter out None values
                    chunk = {k: v for k, v in chunk.items() if v is not None}
                else:
                    chunk = result
                
                if chunk:
                    # Parse metadata if present
                    if 'metadata' in chunk and chunk['metadata']:
                        try:
                            chunk['metadata'] = json.loads(chunk['metadata'])
                        except json.JSONDecodeError:
                            pass
                    output.append(chunk)
            
            return output
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    async def get_recent_chunks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent chunks"""
        if not self.redis:
            return []
        
        try:
            # Get most recent job_ids from sorted set
            job_ids = await self.redis.zrevrange(self.CHUNK_INDEX_KEY, 0, limit - 1)
            
            if not job_ids:
                return []
            
            # Retrieve full chunks
            return await self.get_chunks_by_ids(job_ids)
            
        except Exception as e:
            logger.error(f"Error retrieving recent chunks: {e}")
            return []
    
    async def delete_chunk(self, job_id: str) -> bool:
        """Delete a chunk by job_id"""
        if not self.redis:
            return False
        
        key = self.CHUNK_KEY_PREFIX + job_id
        
        try:
            # Delete from hash
            await self.redis.delete(key)
            
            # Remove from index
            await self.redis.zrem(self.CHUNK_INDEX_KEY, job_id)
            
            logger.debug(f"Deleted chunk {job_id} from Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunk {job_id}: {e}")
            return False
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks stored"""
        if not self.redis:
            return 0
        
        try:
            return await self.redis.zcard(self.CHUNK_INDEX_KEY)
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
    
    async def shutdown(self):
        """Close Redis connection pool"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
