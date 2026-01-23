"""
Redis Configuration
Redis connection for caching and task queue
"""

import json
from typing import Optional, Any, List
from redis import asyncio as aioredis
from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Global Redis client
redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Get Redis client instance"""
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info(f"Redis connected: {settings.REDIS_URL}")
    return redis_client


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


class RedisCache:
    """Redis cache operations"""
    
    def __init__(self, prefix: str = "theta"):
        self.prefix = prefix

    def _key(self, key: str) -> str:
        """Generate prefixed key"""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        redis = await get_redis()
        value = await redis.get(self._key(key))
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration (default 1 hour)"""
        redis = await get_redis()
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await redis.set(self._key(key), value, ex=expire)

    async def delete(self, key: str):
        """Delete key from cache"""
        redis = await get_redis()
        await redis.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        redis = await get_redis()
        return await redis.exists(self._key(key)) > 0

    async def expire(self, key: str, seconds: int):
        """Set expiration on key"""
        redis = await get_redis()
        await redis.expire(self._key(key), seconds)


class TaskQueue:
    """Simple task queue using Redis lists"""
    
    def __init__(self, queue_name: str = "task_queue"):
        self.queue_name = f"theta:{queue_name}"

    async def enqueue(self, task_data: dict):
        """Add task to queue"""
        redis = await get_redis()
        await redis.rpush(self.queue_name, json.dumps(task_data))
        logger.info(f"Task enqueued: {task_data.get('task_id', 'unknown')}")

    async def dequeue(self, timeout: int = 0) -> Optional[dict]:
        """Get task from queue (blocking with timeout)"""
        redis = await get_redis()
        if timeout > 0:
            result = await redis.blpop(self.queue_name, timeout=timeout)
            if result:
                return json.loads(result[1])
        else:
            result = await redis.lpop(self.queue_name)
            if result:
                return json.loads(result)
        return None

    async def length(self) -> int:
        """Get queue length"""
        redis = await get_redis()
        return await redis.llen(self.queue_name)

    async def clear(self):
        """Clear queue"""
        redis = await get_redis()
        await redis.delete(self.queue_name)


class TaskProgress:
    """Track task progress in Redis"""
    
    PREFIX = "theta:task_progress"

    @classmethod
    async def set_progress(cls, task_id: str, progress: float, status: str = None, message: str = None):
        """Set task progress"""
        redis = await get_redis()
        data = {
            "progress": progress,
            "updated_at": json.dumps(None)  # Will use Redis server time
        }
        if status:
            data["status"] = status
        if message:
            data["message"] = message
        
        await redis.hset(f"{cls.PREFIX}:{task_id}", mapping=data)
        await redis.expire(f"{cls.PREFIX}:{task_id}", 86400)  # 24 hours

    @classmethod
    async def get_progress(cls, task_id: str) -> Optional[dict]:
        """Get task progress"""
        redis = await get_redis()
        data = await redis.hgetall(f"{cls.PREFIX}:{task_id}")
        if data:
            return {
                "progress": float(data.get("progress", 0)),
                "status": data.get("status"),
                "message": data.get("message")
            }
        return None

    @classmethod
    async def delete_progress(cls, task_id: str):
        """Delete task progress"""
        redis = await get_redis()
        await redis.delete(f"{cls.PREFIX}:{task_id}")


# Create global instances
cache = RedisCache()
task_queue = TaskQueue()
