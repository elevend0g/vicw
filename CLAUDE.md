# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VICW Phase 2 is a Virtual Infinite Context Window system that enables LLMs to maintain persistent context beyond their native token limits. It uses a hot/cold path architecture to separate latency-critical operations (pressure control, token counting) from throughput-optimized operations (embeddings, summarization, storage).

## Common Commands

### Development
```bash
# Start full stack locally
docker-compose up -d

# Start only infrastructure (for local API development)
docker-compose up -d redis qdrant neo4j

# Run API server locally
cd app && python api_server.py

# Run CLI mode
cd app && python main.py

# Check system health
curl http://localhost:8000/health

# Get system statistics
curl http://localhost:8000/stats

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "your message", "use_rag": true}'
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Docker Operations
```bash
# View logs
docker-compose logs -f vicw_api

# Restart specific service
docker-compose restart vicw_api

# Stop and remove all containers
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

## Architecture

### Hot Path vs Cold Path Separation

**Hot Path** (synchronous, latency-critical):
- Lives in `context_manager.py`
- Token counting and pressure monitoring
- Triggers offload when context reaches 80% capacity (OFFLOAD_THRESHOLD)
- Drops context to 60% (TARGET_AFTER_RELIEF) with hysteresis at 70% (HYSTERESIS_THRESHOLD)
- Enqueues jobs to offload_queue (non-blocking)
- Handles RAG injection during generation

**Cold Path** (asynchronous, background):
- Orchestrated by `cold_path_worker.py` using `semantic_manager.py`
- CPU-bound tasks (summarization, embedding) run in thread pool via `asyncio.to_thread()`
- I/O-bound tasks (Redis, Qdrant, Neo4j) use async clients
- Pauses during LLM generation to prevent CPU contention

### Data Flow
1. User message → `context_manager.add_message()` → check pressure
2. If pressure > 80%: extract messages → create OffloadJob → enqueue (non-blocking)
3. Before LLM generation: optionally call `augment_context_with_memory()` for RAG
4. `cold_path_worker` continuously processes queue in background
5. Cold path: summarize → embed → Redis → Qdrant → Neo4j

### Storage Architecture

**Redis** (`redis_storage.py`):
- Key-value storage for conversation chunks
- Schema: `chunk:<job_id>` contains hash with chunk_text, summary, metadata, timestamp
- Index: `chunk_index` sorted set for temporal queries

**Qdrant** (`qdrant_vector_db.py`):
- Vector database for semantic similarity search
- Collection: `vicw_memory` with 384-dimensional vectors (all-MiniLM-L6-v2)
- Used for semantic retrieval in RAG

**Neo4j** (`neo4j_knowledge_graph.py`):
- Knowledge graph for relational tracking
- Nodes: `:Chunk` and `:Entity`
- Used for relational retrieval in RAG

### Module Dependency Tree
```
api_server.py
├── context_manager.py
│   ├── offload_queue.py (thread-safe async queue)
│   └── semantic_manager.py
│       ├── redis_storage.py
│       ├── qdrant_vector_db.py
│       └── neo4j_knowledge_graph.py
├── cold_path_worker.py (background processing loop)
├── llm_inference.py (external LLM client)
├── data_models.py (OffloadJob, Message, PinnedHeader, RAGResult)
└── config.py (centralized env var configuration)
```

## Key Implementation Details

### Pressure Control with Hysteresis
To prevent "thrashing" (repeatedly offloading small amounts), the system uses three thresholds:
- OFFLOAD_THRESHOLD (0.80): Trigger relief at 3276 tokens (for 4096 max)
- TARGET_AFTER_RELIEF (0.60): Drop to ~2457 tokens
- HYSTERESIS_THRESHOLD (0.70): Don't re-trigger until 2867 tokens

This is implemented in `context_manager.py:add_message()` and `_relieve_pressure()`.

### Async/Thread Pool Pattern
CPU-bound operations (embeddings, summarization) cannot run in the event loop without blocking. The codebase uses:
```python
# In semantic_manager.py
await asyncio.to_thread(self._summarize_sync, chunk_text)
await asyncio.to_thread(self._embed_sync, text)
```

### RAG (Retrieval-Augmented Generation)
Hybrid retrieval combines:
1. **Semantic search** (Qdrant): Vector similarity using query embeddings
2. **Relational search** (Neo4j): Graph traversal for entity relationships

Results are injected into context before LLM generation via `context_manager.augment_context_with_memory()`.

### Pinned State Header
`PinnedHeader` (in `data_models.py`) contains state that never gets offloaded:
- Goals, constraints, current plan, active entities
- Always included at the start of context via `context_manager.get_context_window()`

## Configuration

All configuration is in environment variables (see `config.py` and `docker-compose.yml`).

**Critical variables:**
- `VICW_LLM_API_KEY`: Required for external LLM (OpenRouter, OpenAI, etc.)
- `VICW_LLM_API_URL`: Default is OpenRouter
- `VICW_LLM_MODEL_NAME`: Default is `mistralai/mistral-7b-instruct`
- `MAX_CONTEXT_TOKENS`: Default 4096
- `OFFLOAD_THRESHOLD`: Default 0.80
- `TARGET_AFTER_RELIEF`: Default 0.60

**Database connections:**
- Redis: `REDIS_HOST`, `REDIS_PORT` (default: localhost:6379)
- Qdrant: `QDRANT_HOST`, `QDRANT_PORT` (default: localhost:6333)
- Neo4j: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

## Code Conventions

### File Organization
- All Python modules are in `app/` directory
- Documentation in `dox/` directory
- Deployment configs (Dockerfile, docker-compose.yml) at root

### Imports
- Relative imports between modules in `app/`: `from data_models import OffloadJob`
- Dockerfile copies `*.py` from root to `/app`, so imports work without package prefix

### Error Handling
- Use `logger.error()` for errors with stack traces
- Use `metrics_logger.info()` for structured metrics
- Critical failures should raise exceptions; non-critical failures should log and continue

### Type Hints
All functions use type hints. When adding new code, maintain this pattern:
```python
async def process_job(self, job: OffloadJob) -> bool:
    ...
```

### Logging Pattern
```python
logger.info(f"Human-readable message")
metrics_logger.info(f"METRIC_NAME | key1=value1 | key2=value2")
```

## Common Development Tasks

### Adding a new storage backend
1. Create new module in `app/` (e.g., `postgres_storage.py`)
2. Implement same interface as `redis_storage.py` (store_chunk, retrieve_chunk)
3. Add configuration in `config.py`
4. Modify `semantic_manager.py` to use new backend

### Modifying pressure control thresholds
1. Update `config.py` defaults or set environment variables
2. Logic is in `context_manager.py:add_message()` and `_relieve_pressure()`
3. Test with different MAX_CONTEXT_TOKENS values

### Adding new RAG retrieval strategies
1. Modify `semantic_manager.py:query_memory()`
2. Update `RAGResult` in `data_models.py` if needed
3. Adjust `RAG_TOP_K_SEMANTIC` and `RAG_TOP_K_RELATIONAL` in config

### Changing summarization method
1. Modify `semantic_manager.py:_summarize_sync()`
2. Current implementation is extractive (simple sentence extraction)
3. For LLM-based summarization, add call to `llm_inference.py`

## Troubleshooting

### "Redis connection failed"
- Ensure Redis is running: `docker-compose up -d redis`
- Check REDIS_HOST and REDIS_PORT match running service

### "Qdrant client not initialized"
- Ensure Qdrant is running: `docker-compose up -d qdrant`
- Check health: `curl http://localhost:6333/health`

### "Neo4j authentication failed"
- Verify NEO4J_PASSWORD matches docker-compose.yml (default: vicw_password)
- Check logs: `docker-compose logs neo4j`

### High memory usage
- Reduce MAX_CONTEXT_TOKENS
- Reduce MAX_OFFLOAD_QUEUE_SIZE
- Adjust COLD_PATH_WORKERS and COLD_PATH_BATCH_SIZE

### Slow RAG performance
- Reduce RAG_TOP_K_SEMANTIC and RAG_TOP_K_RELATIONAL
- Check Qdrant and Neo4j performance with their admin UIs

## Important Notes

- The Dockerfile copies `*.py` files from root but code is actually in `app/` directory. When running locally, work in `app/` directory.
- Thread limits (OMP_NUM_THREADS, etc.) must be set before importing torch/numpy to prevent CPU oversubscription
- Cold path worker pauses during LLM generation to prevent CPU contention
- Offload jobs are processed asynchronously; don't expect immediate storage after enqueue
- Placeholder cards (`[ARCHIVED mem_id:...]`) remain in context after offload; full content is in storage
