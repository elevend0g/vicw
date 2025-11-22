# VICW Phase 2 - File Manifest

This document lists all files in the VICW Phase 2 codebase and their purposes.

## Core Python Modules (12 files)

### Configuration & Data Models
- **config.py** - Centralized configuration from environment variables
- **data_models.py** - Data classes (OffloadJob, Message, PinnedHeader, RAGResult, etc.)

### Context Management (Hot Path)
- **context_manager.py** - Hot path pressure control, context assembly, RAG injection
- **offload_queue.py** - Thread-safe async queue for offload jobs

### Semantic Processing (Cold Path)
- **semantic_manager.py** - Summarization, embedding, storage coordination
- **cold_path_worker.py** - Background worker for async job processing

### Storage Interfaces
- **redis_storage.py** - Redis client for chunk persistence
- **qdrant_vector_db.py** - Qdrant client for vector search
- **neo4j_knowledge_graph.py** - Neo4j client for knowledge graph

### Infrastructure
- **llm_inference.py** - External LLM client (OpenAI-compatible APIs)
- **api_server.py** - FastAPI HTTP server with endpoints
- **main.py** - CLI orchestrator for interactive use

## Documentation (5 files)

- **README.md** - Main documentation, overview, usage guide
- **QUICKSTART.md** - 5-minute quick start guide
- **DEPLOYMENT.md** - Production deployment guide
- **ARCHITECTURE.md** - In-depth architecture documentation
- **FILE_MANIFEST.md** - This file

## Deployment & Configuration (6 files)

- **requirements.txt** - Python dependencies
- **Dockerfile** - Container image definition
- **docker-compose.yml** - Full stack orchestration
- **.env.example** - Environment variable template
- **.gitignore** - Git ignore patterns
- **system_prompt.txt** - Sample system prompt for the LLM

## Total: 23 Files

## Quick Reference

### To Start Development:
1. Read: QUICKSTART.md
2. Configure: .env.example → .env
3. Run: `docker-compose up -d`

### To Deploy to Production:
1. Read: DEPLOYMENT.md
2. Review: ARCHITECTURE.md
3. Secure: Follow security best practices
4. Monitor: Set up metrics and logging

### To Understand the System:
1. High-level: README.md
2. Technical: ARCHITECTURE.md
3. Code: Start with api_server.py → context_manager.py → semantic_manager.py

## Module Dependencies

```
api_server.py
├── config.py
├── context_manager.py
│   ├── data_models.py
│   ├── offload_queue.py
│   └── semantic_manager.py
│       ├── redis_storage.py
│       ├── qdrant_vector_db.py
│       └── neo4j_knowledge_graph.py
├── cold_path_worker.py
│   └── semantic_manager.py
└── llm_inference.py
    └── config.py
```

## Lines of Code

| Category | Files | Approx. Lines |
|----------|-------|---------------|
| Core Logic | 12 | ~3,500 |
| Documentation | 5 | ~2,500 |
| Configuration | 6 | ~300 |
| **Total** | **23** | **~6,300** |

## File Size Summary

| File | Lines | Purpose |
|------|-------|---------|
| api_server.py | ~260 | HTTP API endpoints |
| main.py | ~220 | CLI orchestrator |
| context_manager.py | ~350 | Hot path management |
| semantic_manager.py | ~280 | Cold path processing |
| cold_path_worker.py | ~170 | Background worker |
| redis_storage.py | ~180 | Redis interface |
| qdrant_vector_db.py | ~180 | Qdrant interface |
| neo4j_knowledge_graph.py | ~240 | Neo4j interface |
| llm_inference.py | ~130 | LLM client |
| offload_queue.py | ~120 | Async queue |
| data_models.py | ~160 | Data structures |
| config.py | ~90 | Configuration |
| README.md | ~550 | Main docs |
| ARCHITECTURE.md | ~650 | Technical docs |
| DEPLOYMENT.md | ~550 | Deployment guide |
| QUICKSTART.md | ~250 | Quick start |

## Key Design Patterns

1. **Async/Await Throughout**: All I/O operations are async
2. **Dependency Injection**: Components receive dependencies in __init__
3. **Thread Pool for CPU-Bound**: Embedding/summarization in executor
4. **Queue-Based Decoupling**: Hot/cold path separation
5. **Type Hints**: All functions use type annotations
6. **Centralized Configuration**: Single config.py module
7. **Modular Architecture**: Each component is independent

## Testing Strategy (To Be Implemented)

Recommended test structure:
```
tests/
├── test_context_manager.py
├── test_semantic_manager.py
├── test_offload_queue.py
├── test_redis_storage.py
├── test_qdrant_vector_db.py
├── test_neo4j_knowledge_graph.py
├── test_llm_inference.py
└── test_integration.py
```

## Extension Points

To add new features:

1. **New Storage Backend**: Implement same interface as redis_storage.py
2. **New Vector DB**: Implement same interface as qdrant_vector_db.py
3. **New LLM Provider**: Modify llm_inference.py
4. **New Summarization**: Modify semantic_manager._summarize_sync()
5. **New Retrieval Strategy**: Modify semantic_manager.query_memory()

## Version Information

- **Phase**: 2.0.0
- **Python**: 3.11+
- **API Framework**: FastAPI 0.104.1
- **Databases**: Redis 7, Qdrant 1.7, Neo4j 5
- **Architecture**: Async-first, microservices-ready

---

For questions about specific files, see the inline comments and docstrings in each module.
