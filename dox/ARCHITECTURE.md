# VICW Phase 2 Architecture Documentation

This document provides an in-depth explanation of the Virtual Infinite Context Window (VICW) Phase 2 architecture, design decisions, and implementation details.

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Memory Systems](#memory-systems)
6. [Design Decisions](#design-decisions)
7. [Performance Characteristics](#performance-characteristics)

## Overview

VICW Phase 2 is a production-ready implementation of a virtual infinite context window system that enables language models to maintain persistent context beyond their native token limits.

### Key Innovations

1. **Hot/Cold Path Separation**: Critical path optimization with async offload
2. **Hybrid Memory Architecture**: Vector + Graph + Key-Value stores
3. **Deterministic Pressure Control**: Predictable behavior with hysteresis
4. **RAG-Enhanced Generation**: Automatic memory retrieval

### Evolution from Phase 1

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| LLM | Local (llama.cpp) | External API (OpenRouter/OpenAI) |
| Vector DB | FAISS (file-based) | Qdrant (network service) |
| Chunk Store | SQLite (file-based) | Redis (network service) |
| Knowledge Graph | None | Neo4j (graph database) |
| Deployment | Local only | Docker-based, scalable |
| Architecture | Monolithic | Modular microservices |

## Core Concepts

### 1. Virtual Infinite Context Window

The fundamental problem: LLMs have fixed context windows (e.g., 4K, 8K, 128K tokens), but conversations can be arbitrarily long.

**Solution**: Separate "working memory" from "long-term memory"

```
Working Memory (Context Window)
├── Pinned State Header (never offloaded)
│   ├── Goals
│   ├── Constraints
│   ├── Current Plan
│   └── Active Entities
└── Recent Messages (dynamically managed)
    ├── System message
    ├── User message
    ├── Assistant response
    └── ... (oldest messages offloaded when full)

Long-Term Memory (External Storage)
├── Full conversation history (Redis)
├── Semantic embeddings (Qdrant)
└── Knowledge graph (Neo4j)
```

### 2. Hot Path vs Cold Path

**Hot Path** (latency-critical):
- Synchronous operations in request/response cycle
- Token counting, pressure detection
- Context assembly, LLM generation
- Target: <10ms for pressure relief

**Cold Path** (throughput-optimized):
- Asynchronous background processing
- Summarization, embedding generation
- Database writes, graph updates
- Target: <1s per job

### 3. Pressure Control with Hysteresis

Prevents "thrashing" where system repeatedly offloads small amounts:

```
Tokens:     0 -------|--------|--------|--------|--------| Max
                     60%      70%      80%      90%     100%
                     Target   Hysteresis Trigger
                     
Timeline:
1. Context grows to 80% (3276 tokens) → Trigger offload
2. Drop to 60% (2457 tokens) → Target
3. Context grows to 70% (2867 tokens) → Don't trigger (hysteresis)
4. Context grows to 80% (3276 tokens) → Trigger again
```

### 4. Hybrid RAG Retrieval

Combines two search paradigms:

**Semantic Search** (Qdrant):
```
Query: "Tell me about our Q3 strategy"
  ↓ (embed)
[0.234, -0.567, 0.123, ...]
  ↓ (cosine similarity)
Similar vectors in Qdrant
  ↓ (lookup job_ids)
Summaries from Redis
```

**Relational Search** (Neo4j):
```
Query: "Tell me about our Q3 strategy"
  ↓ (keyword extraction)
Match entities: ["Q3", "strategy"]
  ↓ (graph traversal)
Related nodes and relationships
  ↓ (format)
Structured facts: "(Q3)-[:HAS_GOAL]->(Revenue Target)"
```

## Component Architecture

### Layer 1: API Layer

```python
api_server.py (FastAPI)
├── /chat endpoint
│   ├── Add user message
│   ├── Trigger RAG
│   ├── Generate response
│   └── Add assistant message
├── /health endpoint
├── /stats endpoint
└── /reset endpoint
```

### Layer 2: Context Management (Hot Path)

```python
context_manager.py
├── Token counting (_token_count)
├── Pressure monitoring (add_message)
├── Offload triggering (_relieve_pressure)
│   ├── Extract oldest messages
│   ├── Create OffloadJob
│   ├── Enqueue (non-blocking)
│   └── Insert placeholder
└── RAG injection (augment_context_with_memory)
    ├── Generate query embedding
    ├── Hybrid retrieval
    └── Inject into context
```

### Layer 3: Queue Infrastructure

```python
offload_queue.py
├── Async queue (deque + asyncio.Lock)
├── Enqueue (hot path)
├── Dequeue batch (cold path)
└── Statistics tracking
```

### Layer 4: Semantic Processing (Cold Path)

```python
semantic_manager.py
├── Process job
│   ├── Summarization (CPU-bound)
│   ├── Embedding (CPU-bound)
│   ├── Redis storage (async I/O)
│   ├── Qdrant indexing (async I/O)
│   └── Neo4j update (async I/O)
└── Query memory
    ├── Semantic search (Qdrant)
    └── Relational search (Neo4j)

cold_path_worker.py
├── Background loop
├── Batch processing
└── Pause/resume coordination
```

### Layer 5: Storage Layer

```python
redis_storage.py
├── Chunk storage (HSET)
├── Index tracking (ZADD)
└── Retrieval (HGETALL, pipeline)

qdrant_vector_db.py
├── Collection management
├── Vector upsert
└── Similarity search

neo4j_knowledge_graph.py
├── Entity management
├── Relationship creation
└── Graph traversal queries
```

### Layer 6: LLM Interface

```python
llm_inference.py
├── HTTP client (httpx.AsyncClient)
├── Generation (OpenAI-compatible API)
└── Retry logic
```

## Data Flow

### Chat Request Flow

```
1. User sends message
   ↓
2. api_server.py: POST /chat
   ↓
3. context_manager.add_message("user", message)
   ├── Check pressure
   └── Trigger offload if needed
        ├── Extract messages
        ├── Create OffloadJob
        └── Queue → offload_queue
   ↓
4. context_manager.augment_context_with_memory(message)
   ├── Generate query embedding
   ├── semantic_manager.query_memory()
   │   ├── Qdrant search
   │   ├── Redis lookup
   │   └── Neo4j query
   └── Inject retrieved context
   ↓
5. cold_path_worker.pause() (prevent CPU contention)
   ↓
6. llm.generate(context_window)
   ├── HTTP POST to external API
   └── Await response
   ↓
7. cold_path_worker.resume()
   ↓
8. context_manager.add_message("assistant", response)
   ↓
9. Return response to user
```

### Cold Path Processing

```
Background Loop (continuous):
1. Dequeue batch from offload_queue
   ↓
2. For each OffloadJob:
   ├── Summarize (CPU-bound, in thread pool)
   ├── Embed (CPU-bound, in thread pool)
   ├── Store in Redis (async I/O)
   ├── Index in Qdrant (async I/O)
   └── Update Neo4j (async I/O)
   ↓
3. Update processed count
   ↓
4. Sleep if queue empty
```

## Memory Systems

### 1. Redis (Chunk Storage)

**Purpose**: Fast key-value storage for conversation chunks

**Schema**:
```
Key: chunk:<job_id>
Value: Hash {
  job_id: "job_abc123"
  chunk_text: "user: ... assistant: ..."
  summary: "Discussion about X..."
  metadata: JSON
  timestamp: Unix timestamp
  token_count: Integer
  message_count: Integer
}

Index: chunk_index (Sorted Set)
  Score: timestamp
  Member: job_id
```

**Operations**:
- Store: O(1) HSET
- Retrieve: O(1) HGETALL
- Batch retrieve: O(N) pipeline
- Recent chunks: O(log N) ZREVRANGE

### 2. Qdrant (Vector Search)

**Purpose**: Semantic similarity search using embeddings

**Schema**:
```
Collection: vicw_memory
  Vectors: 384-dimensional (all-MiniLM-L6-v2)
  Distance: Cosine
  
Point: {
  id: job_id (string)
  vector: [0.234, -0.567, ...]
  payload: {
    job_id: "job_abc123"
    summary: "Discussion about X..."
    token_count: 150
    timestamp: 1705843200
  }
}
```

**Operations**:
- Upsert: O(log N) with HNSW index
- Search: O(log N) approximate nearest neighbors
- Batch search: O(K * log N) for K queries

### 3. Neo4j (Knowledge Graph)

**Purpose**: Relational tracking of entities and relationships

**Schema**:
```
Nodes:
  (:Chunk {job_id, summary, created_at})
  (:Entity {name, type, properties})

Relationships:
  (:Chunk)-[:EXTRACTED_FROM]->(:Entity)
  (:Entity)-[:RELATES_TO]->(:Entity)
  (:Entity)-[:HAS_PROPERTY]->(value)
```

**Queries**:
```cypher
// Find related entities
MATCH (e:Entity)-[r]->(m)
WHERE e.name CONTAINS $query
RETURN e, type(r), m
LIMIT 5

// Update chunk
MERGE (c:Chunk {job_id: $job_id})
SET c.summary = $summary, c.processed_at = timestamp()
```

## Design Decisions

### 1. Why External LLM Instead of Local?

**Pros**:
- No GPU required (lower infrastructure cost)
- Access to latest models (GPT-4, Claude, etc.)
- Better quality for complex tasks
- Easier scaling (just API calls)

**Cons**:
- API costs per request
- Network latency
- Dependency on external service

**Decision**: External API for production flexibility

### 2. Why Qdrant Instead of FAISS?

**FAISS Limitations**:
- File-based (not network accessible)
- No built-in persistence
- Single-node only
- Requires manual index management

**Qdrant Advantages**:
- Network service (scalable)
- Built-in persistence
- Distributed support
- REST API

**Decision**: Qdrant for production scalability

### 3. Why Redis Instead of SQLite?

**SQLite Limitations**:
- File-based (contention issues)
- Limited concurrency
- No network access
- Complex async handling

**Redis Advantages**:
- Network service
- Atomic operations
- High concurrency
- Native async support
- Built-in data structures (sorted sets)

**Decision**: Redis for performance and scalability

### 4. Why Neo4j for Knowledge Graph?

**Alternatives Considered**:
- NetworkX (in-memory only)
- ArangoDB (multi-model but complex)
- SQL with joins (poor graph performance)

**Neo4j Advantages**:
- Native graph storage
- Cypher query language
- ACID transactions
- Mature ecosystem

**Decision**: Neo4j for graph-native operations

### 5. Why Thread Pool for CPU-Bound Tasks?

**Problem**: Embedding and summarization are CPU-intensive

**Options**:
1. Run in event loop (blocks everything)
2. Multiprocessing (high overhead)
3. Thread pool (GIL released in NumPy/PyTorch)

**Decision**: Thread pool via `asyncio.to_thread()` and `ThreadPoolExecutor`

### 6. Why Extractive Summarization?

**Options**:
1. LLM-based (high quality but slow/expensive)
2. Extractive (fast but lower quality)

**Decision**: Extractive for Phase 2, LLM-based optional for future

## Performance Characteristics

### Latency Targets

| Operation | Target | Typical |
|-----------|--------|---------|
| Pressure relief (hot) | <10ms | 5-8ms |
| RAG retrieval | <100ms | 50-80ms |
| LLM generation | <2s | 0.5-2s |
| Cold path job | <1s | 300-800ms |

### Throughput

| Metric | Value |
|--------|-------|
| Messages/second | 10-50 |
| Offload jobs/second | 5-10 |
| RAG queries/second | 20-100 |

### Memory Usage

| Component | RAM |
|-----------|-----|
| API server | 1-2 GB |
| Redis | 0.5-2 GB |
| Qdrant | 1-2 GB |
| Neo4j | 1-2 GB |
| **Total** | **4-8 GB** |

### Scaling Limits

| Dimension | Limit |
|-----------|-------|
| Context tokens | 128K (API dependent) |
| Stored chunks | Millions (Redis/Qdrant) |
| Graph nodes | Millions (Neo4j) |
| Concurrent users | 100+ (with load balancer) |

## Future Enhancements

### Planned Features
1. **Advanced Summarization**: LLM-based abstractive summaries
2. **Entity Extraction**: NLP-based entity recognition
3. **Relationship Extraction**: Automatic relationship detection
4. **Multi-tenant Support**: Isolated contexts per user
5. **Streaming Responses**: SSE for real-time generation
6. **Conversation Branching**: Support for alternative conversation paths
7. **Fine-grained Access Control**: Role-based memory access

### Research Directions
1. **Adaptive Compression**: Dynamic compression based on importance
2. **Hierarchical Memory**: Multi-level memory organization
3. **Cross-conversation Transfer**: Learning across sessions
4. **Markovian Compression**: Intelligent state compression

---

This architecture balances simplicity, performance, and scalability while maintaining extensibility for future enhancements.
