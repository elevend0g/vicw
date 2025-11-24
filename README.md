# VICW - Virtual Infinite Context Window

A production-ready implementation of a Virtual Infinite Context Window system that enables language models to maintain persistent context beyond their native token limits.

## 🎯 Overview

VICW combines smart archiving through external memory (knowledge graphs, vector embeddings) with intelligent retrieval mechanisms to create a scalable, async-first architecture for unlimited conversation context.

### Key Features

- **🔥 Hot/Cold Path Separation**: Deterministic pressure control (hot path) decoupled from semantic processing (cold path)
- **🧠 Hybrid Memory System**:
  - Redis for chunk storage
  - Qdrant for semantic vector search
  - Neo4j for knowledge graph and relational tracking
- **🔄 Advanced Loop Prevention System**:
  - **State Machine**: Tracks goals, tasks, decisions, and facts across domains
  - **Boredom Detection**: Monitors state visit frequency to detect repetitive focus
  - **Echo Guard**: Detects duplicate LLM responses with escalating interventions
- **🌐 External LLM Integration**: Works with any OpenAI-compatible API (OpenRouter, OpenAI, etc.)
- **📊 RAG-Enhanced Generation**: Automatic retrieval of relevant memories during inference
- **⚡ Async-First Architecture**: Non-blocking operations throughout
- **🐳 Full Docker Stack**: One-command deployment with docker-compose

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     HOT PATH (Context Manager)           │
│  - Token counting and pressure monitoring                │
│  - Offload triggering at 80% capacity                    │
│  - Shed-to-target with hysteresis (prevents thrashing)   │
│  - RAG injection for retrieved memories                  │
└─────────────┬────────────────────────────────────────────┘
              │ Enqueue Job
              ▼
┌──────────────────────────────────────────────────────────┐
│                     OFFLOAD QUEUE                        │
│  - Thread-safe async queue                               │
│  - Max size: 100 jobs                                    │
└─────────────┬────────────────────────────────────────────┘
              │ Batch Processing
              ▼
┌──────────────────────────────────────────────────────────┐
│              COLD PATH (Semantic Manager)                │
│  1. Extractive summarization (CPU-bound)                 │
│  2. Embedding generation (CPU-bound)                     │
│  3. Redis storage (async I/O)                           │
│  4. Qdrant indexing (async I/O)                         │
│  5. Neo4j graph update (async I/O)                      │
│  6. State extraction & tracking (pattern-based)          │
│  7. Response similarity check (echo detection)           │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│           LOOP PREVENTION (Multi-Layered)                │
│  • State Machine: Track goals/tasks/decisions/facts      │
│  • Boredom Detection: Monitor state visit frequency      │
│  • Echo Guard: Detect & regenerate duplicate responses   │
└──────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- An API key for an OpenAI-compatible LLM service

### Quick Start with Docker

1. **Clone and configure**:
```bash
cd vicw_api
cp .env.example .env
# Edit .env and add your VICW_LLM_API_KEY
```

2. **Start the stack**:
```bash
docker-compose up -d
```

3. **Check health**:
```bash
curl http://localhost:8000/health
```

The API will be available at `http://localhost:8000`

### Local Development Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start infrastructure** (Redis, Qdrant, Neo4j):
```bash
docker-compose up -d redis qdrant neo4j
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Run the API server**:
```bash
python api_server.py
```

Or run CLI mode:
```bash
python main.py
```

## 🚀 Usage

### API Endpoints

#### POST /chat
Send a chat message and get a response with optional RAG.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the VICW architecture",
    "use_rag": true
  }'
```

Response:
```json
{
  "response": "The VICW architecture consists of...",
  "timestamp": "2025-01-20T10:30:00",
  "tokens_in_context": 2048,
  "rag_items_injected": 3
}
```

#### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

#### GET /stats
Get system statistics including context, queue, and worker metrics.

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "context": {
    "current_tokens": 2048,
    "max_tokens": 4096,
    "message_count": 15,
    "offload_count": 2,
    "pressure_percentage": 50.0
  },
  "queue": {
    "current_size": 1,
    "max_size": 100,
    "processed_total": 5,
    "dropped_total": 0
  },
  "worker": {
    "is_running": true,
    "processed_count": 5,
    "failed_count": 0,
    "success_rate": 1.0
  }
}
```

#### POST /reset
Reset the context (useful for testing).

```bash
curl -X POST http://localhost:8000/reset
```

### CLI Mode

For interactive testing:

```bash
python main.py
```

Commands:
- Type your message to chat
- `stats` - Show system statistics
- `exit` - Quit

## ⚙️ Configuration

All configuration is done through environment variables. See `.env.example` for the full list.

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `VICW_LLM_API_KEY` | - | **Required**: Your LLM API key |
| `VICW_LLM_MODEL_NAME` | `mistralai/mistral-7b-instruct` | Model to use |
| `MAX_CONTEXT_TOKENS` | `4096` | Maximum context window size |
| `OFFLOAD_THRESHOLD` | `0.80` | Trigger offload at 80% capacity |
| `TARGET_AFTER_RELIEF` | `0.60` | Drop to 60% after offload |
| `RAG_TOP_K_SEMANTIC` | `2` | Number of semantic chunks to retrieve |
| `RAG_TOP_K_RELATIONAL` | `5` | Number of relational facts to retrieve |
| `STATE_TRACKING_ENABLED` | `true` | Enable state machine for loop prevention |
| `STATE_LIMIT_GOAL` | `2` | Max goals to inject into context |
| `STATE_LIMIT_TASK` | `3` | Max tasks to inject into context |
| `STATE_LIMIT_DECISION` | `2` | Max decisions to inject into context |
| `STATE_LIMIT_FACT` | `3` | Max facts to inject into context |
| `BOREDOM_DETECTION_ENABLED` | `true` | Enable state visit frequency monitoring |
| `BOREDOM_THRESHOLD` | `5` | Number of visits before warning triggers |
| `ECHO_GUARD_ENABLED` | `true` | Enable duplicate response detection |
| `ECHO_SIMILARITY_THRESHOLD` | `0.95` | Similarity threshold for duplicate detection |
| `ECHO_RESPONSE_HISTORY_SIZE` | `10` | Number of recent responses to compare |
| `MAX_REGENERATION_ATTEMPTS` | `3` | Max retries on duplicate detection |

### Pressure Control Tuning

The system uses three thresholds to prevent context overflow:

1. **OFFLOAD_THRESHOLD** (0.80): When to trigger pressure relief
2. **TARGET_AFTER_RELIEF** (0.60): Where to drop context after relief
3. **HYSTERESIS_THRESHOLD** (0.70): Prevent re-triggering until this point

Example: With a 4096 token limit:
- Offload triggers at 3276 tokens (80%)
- Context drops to ~2457 tokens (60%)
- Won't trigger again until 2867 tokens (70%)

This prevents "thrashing" where the system repeatedly offloads small amounts.

## 🧩 Module Overview

### Core Modules

- **`config.py`**: Centralized configuration from environment variables
- **`data_models.py`**: Data classes (OffloadJob, Message, PinnedHeader, etc.)
- **`context_manager.py`**: Hot path - pressure control and context management
- **`semantic_manager.py`**: Cold path - embeddings, summarization, storage
- **`cold_path_worker.py`**: Background worker for async job processing

### Storage Modules

- **`redis_storage.py`**: Redis client for chunk persistence
- **`qdrant_vector_db.py`**: Qdrant client for vector search
- **`neo4j_knowledge_graph.py`**: Neo4j client for knowledge graph and state tracking
- **`state_extractor.py`**: Pattern-based state detection for loop prevention
- **`state_config.yaml`**: Configurable patterns for state types (goals, tasks, decisions, facts)

### Infrastructure Modules

- **`offload_queue.py`**: Thread-safe async queue for offload jobs
- **`llm_inference.py`**: External LLM client (OpenAI-compatible)

### Entry Points

- **`api_server.py`**: FastAPI server for HTTP API
- **`main.py`**: CLI orchestrator for interactive use

## 📊 Metrics and Monitoring

The system logs detailed metrics to `vicw_metrics.log`:

- **CONTEXT_PRESSURE**: Token count and percentage per message
- **PRESSURE_RELIEF_HOT_PATH**: Offload triggering and timing
- **OFFLOAD_JOB_COMPLETE**: Cold path processing time
- **LLM_GENERATION**: Generation latency and response length
- **SEMANTIC_RETRIEVAL**: RAG retrieval performance
- **HYBRID_RETRIEVAL**: Combined semantic + relational retrieval
- **STATE_EXTRACTION**: State tracking detection and counts
- **BOREDOM_DETECTED**: State visit frequency warnings
- **ECHO_DETECTED**: Duplicate response detection and regeneration attempts
- **ECHO_GUARD_RETRY**: Similarity scores and retry attempts

Example metric log:
```
2025-01-20 10:30:15 - CONTEXT_PRESSURE | tokens=3500 | max=4096 | percentage=85.4 | message_role=user
2025-01-20 10:30:15 - PRESSURE_RELIEF_HOT_PATH | tokens_before=3500 | tokens_after=2400 | job_id=job_abc123 | relief_time_ms=8.2
2025-01-20 10:30:16 - LLM_GENERATION | time_ms=1250.5 | response_length=450 | model=mistralai/mistral-7b-instruct
```

## 🔧 Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio

# Run tests (when implemented)
pytest tests/

# Test state machine specifically
python3 test_state_machine.py
```

### Code Structure

```
vicw_api/
├── app/
│   ├── config.py                      # Configuration
│   ├── data_models.py                 # Data classes
│   ├── context_manager.py             # Hot path
│   ├── semantic_manager.py            # Cold path processing
│   ├── cold_path_worker.py            # Background worker
│   ├── offload_queue.py               # Async queue
│   ├── redis_storage.py               # Redis client
│   ├── qdrant_vector_db.py            # Qdrant client
│   ├── neo4j_knowledge_graph.py       # Neo4j client & state tracking
│   ├── state_extractor.py             # Pattern-based state detection
│   ├── state_config.yaml              # State machine patterns
│   ├── llm_inference.py               # External LLM client
│   ├── api_server.py                  # FastAPI server
│   └── main.py                        # CLI mode
├── test_state_machine.py          # State machine tests
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container image
├── docker-compose.yml             # Full stack
└── .env.example                   # Config template
```

## 🎓 Key Concepts

### Virtual Infinite Context Window (VICW)

The core idea is to separate "working memory" (the LLM's context window) from "long-term memory" (external storage). As the context fills up:

1. **Hot Path** detects pressure and extracts oldest messages
2. Messages are **queued** for processing (non-blocking)
3. **Cold Path** processes asynchronously:
   - Generates summary
   - Creates embedding
   - Stores in Redis (text)
   - Indexes in Qdrant (vector)
   - Updates Neo4j (graph)
   - Extracts and tracks states (goals, tasks, decisions, facts)
4. During generation, **RAG** retrieves relevant memories and **state machine** injects current states

### Hot Path vs Cold Path

**Hot Path** (synchronous, latency-critical):
- Token counting
- Pressure detection
- Context assembly
- LLM generation

**Cold Path** (asynchronous, throughput-optimized):
- Summarization
- Embedding generation
- Database writes
- Knowledge graph updates

### Hybrid Retrieval (RAG)

Combines two search methods:

1. **Semantic Search** (Qdrant): "What's similar to this query?"
   - Uses vector embeddings
   - Finds semantically related content
   
2. **Relational Search** (Neo4j): "What's connected to these entities?"
   - Uses knowledge graph
   - Finds structured relationships

Results are combined and injected into context before generation.

### State Machine (Loop Prevention)

The state machine prevents conversational loops by tracking conversation state across domains (narrative, coding, research, etc.).

**The Problem**: Without state tracking, LLMs can loop endlessly:
- Narrative: "Let's go to the Hydro-Plant" → arrives → forgets → "Let's go to the Hydro-Plant" (loop)
- Coding: "Let's refactor the auth module" → merges → forgets → "Let's refactor the auth module" (loop)

**The Solution**: Track state changes and inject them into context:

1. **Pattern-Based Detection** (Cold Path):
   - During offload, `state_extractor` scans text for state patterns
   - Detects 4 state types: **goals**, **tasks**, **decisions**, **facts**
   - Uses simple keyword patterns from `state_config.yaml`

2. **Neo4j Storage**:
   - Creates/updates `:State` nodes with type, description, status
   - Status transitions: `active` → `completed` or `invalid`
   - Fuzzy deduplication prevents duplicate states

3. **Context Injection** (Hot Path):
   - Before LLM generation, queries Neo4j for current states
   - Injects formatted state message with hard limits:
     - 2 goals, 3 tasks, 2 decisions, 3 facts (configurable)
   - Includes recently completed states as warnings

**Example Flow**:
```
User: "Let's go to the Hydro-Plant"
→ Offload → Extract → Create State(goal, "go to the hydro-plant", active)

User: "We arrived at the Hydro-Plant"
→ Offload → Extract → Update State(goal, "go to the hydro-plant", completed)

Next LLM generation:
→ Query Neo4j → Inject "[STATE MEMORY] Completed: go to the hydro-plant"
→ LLM sees completed state → Proposes next goal instead of looping
```

**Key Features**:
- Domain-agnostic: Works for narrative, coding, research, etc.
- Configurable: Edit patterns in `state_config.yaml` without code changes
- Soft prevention: Warns LLM but doesn't force behavior
- Fast: Rule-based pattern matching, no API calls

### Advanced Loop Prevention: Boredom Detection & Echo Guard

The system includes two additional mechanisms to prevent conversational loops:

#### 1. Boredom Detection (State Visit Counting)

Tracks how many times each state is injected into the context to detect when the LLM is stuck focusing on the same action repeatedly.

**How it works**:
- Each State node has `visit_count` and `last_visited` properties
- Every time a state is injected into context, its `visit_count` increments
- When `visit_count >= BOREDOM_THRESHOLD` (default: 5), a warning is injected:
  ```
  ⚠️ LOOP DETECTED: Repeated focus on [state_desc]. Consider concluding or exploring alternatives.
  ```
- Visit count resets to 0 when the state's status changes (e.g., active → completed)

**Example**:
```
Turn 1-5: "Let's format Table 2.3" (visit_count: 1 → 2 → 3 → 4 → 5)
Turn 6: Warning injected: "⚠️ LOOP DETECTED: Repeated focus on format Table 2.3"
LLM sees warning and concludes the action or pivots to different task
```

#### 2. Echo Guard (Response Similarity Detection)

Detects when the LLM generates nearly identical responses repeatedly and uses escalating interventions to force recovery.

**How it works**:
- After LLM generates a response, compute its embedding
- Compare embedding with recent response history (last N responses stored in Redis)
- If cosine similarity >= `ECHO_SIMILARITY_THRESHOLD` (default: 0.95):
  - Discard the duplicate response (don't add to context)
  - Inject escalating warning (progressively stronger with each retry)
  - Regenerate immediately (up to `MAX_REGENERATION_ATTEMPTS`, default: 3)

**Escalating Intervention Strategy**:

1. **Retry #1 - Polite Warning**: Asks LLM to provide new information
2. **Retry #2 - Forceful Directive**: Mandates specific response formats
3. **Retry #3 - Emergency Override**: Strips all RAG context and forces conclusion

**Critical difference from boredom detection**:
- **Boredom detection**: Prevents repeated focus on same state (e.g., talking about Table 2.3 repeatedly)
- **Echo guard**: Prevents identical responses (e.g., "Here is Table 2.3: [data]" appearing verbatim multiple times)

**Example flow**:
```
Turn 1: LLM generates: "Here is Table 2.3: [data]"
→ System embeds response → stores embedding → adds to context

Turn 2: LLM generates same text: "Here is Table 2.3: [data]"
→ System detects 0.98 similarity → discards response → injects warning → regenerates
→ LLM tries different approach: "Table 2.3 has been provided. Let's proceed to the next section."
```

## 🚨 Troubleshooting

### Common Issues

**Issue**: `Redis connection failed`
- **Solution**: Ensure Redis is running: `docker-compose up -d redis`

**Issue**: `Qdrant client not initialized`
- **Solution**: Check Qdrant is accessible: `curl http://localhost:6333/health`

**Issue**: `Neo4j authentication failed`
- **Solution**: Verify `NEO4J_PASSWORD` in `.env` matches docker-compose

**Issue**: `LLM generation timeout`
- **Solution**: Increase `LLM_TIMEOUT` or check API connectivity

**Issue**: Memory usage high
- **Solution**: Reduce `MAX_CONTEXT_TOKENS` or `MAX_OFFLOAD_QUEUE_SIZE`

**Issue**: States not being detected
- **Solution**: Check `app/state_config.yaml` patterns match your conversation style
- **Solution**: Run `python3 test_state_machine.py` to verify pattern matching
- **Solution**: Verify `STATE_TRACKING_ENABLED=true` in environment

**Issue**: Conversation still loops despite state machine
- **Solution**: Check Neo4j to verify states are created: `MATCH (s:State) RETURN s`
- **Solution**: Check logs for "Injected state tracking" messages
- **Solution**: Increase pattern coverage in `state_config.yaml`
- **Solution**: Ensure RAG is enabled (`use_rag: true` in API calls)

**Issue**: Boredom warnings not appearing
- **Solution**: Verify `BOREDOM_DETECTION_ENABLED=true` in environment
- **Solution**: Check `BOREDOM_THRESHOLD` setting (default: 5) - may be too high
- **Solution**: Query Neo4j to check visit_count: `MATCH (s:State) RETURN s.desc, s.visit_count ORDER BY s.visit_count DESC`
- **Solution**: Check logs for `BOREDOM_DETECTED` metrics
- **Solution**: Lower threshold temporarily for testing: `BOREDOM_THRESHOLD=2`

**Issue**: Echo guard not detecting duplicates
- **Solution**: Verify `ECHO_GUARD_ENABLED=true` in environment
- **Solution**: Check `ECHO_SIMILARITY_THRESHOLD` (default: 0.95) - may be too high
- **Solution**: Lower threshold for testing: `ECHO_SIMILARITY_THRESHOLD=0.85`
- **Solution**: Check logs for `ECHO_DETECTED` metrics
- **Solution**: Verify embeddings are being generated (check semantic_manager logs)

**Issue**: LLM keeps regenerating (too many echo detections)
- **Solution**: Increase `ECHO_SIMILARITY_THRESHOLD` (e.g., 0.98) to be more strict
- **Solution**: Reduce `ECHO_RESPONSE_HISTORY_SIZE` to compare against fewer responses
- **Solution**: Temporarily disable echo guard: `ECHO_GUARD_ENABLED=false`

**Issue**: Echo guard not breaking the loop (persistent repetition)
- **Solution**: Enable aggressive mode: `ECHO_STRIP_CONTEXT_ON_RETRY=1` (strips RAG immediately)
- **Solution**: Check logs for `ECHO_GUARD_RETRY` metrics to see actual similarity scores
- **Solution**: Try reducing `ECHO_SIMILARITY_THRESHOLD` to 0.90 to catch near-duplicates
- **Solution**: Verify warnings are reaching the LLM (check context window in logs)

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

Built on concepts from:
- Markovian Thinking (VentureBeat, 2024)
- Memory-Augmented Attention Mechanisms
- GraphRAG and hybrid retrieval systems

## 📧 Contact

For questions or issues, please check the documentation or open an issue on GitHub.

---

**VICW** - Virtual Infinite Context Window with Hybrid Memory Architecture
