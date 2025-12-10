# VICW - Virtual Infinite Context Window

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**VICW** is a production-ready system that provides virtually unlimited conversation context for Large Language Models (LLMs) through intelligent context management, multi-tier storage, and semantic retrieval.

## Overview

Traditional LLM conversations are constrained by fixed context windows (typically 4K-128K tokens). VICW solves this by implementing a multi-layered memory architecture that automatically manages context pressure, offloads old messages to persistent storage, and retrieves relevant information when needed.

### Key Features

- **Virtual Infinite Context**: Automatic offloading and retrieval of conversation history
- **Multi-Database Architecture**:
  - Redis for fast chunk storage
  - Qdrant for semantic vector search
  - Neo4j for knowledge graph relationships
- **RAG (Retrieval Augmented Generation)**: Semantic retrieval of relevant past context
- **State Tracking**: Automatic extraction and tracking of goals, tasks, decisions, and facts
- **Echo Guard**: Prevents repetitive responses through similarity detection
- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API
- **Document Ingestion**: Direct document embedding endpoint for knowledge bases
- **Production Ready**: Docker-based deployment with health checks and monitoring

## Architecture

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Context Manager               │
│  • Token counting               │
│  • Pressure monitoring (80%)    │
│  • Message deduplication        │
└────────┬────────────────────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────────┐
│  LLM   │  │ Offload Queue│
└────────┘  └──────┬───────┘
                   │
                   ▼
            ┌──────────────────┐
            │ Cold Path Worker │
            │  • Embedding      │
            │  • State Extract  │
            └──────┬───────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌──────┐  ┌────────┐  ┌──────┐
    │Redis │  │Qdrant  │  │Neo4j │
    │Chunks│  │Vectors │  │Graph │
    └──────┘  └────────┘  └──────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key (or compatible LLM API)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/elevend0g/vicw
cd vicw
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Configure your LLM API key in `.env`:
```bash
VICW_LLM_API_KEY=your_api_key_here
VICW_LLM_API_URL=https://api.openrouter.ai/api/v1/chat/completions
VICW_LLM_MODEL_NAME=mistralai/mistral-7b-instruct
```

4. Start the services:
```bash
docker-compose up -d
```

5. Verify the system is running:
```bash
curl http://localhost:8000/health
```

## Usage

### API Server Mode

The API server provides both custom and OpenAI-compatible endpoints.

#### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Tell me about the solar system.",
    "use_rag": true
  }'
```

#### OpenAI-Compatible Endpoint
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicw-mistralai/mistral-7b-instruct",
    "messages": [
      {"role": "user", "content": "What is VICW?"}
    ],
    "stream": false
  }'
```

#### Document Ingestion
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Large document content here...",
    "metadata": {"source": "documentation", "topic": "architecture"}
  }'
```

### CLI Mode

Run VICW in interactive CLI mode:

```bash
docker-compose exec vicw_api python app/main.py
```

Commands:
- `stats` - View system statistics
- `exit` - Quit the session

### Integration with OpenWebUI

VICW can be used as a custom model in OpenWebUI:

1. Add VICW as a custom model in OpenWebUI settings
2. Set the API endpoint to `http://vicw_api:8000/v1`
3. Use model name: `vicw-mistralai/mistral-7b-instruct`

See [OPENWEBUI_SETUP.md](documentation/OPENWEBUI_SETUP.md) for detailed instructions.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONTEXT_TOKENS` | 4096 | Maximum context window size |
| `OFFLOAD_THRESHOLD` | 0.80 | Trigger offload at 80% capacity |
| `RAG_SCORE_THRESHOLD` | 0.4 | Minimum similarity for retrieval (0.0-1.0) |
| `ECHO_GUARD_ENABLED` | true | Enable duplicate response detection |
| `ECHO_SIMILARITY_THRESHOLD` | 0.95 | Similarity threshold for echo detection |
| `STATE_TRACKING_ENABLED` | true | Enable automatic state extraction |
| `EMBEDDING_MODEL_TYPE` | llama_cpp | Embedding model type (llama_cpp or sentence_transformer) |
| `EMBEDDING_MODEL_PATH` | models/snowflake-arctic-embed-l-v2.0-q8_0.gguf | Path to embedding model |

### Database Configuration

- **Redis**: Stores compressed conversation chunks with 24-hour TTL
- **Qdrant**: Vector database for semantic search with configurable collection
- **Neo4j**: Knowledge graph for entity relationships and state tracking

## How It Works

### Context Management

1. **Monitoring**: Context manager continuously monitors token count
2. **Pressure Detection**: At 80% capacity, triggers offload process
3. **Offloading**: Oldest messages queued for background processing
4. **Relief**: Context reduced to 60%, maintaining recent conversation
5. **Hysteresis**: Won't re-trigger until 70% to prevent thrashing

### Semantic Retrieval (RAG)

1. User query generates an embedding
2. Qdrant searches for semantically similar past chunks
3. Neo4j retrieves related state information (goals, tasks, facts)
4. Relevant memories injected into context before generation
5. LLM generates response with full context awareness

### Echo Guard

1. Each response embedding stored in rotating history (10 most recent)
2. Before accepting response, checks similarity with recent outputs
3. If similarity > 95%, regenerates with escalating warnings
4. After 3 attempts, strips RAG context and forces acknowledgment
5. Prevents infinite loops and repetitive responses

### State Tracking

Automatically extracts and tracks:
- **Goals**: User objectives and desired outcomes
- **Tasks**: Action items and to-dos
- **Decisions**: Key choices and conclusions
- **Facts**: Important information and data points

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send chat message with custom response |
| `/ingest` | POST | Ingest document for background embedding |
| `/stats` | GET | Get system statistics |
| `/health` | GET | Health check |
| `/reset` | POST | Reset conversation context |
| `/v1/models` | GET | List available models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |

### Response Format

```json
{
  "response": "Assistant response text",
  "timestamp": "2025-12-10T12:34:56",
  "tokens_in_context": 1234,
  "rag_items_injected": 5
}
```

## Monitoring

### Statistics Endpoint

```bash
curl http://localhost:8000/stats
```

Returns:
- Context tokens and pressure percentage
- Offload queue size and processed count
- Cold path worker statistics
- Qdrant collection info

### Logs

- Application logs: `docker-compose logs -f vicw_api`
- Metrics log: `logs/vicw_metrics.log`

## Development

### Project Structure

```
vicw/
├── app/
│   ├── main.py              # CLI entry point
│   ├── api_server.py        # FastAPI server
│   ├── config.py            # Configuration
│   ├── data_models.py       # Pydantic models
│   ├── context_manager.py   # Context management
│   ├── semantic_manager.py  # RAG and retrieval
│   ├── offload_queue.py     # Queue management
│   ├── cold_path_worker.py  # Background processing
│   ├── llm_inference.py     # External LLM client
│   ├── redis_storage.py     # Redis client
│   ├── qdrant_vector_db.py  # Qdrant client
│   ├── neo4j_knowledge_graph.py # Neo4j client
│   └── state_extractor.py   # State tracking
├── docker-compose.yml       # Service orchestration
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── system_prompt.txt       # System instructions
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start databases:
```bash
docker-compose up -d redis qdrant neo4j
```

3. Run API server:
```bash
python app/api_server.py
```

4. Run CLI:
```bash
python app/main.py
```

## Performance

- **Context Offload**: < 50ms (async, non-blocking)
- **RAG Retrieval**: 100-300ms (depending on database size)
- **LLM Generation**: Depends on external API
- **Cold Path Processing**: Background, no impact on latency

## Limitations

- Single-user conversations (session-based context management)
- Requires external LLM API (OpenRouter, OpenAI, etc.)
- Memory usage scales with conversation size (mitigated by Redis TTL)
- Streaming responses simulated (chunks generated response)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Embeddings powered by [Sentence Transformers](https://www.sbert.net/) and [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Vector search by [Qdrant](https://qdrant.tech/)
- Knowledge graphs with [Neo4j](https://neo4j.com/)
- Storage with [Redis](https://redis.io/)

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
