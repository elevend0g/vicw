"""Configuration constants for VICW"""

import os
import json

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_TITLE = "VICW API"
API_VERSION = "2.0.0"

# External LLM Configuration
EXTERNAL_API_URL = os.getenv('VICW_LLM_API_URL', 'https://api.openrouter.ai/api/v1/chat/completions')
EXTERNAL_API_KEY = os.getenv('VICW_LLM_API_KEY', '')
EXTERNAL_MODEL_NAME = os.getenv('VICW_LLM_MODEL_NAME', 'mistralai/mistral-7b-instruct')
VICW_BRANDED_MODEL_NAME = f"vicw-{EXTERNAL_MODEL_NAME}"
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '90'))  # seconds
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '500'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))

# Response format - parse JSON string from env var
_response_format_str = os.getenv('LLM_RESPONSE_FORMAT', '{"type": "text"}')
try:
    LLM_RESPONSE_FORMAT = json.loads(_response_format_str)
except json.JSONDecodeError:
    LLM_RESPONSE_FORMAT = {"type": "text"}

# Context Configuration
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', '4096'))
OFFLOAD_THRESHOLD = float(os.getenv('OFFLOAD_THRESHOLD', '0.80'))  # Trigger at 80%
TARGET_AFTER_RELIEF = float(os.getenv('TARGET_AFTER_RELIEF', '0.60'))  # Drop to 60%
HYSTERESIS_THRESHOLD = float(os.getenv('HYSTERESIS_THRESHOLD', '0.70'))  # Don't re-trigger until 70%

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_CHUNK_TTL = int(os.getenv('REDIS_CHUNK_TTL', '86400'))  # 24 hours

# Qdrant Configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'vicw_memory')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1024'))  # Qwen3-Embedding-0.6B

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Embedding Model Configuration
EMBEDDING_MODEL_TYPE = os.getenv('EMBEDDING_MODEL_TYPE', 'llama_cpp') # 'sentence_transformer' or 'llama_cpp'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'snowflake-arctic-embed-l-v2.0-q8_0.gguf')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', 'models/snowflake-arctic-embed-l-v2.0-q8_0.gguf')
EMBEDDING_MODEL_CTX = int(os.getenv('EMBEDDING_MODEL_CTX', '8192'))  # Full context for Snowflake Arctic (8192 train ctx)

# Cold Path Configuration
COLD_PATH_BATCH_SIZE = int(os.getenv('COLD_PATH_BATCH_SIZE', '3'))
COLD_PATH_WORKERS = int(os.getenv('COLD_PATH_WORKERS', '4'))
MAX_OFFLOAD_QUEUE_SIZE = int(os.getenv('MAX_OFFLOAD_QUEUE_SIZE', '100'))

# Proactive Embedding Configuration
# Enable eager embedding of large messages in background (even below pressure threshold)
PROACTIVE_EMBED_ENABLED = os.getenv('PROACTIVE_EMBED_ENABLED', 'true').lower() == 'true'
PROACTIVE_EMBED_THRESHOLD = int(os.getenv('PROACTIVE_EMBED_THRESHOLD', '500'))  # Tokens threshold for proactive embedding

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
METRICS_LOG_FILE = os.getenv('METRICS_LOG_FILE', 'vicw_metrics.log')

# RAG Configuration
RAG_TOP_K_SEMANTIC = int(os.getenv('RAG_TOP_K_SEMANTIC', '10'))
RAG_TOP_K_RELATIONAL = int(os.getenv('RAG_TOP_K_RELATIONAL', '5'))

# RAG Score Threshold - Minimum cosine similarity for Qdrant results
# Range: 0.0-1.0 (cosine similarity)
# Recommended: 0.3-0.5 for general use, 0.5-0.7 for high precision
RAG_SCORE_THRESHOLD = float(os.getenv('RAG_SCORE_THRESHOLD', '0.4'))

# State Tracking Configuration
STATE_TRACKING_ENABLED = os.getenv('STATE_TRACKING_ENABLED', 'true').lower() == 'true'
STATE_CONFIG_PATH = os.getenv('STATE_CONFIG_PATH', 'app/state_config.yaml')
STATE_INJECTION_LIMITS = {
    'goal': int(os.getenv('STATE_LIMIT_GOAL', '2')),
    'task': int(os.getenv('STATE_LIMIT_TASK', '3')),
    'decision': int(os.getenv('STATE_LIMIT_DECISION', '2')),
    'fact': int(os.getenv('STATE_LIMIT_FACT', '3'))
}

# Boredom Detection Configuration (Loop Prevention)
BOREDOM_DETECTION_ENABLED = os.getenv('BOREDOM_DETECTION_ENABLED', 'true').lower() == 'true'
BOREDOM_THRESHOLD = int(os.getenv('BOREDOM_THRESHOLD', '5'))  # Visit count threshold
BOREDOM_ALTERNATIVE_COUNT = int(os.getenv('BOREDOM_ALTERNATIVE_COUNT', '3'))  # Alternative suggestions

# Echo Guard Configuration (Response Similarity Detection)
ECHO_GUARD_ENABLED = os.getenv('ECHO_GUARD_ENABLED', 'true').lower() == 'true'
ECHO_SIMILARITY_THRESHOLD = float(os.getenv('ECHO_SIMILARITY_THRESHOLD', '0.95'))  # Cosine similarity threshold
ECHO_RESPONSE_HISTORY_SIZE = int(os.getenv('ECHO_RESPONSE_HISTORY_SIZE', '10'))  # Number of recent responses to compare
MAX_REGENERATION_ATTEMPTS = int(os.getenv('MAX_REGENERATION_ATTEMPTS', '3'))  # Max retries on duplicate detection
ECHO_STRIP_CONTEXT_ON_RETRY = int(os.getenv('ECHO_STRIP_CONTEXT_ON_RETRY', '3'))  # Which retry to strip RAG context (1-3, default: 3)
