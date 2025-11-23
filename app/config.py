"""Configuration constants for VICW Phase 2"""

import os

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_TITLE = "VICW Phase 2 API"
API_VERSION = "2.0.0"

# External LLM Configuration
EXTERNAL_API_URL = os.getenv('VICW_LLM_API_URL', 'https://api.openrouter.ai/api/v1/chat/completions')
EXTERNAL_API_KEY = os.getenv('VICW_LLM_API_KEY', '')
EXTERNAL_MODEL_NAME = os.getenv('VICW_LLM_MODEL_NAME', 'mistralai/mistral-7b-instruct')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '90'))  # seconds
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '500'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))

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
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))  # all-MiniLM-L6-v2

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

# Thread Limits (Must be set before imports)
THREAD_CONFIG = {
    'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS', '2'),
    'MKL_NUM_THREADS': os.getenv('MKL_NUM_THREADS', '2'),
    'OPENBLAS_NUM_THREADS': os.getenv('OPENBLAS_NUM_THREADS', '2'),
    'NUMEXPR_NUM_THREADS': os.getenv('NUMEXPR_NUM_THREADS', '2')
}

# Cold Path Configuration
COLD_PATH_BATCH_SIZE = int(os.getenv('COLD_PATH_BATCH_SIZE', '3'))
COLD_PATH_WORKERS = int(os.getenv('COLD_PATH_WORKERS', '4'))
MAX_OFFLOAD_QUEUE_SIZE = int(os.getenv('MAX_OFFLOAD_QUEUE_SIZE', '100'))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
METRICS_LOG_FILE = os.getenv('METRICS_LOG_FILE', 'vicw_metrics.log')

# RAG Configuration
RAG_TOP_K_SEMANTIC = int(os.getenv('RAG_TOP_K_SEMANTIC', '2'))
RAG_TOP_K_RELATIONAL = int(os.getenv('RAG_TOP_K_RELATIONAL', '5'))

# State Tracking Configuration
STATE_TRACKING_ENABLED = os.getenv('STATE_TRACKING_ENABLED', 'true').lower() == 'true'
STATE_CONFIG_PATH = os.getenv('STATE_CONFIG_PATH', 'app/state_config.yaml')
STATE_INJECTION_LIMITS = {
    'goal': int(os.getenv('STATE_LIMIT_GOAL', '2')),
    'task': int(os.getenv('STATE_LIMIT_TASK', '3')),
    'decision': int(os.getenv('STATE_LIMIT_DECISION', '2')),
    'fact': int(os.getenv('STATE_LIMIT_FACT', '3'))
}


def apply_thread_config():
    """Apply thread configuration to environment variables before heavy imports"""
    for key, value in THREAD_CONFIG.items():
        os.environ[key] = value
