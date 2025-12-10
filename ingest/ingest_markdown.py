#!/usr/bin/env python3
"""
Markdown Ingestion Script for VICW

Ingests markdown files into the Qdrant vector database for RAG retrieval.
Splits content by scene breaks (---) and processes each chunk through the
semantic manager pipeline.

Usage:
    python ingest/ingest_markdown.py ingest/chapter_1.md --domain creative
    python ingest/ingest_markdown.py ingest/*.md --domain <domain>
"""

import sys
import os
import asyncio
import argparse
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from data_models import OffloadJob
from semantic_manager import SemanticManager
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from llm_inference import LLMClient
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_MODEL_TYPE, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_CTX,
    COLD_PATH_WORKERS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_embedding_model():
    """Load the embedding model based on configuration"""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_TYPE}")

    if EMBEDDING_MODEL_TYPE == 'llama_cpp':
        from llama_cpp import Llama
        model = Llama(
            model_path=EMBEDDING_MODEL_PATH,
            embedding=True,
            n_ctx=EMBEDDING_MODEL_CTX,
            n_batch=512,
            verbose=False
        )
        logger.info(f"Loaded llama.cpp model from {EMBEDDING_MODEL_PATH} (n_ctx={EMBEDDING_MODEL_CTX})")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logger.info(f"Loaded SentenceTransformer model")

    return model


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic"""
    return int(len(text.split()) / 0.75)


def split_markdown_into_chunks(content: str, min_chunk_size: int = 500) -> list[dict]:
    """
    Split markdown content into chunks by scene breaks (---).
    Each chunk includes its section header if available.

    Returns list of dicts with 'text' and 'metadata' keys.
    """
    chunks = []

    # Split by scene breaks (---)
    sections = re.split(r'\n---+\n', content)

    current_chunk = ""
    current_headers = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract headers from section
        header_match = re.search(r'^####?\s*\|?\s*([^|]+)', section, re.MULTILINE)
        if header_match:
            current_headers = [header_match.group(1).strip()]

        # If adding this section would make chunk too small, accumulate
        if len(current_chunk) + len(section) < min_chunk_size:
            current_chunk += "\n\n" + section if current_chunk else section
        else:
            # Save current chunk if exists
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {'headers': current_headers.copy()}
                })

            # Start new chunk
            current_chunk = section
            if header_match:
                current_headers = [header_match.group(1).strip()]

    # Don't forget last chunk
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'metadata': {'headers': current_headers.copy()}
        })

    return chunks


async def ingest_file(
    filepath: Path,
    semantic_manager: SemanticManager,
    domain: str = "creative",
    thread_id: str = None
) -> int:
    """
    Ingest a single markdown file into the vector database.

    Returns: Number of chunks processed
    """
    logger.info(f"Ingesting file: {filepath}")

    # Read file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into chunks
    chunks = split_markdown_into_chunks(content)
    logger.info(f"Split into {len(chunks)} chunks")

    # Generate thread_id from filename if not provided
    if not thread_id:
        thread_id = f"ingest_{filepath.stem}"

    processed = 0
    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data['text']
        metadata = chunk_data['metadata']

        # Estimate tokens
        token_count = estimate_tokens(chunk_text)

        # Create OffloadJob
        job = OffloadJob.create(
            chunk_text=chunk_text,
            token_count=token_count,
            message_count=1,
            metadata={
                'domain': domain,
                'thread_id': thread_id,
                'source': str(filepath),
                'chunk_index': i,
                'total_chunks': len(chunks),
                **metadata
            }
        )

        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({token_count} tokens)")

        # Process through semantic manager
        result = await semantic_manager.process_job(job)

        if result and result.success:
            processed += 1
            logger.info(f"  ✓ Chunk {i+1} processed: {result.summary}")
        else:
            error_msg = result.error if result else "Unknown error"
            logger.error(f"  ✗ Chunk {i+1} failed: {error_msg}")

    return processed


async def main():
    parser = argparse.ArgumentParser(description='Ingest markdown files into VICW vector database')
    parser.add_argument('files', nargs='+', help='Markdown files to ingest')
    parser.add_argument('--domain', default='creative', help='Domain tag (default: creative)')
    parser.add_argument('--thread-id', help='Thread ID for flow tracking')
    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing components...")

    # Load embedding model
    embedding_model = load_embedding_model()

    # Initialize storage backends
    redis_storage = RedisStorage(REDIS_HOST, REDIS_PORT, REDIS_DB)
    qdrant_db = QdrantVectorDB(QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION)
    neo4j_graph = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Initialize LLM client for entity extraction
    llm_client = LLMClient()

    # Create thread pool for CPU-bound operations
    executor = ThreadPoolExecutor(max_workers=COLD_PATH_WORKERS, thread_name_prefix='ingest')

    # Initialize semantic manager
    semantic_manager = SemanticManager(
        embedding_model=embedding_model,
        redis_storage=redis_storage,
        qdrant_db=qdrant_db,
        neo4j_graph=neo4j_graph,
        llm_client=llm_client,
        executor=executor
    )

    # Process each file
    total_processed = 0
    total_files = 0

    for file_pattern in args.files:
        # Handle glob patterns
        if '*' in file_pattern:
            from glob import glob
            files = [Path(f) for f in glob(file_pattern)]
        else:
            files = [Path(file_pattern)]

        for filepath in files:
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                continue

            if not filepath.suffix.lower() in ['.md', '.markdown', '.txt']:
                logger.warning(f"Skipping non-markdown file: {filepath}")
                continue

            try:
                processed = await ingest_file(
                    filepath,
                    semantic_manager,
                    domain=args.domain,
                    thread_id=args.thread_id
                )
                total_processed += processed
                total_files += 1
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}", exc_info=True)

    # Cleanup
    executor.shutdown(wait=True)
    neo4j_graph.close()

    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Chunks ingested: {total_processed}")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
