"""FastAPI server for VICW Phase 2"""

import os
import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Set thread limits BEFORE imports
from config import (
    apply_thread_config,
    API_HOST,
    API_PORT,
    API_TITLE,
    API_VERSION,
    MAX_CONTEXT_TOKENS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EXTERNAL_API_URL,
    EXTERNAL_API_KEY,
    EXTERNAL_MODEL_NAME,
    VICW_BRANDED_MODEL_NAME,
    LLM_TIMEOUT,
    EMBEDDING_MODEL_NAME,
    ECHO_GUARD_ENABLED,
    ECHO_SIMILARITY_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS,
    ECHO_STRIP_CONTEXT_ON_RETRY
)

apply_thread_config()

# Now import heavy libraries
from sentence_transformers import SentenceTransformer

from context_manager import ContextManager
from offload_queue import OffloadQueue
from cold_path_worker import ColdPathWorker
from semantic_manager import SemanticManager
from llm_inference import ExternalLLMInference
from redis_storage import RedisStorage
from qdrant_vector_db import QdrantVectorDB
from neo4j_knowledge_graph import Neo4jKnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title=API_TITLE, version=API_VERSION)

# Global components
context_manager: Optional[ContextManager] = None
llm: Optional[ExternalLLMInference] = None
cold_path_worker: Optional[ColdPathWorker] = None
offload_queue: Optional[OffloadQueue] = None
redis_storage: Optional[RedisStorage] = None
qdrant_db: Optional[QdrantVectorDB] = None
neo4j_graph: Optional[Neo4jKnowledgeGraph] = None


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True  # Enable RAG by default


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    tokens_in_context: Optional[int] = None
    rag_items_injected: Optional[int] = 0


# ============================================================================
# OpenAI-Compatible API Models
# ============================================================================

class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=1)  # VICW only supports 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[Any] = None  # Can be string or list of strings


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoiceMessage(BaseModel):
    role: str
    content: str


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIChoiceMessage
    finish_reason: str


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


class OpenAIStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]


@app.on_event("startup")
async def startup_event():
    """Initialize all VICW components on startup"""
    global context_manager, llm, cold_path_worker, offload_queue
    global redis_storage, qdrant_db, neo4j_graph
    
    logger.info("=" * 60)
    logger.info("Starting VICW Phase 2 API Server")
    logger.info("=" * 60)
    
    try:
        # Initialize Redis storage
        logger.info("Initializing Redis storage...")
        redis_storage = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        await redis_storage.init()
        
        # Initialize Qdrant vector database
        logger.info("Initializing Qdrant vector database...")
        qdrant_db = QdrantVectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION,
            dimension=EMBEDDING_DIMENSION
        )
        await qdrant_db.init()
        
        # Initialize Neo4j knowledge graph
        logger.info("Initializing Neo4j knowledge graph...")
        neo4j_graph = Neo4jKnowledgeGraph(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        await neo4j_graph.init()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize offload queue
        offload_queue = OffloadQueue()
        
        # Initialize semantic manager
        logger.info("Initializing semantic manager...")
        semantic_manager = SemanticManager(
            embedding_model=embedding_model,
            redis_storage=redis_storage,
            qdrant_db=qdrant_db,
            neo4j_graph=neo4j_graph
        )
        
        # Initialize context manager
        logger.info("Initializing context manager...")
        context_manager = ContextManager(
            max_context=MAX_CONTEXT_TOKENS,
            offload_queue=offload_queue,
            embedding_model=embedding_model,
            semantic_manager=semantic_manager
        )
        
        # Initialize external LLM
        logger.info(f"Initializing external LLM: {EXTERNAL_MODEL_NAME}...")
        if not EXTERNAL_API_KEY:
            raise ValueError("VICW_LLM_API_KEY environment variable must be set")
        
        llm = ExternalLLMInference(
            api_url=EXTERNAL_API_URL,
            api_key=EXTERNAL_API_KEY,
            model_name=EXTERNAL_MODEL_NAME
        )
        await llm.init()
        
        # Initialize and start cold path worker
        logger.info("Starting cold path worker...")
        cold_path_worker = ColdPathWorker(offload_queue, semantic_manager)
        await cold_path_worker.start()
        
        # Load system prompt if available
        system_prompt_path = Path("system_prompt.txt")
        if system_prompt_path.exists():
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read().strip()
            await context_manager.add_message("system", system_prompt)
            logger.info("System prompt loaded")
        
        logger.info("=" * 60)
        logger.info("VICW Phase 2 API Server ready!")
        logger.info(f"LLM: {EXTERNAL_MODEL_NAME}")
        logger.info(f"Max context: {MAX_CONTEXT_TOKENS} tokens")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize VICW system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global context_manager, llm, cold_path_worker, redis_storage, qdrant_db, neo4j_graph
    
    logger.info("Shutting down VICW Phase 2 API Server...")
    
    if cold_path_worker:
        await cold_path_worker.shutdown()
    
    if llm:
        await llm.shutdown()
    
    if redis_storage:
        await redis_storage.shutdown()
    
    if qdrant_db:
        await qdrant_db.shutdown()
    
    if neo4j_graph:
        await neo4j_graph.close()
    
    logger.info("VICW Phase 2 API Server shutdown complete")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages with optional RAG"""
    global context_manager, llm, cold_path_worker
    
    if not context_manager or not llm:
        raise HTTPException(status_code=503, detail="VICW system not initialized")
    
    try:
        # Add user message
        await context_manager.add_message("user", request.message)
        
        # Perform RAG if enabled
        rag_items = 0
        if request.use_rag and context_manager.semantic_manager:
            rag_items = await context_manager.augment_context_with_memory(request.message)
        
        # Pause cold path during LLM generation to avoid resource contention
        if cold_path_worker:
            await cold_path_worker.pause()

        # Get context window
        context_window = context_manager.get_context_window()

        # Echo Guard: Generate response with duplicate detection and regeneration
        response_text = None
        response_embedding = None
        regeneration_count = 0
        is_repeated = False

        while regeneration_count < MAX_REGENERATION_ATTEMPTS:
            # Generate response with timeout
            try:
                current_response = await asyncio.wait_for(
                    llm.generate(context_window),
                    timeout=LLM_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM generation timeout after {LLM_TIMEOUT}s")
                raise HTTPException(status_code=504, detail="LLM generation timeout")

            # Handle empty/whitespace-only responses (failure mode)
            if not current_response or not current_response.strip():
                regeneration_count += 1
                logger.error(
                    f"LLM generated empty response (attempt {regeneration_count}/{MAX_REGENERATION_ATTEMPTS})"
                )

                if regeneration_count < MAX_REGENERATION_ATTEMPTS:
                    from data_models import Message
                    empty_warning = Message(
                        role="system",
                        content=(
                            "⚠️ ERROR: You generated an empty response.\n\n"
                            "You MUST provide a substantive response. Options:\n"
                            "1. Answer the user's question with available information\n"
                            "2. State clearly: 'I don't have enough information to answer this'\n"
                            "3. Ask for clarification if the request is unclear\n\n"
                            "Empty responses are not acceptable. Respond now with actual content."
                        )
                    )
                    context_window.append(empty_warning.to_dict())
                    continue
                else:
                    # Max retries with empty responses
                    logger.error("Max retries reached with empty responses, returning error message")
                    response_text = "[ERROR] The LLM failed to generate a response after multiple attempts. Please rephrase your question or try again."
                    break

            # Check for echo (duplicate response) if enabled
            if ECHO_GUARD_ENABLED and context_manager.semantic_manager:
                # Generate embedding for response
                response_embedding = await context_manager.semantic_manager.generate_embedding(current_response)

                if response_embedding:
                    # Check similarity with recent responses
                    is_duplicate, similarity = await context_manager.semantic_manager.check_response_similarity(
                        response_embedding,
                        threshold=ECHO_SIMILARITY_THRESHOLD
                    )

                    if is_duplicate:
                        regeneration_count += 1
                        logger.warning(
                            f"Echo detected (attempt {regeneration_count}/{MAX_REGENERATION_ATTEMPTS}): "
                            f"similarity={similarity:.4f}, response_length={len(current_response)}"
                        )

                        # Log metrics for monitoring
                        metrics_logger = logging.getLogger('vicw.metrics')
                        metrics_logger.info(
                            f"ECHO_GUARD_RETRY | attempt={regeneration_count} | "
                            f"similarity={similarity:.4f} | response_len={len(current_response)}"
                        )

                        if regeneration_count < MAX_REGENERATION_ATTEMPTS:
                            # Escalating warnings based on retry attempt
                            from data_models import Message

                            if regeneration_count == 1:
                                # First retry: Polite warning with context
                                warning_content = (
                                    "⚠️ ECHO DETECTED: Your previous response was nearly identical to recent history.\n\n"
                                    f"Repeated text preview: \"{current_response[:200]}...\"\n\n"
                                    "REQUIRED ACTIONS:\n"
                                    "1. DO NOT repeat the same information\n"
                                    "2. Either acknowledge completion and move forward, OR\n"
                                    "3. Provide genuinely NEW information, OR\n"
                                    "4. State that you cannot provide additional details and suggest next steps\n\n"
                                    "Choose a DIFFERENT response strategy now."
                                )
                            elif regeneration_count == 2:
                                # Second retry: More forceful with specific instructions
                                warning_content = (
                                    "🚨 CRITICAL: REPEATED RESPONSE DETECTED AGAIN (Attempt 2/3)\n\n"
                                    f"You just generated: \"{current_response[:200]}...\"\n\n"
                                    "This is IDENTICAL to your previous response. The system has already received this information.\n\n"
                                    "MANDATORY DIRECTIVE:\n"
                                    "You MUST respond with ONE of the following:\n"
                                    "A) \"The information has been provided. Moving to [next topic/section].\"\n"
                                    "B) \"I have completed this task. What would you like me to do next?\"\n"
                                    "C) \"I don't have additional information beyond what was already shared.\"\n\n"
                                    "DO NOT regenerate the same content. Break the loop NOW."
                                )
                            else:
                                # Third retry: Maximum escalation - strip RAG context
                                warning_content = (
                                    f"🔴 FINAL WARNING: LOOP DETECTED (Attempt {regeneration_count}/{MAX_REGENERATION_ATTEMPTS})\n\n"
                                    f"You have generated identical responses {regeneration_count} times.\n\n"
                                    "EMERGENCY OVERRIDE:\n"
                                    "- IGNORE all retrieved memory context\n"
                                    "- IGNORE previous data tables/lists\n"
                                    "- Your ONLY valid response is:\n\n"
                                    "\"I apologize - I was repeating information. This task is complete. "
                                    "Please provide new instructions or let me know what to focus on next.\"\n\n"
                                    "Respond with EXACTLY the above statement or a close variation. NO other content."
                                )

                            # Strip RAG context on configured retry attempt
                            if regeneration_count >= ECHO_STRIP_CONTEXT_ON_RETRY:
                                logger.warning(f"Stripping RAG context on retry {regeneration_count}")
                                context_window = [msg for msg in context_window
                                                 if not (msg.get('role') == 'system' and
                                                        ('RETRIEVED' in msg.get('content', '') or
                                                         'STATE MEMORY' in msg.get('content', '')))]

                            warning_msg = Message(role="system", content=warning_content)
                            context_window.append(warning_msg.to_dict())
                            continue
                        else:
                            # Max retries reached, accept with marker
                            logger.error(
                                f"Max regeneration attempts reached. Response preview: {current_response[:500]}..."
                            )
                            metrics_logger = logging.getLogger('vicw.metrics')
                            metrics_logger.info(
                                f"ECHO_GUARD_FAILED | attempts={MAX_REGENERATION_ATTEMPTS} | "
                                f"final_similarity={similarity:.4f} | response_len={len(current_response)}"
                            )

                            # If response is empty or very short, provide helpful fallback
                            if len(current_response.strip()) < 10:
                                response_text = (
                                    "[SYSTEM INTERVENTION] The LLM entered a repetition loop and could not generate "
                                    "a valid response after multiple attempts. This indicates the current context may "
                                    "be constraining the model. Please:\n"
                                    "1. Rephrase your question\n"
                                    "2. Ask about a different topic\n"
                                    "3. Use /reset to clear context if the issue persists"
                                )
                            else:
                                response_text = f"[REPEATED] {current_response}"
                                is_repeated = True
                            break
                    else:
                        # Not a duplicate, accept response
                        response_text = current_response
                        break
                else:
                    # Failed to generate embedding, accept response without check
                    logger.warning("Failed to generate response embedding, skipping echo detection")
                    response_text = current_response
                    break
            else:
                # Echo guard disabled, accept response
                response_text = current_response
                break

        # Store response embedding for future comparisons
        if ECHO_GUARD_ENABLED and response_embedding and context_manager.semantic_manager:
            await context_manager.semantic_manager.store_response_embedding(response_embedding)

        # Resume cold path
        if cold_path_worker:
            await cold_path_worker.resume()

        # Add assistant response
        await context_manager.add_message("assistant", response_text)
        
        # Get token count
        token_count = context_manager._token_count()
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            tokens_in_context=token_count,
            rag_items_injected=rag_items
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "VICW Phase 2",
        "model": VICW_BRANDED_MODEL_NAME,
        "context_initialized": context_manager is not None,
        "llm_initialized": llm is not None
    }


@app.get("/stats")
async def stats():
    """Get system statistics"""
    global context_manager, offload_queue, cold_path_worker, qdrant_db
    
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    stats_data = {
        "context": context_manager.get_stats(),
        "queue": offload_queue.get_stats() if offload_queue else {},
        "worker": cold_path_worker.get_stats() if cold_path_worker else {}
    }
    
    # Add Qdrant stats if available
    if qdrant_db:
        try:
            qdrant_info = await qdrant_db.get_collection_info()
            stats_data["qdrant"] = qdrant_info
        except Exception as e:
            logger.warning(f"Failed to get Qdrant stats: {e}")
    
    return stats_data


@app.post("/reset")
async def reset_context():
    """Reset the context (useful for testing)"""
    global context_manager

    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")

    # Create new context manager
    context_manager.working_context = []
    context_manager.offload_job_count = 0
    context_manager.placeholder_markers = {}
    context_manager.last_relief_tokens = 0

    logger.info("Context reset")

    return {"status": "success", "message": "Context reset"}


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================

@app.get("/v1/models", response_model=OpenAIModelList)
async def list_models():
    """List available models (OpenAI-compatible endpoint)"""
    return OpenAIModelList(
        data=[
            OpenAIModel(
                id=VICW_BRANDED_MODEL_NAME,
                created=int(time.time()),
                owned_by="vicw"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming and non-streaming responses.
    """
    global context_manager, llm, cold_path_worker

    # DEBUG: Log incoming request parameters
    logger.info("=" * 60)
    logger.info("OpenWebUI Request Received:")
    logger.info(f"  model: {request.model}")
    logger.info(f"  response_format: {request.response_format}")
    logger.info(f"  stop: {request.stop}")
    logger.info(f"  temperature: {request.temperature}")
    logger.info(f"  stream: {request.stream}")
    logger.info(f"  messages count: {len(request.messages)}")
    logger.info("=" * 60)

    if not context_manager or not llm:
        raise HTTPException(status_code=503, detail="VICW system not initialized")

    try:
        # Generate a unique ID for this completion
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created_time = int(time.time())

        # Extract the last user message for RAG query
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        last_user_message = user_messages[-1].content

        # Add all messages to context (clear and rebuild from OpenAI format)
        # Note: For production, you may want to implement proper conversation tracking
        # For now, we'll just process the last user message
        await context_manager.add_message("user", last_user_message)

        # Perform RAG (always enabled for OpenAI endpoint)
        rag_items = 0
        if context_manager.semantic_manager:
            rag_items = await context_manager.augment_context_with_memory(last_user_message)

        # Pause cold path during generation
        if cold_path_worker:
            await cold_path_worker.pause()

        # Get context window
        context_window = context_manager.get_context_window()

        # Handle streaming vs non-streaming
        if request.stream:
            # Streaming response
            async def generate_stream():
                """Generate SSE stream for OpenAI-compatible streaming"""
                try:
                    # In streaming mode, we'll generate the full response first
                    # then stream it token-by-token (VICW doesn't support true streaming yet)
                    response_text = None
                    response_embedding = None
                    regeneration_count = 0

                    # Echo Guard loop (same as non-streaming)
                    while regeneration_count < MAX_REGENERATION_ATTEMPTS:
                        try:
                            current_response = await asyncio.wait_for(
                                llm.generate(
                                    context_window,
                                    response_format=request.response_format,
                                    stop=request.stop
                                ),
                                timeout=LLM_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"LLM generation timeout after {LLM_TIMEOUT}s")
                            # Send error chunk
                            error_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": "[ERROR: Generation timeout]"},
                                    "finish_reason": "error"
                                }]
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # Handle empty responses
                        if not current_response or not current_response.strip():
                            regeneration_count += 1
                            if regeneration_count >= MAX_REGENERATION_ATTEMPTS:
                                current_response = "[ERROR] Failed to generate response"
                                break
                            continue

                        # Check for echo if enabled
                        if ECHO_GUARD_ENABLED and context_manager.semantic_manager:
                            response_embedding = await context_manager.semantic_manager.generate_embedding(current_response)

                            if response_embedding:
                                is_duplicate, similarity = await context_manager.semantic_manager.check_response_similarity(
                                    response_embedding,
                                    threshold=ECHO_SIMILARITY_THRESHOLD
                                )

                                if is_duplicate:
                                    regeneration_count += 1
                                    if regeneration_count >= MAX_REGENERATION_ATTEMPTS:
                                        current_response = f"[REPEATED] {current_response}"
                                        break
                                    continue

                        response_text = current_response
                        break

                    if response_text is None:
                        response_text = "[ERROR] Failed to generate response"

                    # Store response embedding
                    if ECHO_GUARD_ENABLED and response_embedding and context_manager.semantic_manager:
                        await context_manager.semantic_manager.store_response_embedding(response_embedding)

                    # Send initial chunk with role
                    initial_chunk = OpenAIChatCompletionChunk(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta={"role": "assistant"},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {initial_chunk.model_dump_json()}\n\n"

                    # Stream the response in chunks (simulate token-by-token)
                    # Split by words for smoother streaming
                    words = response_text.split()
                    for i, word in enumerate(words):
                        content = word if i == 0 else f" {word}"
                        chunk = OpenAIChatCompletionChunk(
                            id=completion_id,
                            created=created_time,
                            model=request.model,
                            choices=[
                                OpenAIStreamChoice(
                                    index=0,
                                    delta={"content": content},
                                    finish_reason=None
                                )
                            ]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for smoother streaming

                    # Send final chunk with finish_reason
                    final_chunk = OpenAIChatCompletionChunk(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta={},
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    # Add response to context
                    await context_manager.add_message("assistant", response_text)

                finally:
                    # Resume cold path
                    if cold_path_worker:
                        await cold_path_worker.resume()

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        else:
            # Non-streaming response (same logic as /chat endpoint)
            response_text = None
            response_embedding = None
            regeneration_count = 0

            while regeneration_count < MAX_REGENERATION_ATTEMPTS:
                try:
                    current_response = await asyncio.wait_for(
                        llm.generate(
                            context_window,
                            response_format=request.response_format,
                            stop=request.stop
                        ),
                        timeout=LLM_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"LLM generation timeout after {LLM_TIMEOUT}s")
                    raise HTTPException(status_code=504, detail="LLM generation timeout")

                # Handle empty responses
                if not current_response or not current_response.strip():
                    regeneration_count += 1
                    if regeneration_count < MAX_REGENERATION_ATTEMPTS:
                        from data_models import Message
                        empty_warning = Message(
                            role="system",
                            content=(
                                "⚠️ ERROR: You generated an empty response.\n\n"
                                "You MUST provide a substantive response."
                            )
                        )
                        context_window.append(empty_warning.to_dict())
                        continue
                    else:
                        response_text = "[ERROR] Failed to generate response after multiple attempts"
                        break

                # Check for echo if enabled
                if ECHO_GUARD_ENABLED and context_manager.semantic_manager:
                    response_embedding = await context_manager.semantic_manager.generate_embedding(current_response)

                    if response_embedding:
                        is_duplicate, similarity = await context_manager.semantic_manager.check_response_similarity(
                            response_embedding,
                            threshold=ECHO_SIMILARITY_THRESHOLD
                        )

                        if is_duplicate:
                            regeneration_count += 1
                            logger.warning(
                                f"Echo detected (attempt {regeneration_count}/{MAX_REGENERATION_ATTEMPTS}): "
                                f"similarity={similarity:.4f}"
                            )

                            if regeneration_count < MAX_REGENERATION_ATTEMPTS:
                                from data_models import Message
                                warning_content = (
                                    "⚠️ ECHO DETECTED: Your previous response was nearly identical to recent history.\n"
                                    "Provide a DIFFERENT response now."
                                )

                                if regeneration_count >= ECHO_STRIP_CONTEXT_ON_RETRY:
                                    context_window = [msg for msg in context_window
                                                     if not (msg.get('role') == 'system' and
                                                            ('RETRIEVED' in msg.get('content', '') or
                                                             'STATE MEMORY' in msg.get('content', '')))]

                                warning_msg = Message(role="system", content=warning_content)
                                context_window.append(warning_msg.to_dict())
                                continue
                            else:
                                response_text = f"[REPEATED] {current_response}"
                                break
                        else:
                            response_text = current_response
                            break
                    else:
                        response_text = current_response
                        break
                else:
                    response_text = current_response
                    break

            if response_text is None:
                response_text = "[ERROR] Failed to generate response"

            # Store response embedding
            if ECHO_GUARD_ENABLED and response_embedding and context_manager.semantic_manager:
                await context_manager.semantic_manager.store_response_embedding(response_embedding)

            # Resume cold path
            if cold_path_worker:
                await cold_path_worker.resume()

            # Add response to context
            await context_manager.add_message("assistant", response_text)

            # Get token counts (approximate)
            token_count = context_manager._token_count()
            prompt_tokens = token_count - len(response_text.split())  # Rough estimate
            completion_tokens = len(response_text.split())

            # Build OpenAI-compatible response
            return OpenAIChatCompletionResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIChoiceMessage(
                            role="assistant",
                            content=response_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=max(1, prompt_tokens),
                    completion_tokens=completion_tokens,
                    total_tokens=token_count
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
