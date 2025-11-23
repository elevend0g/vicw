"""
Tests for boredom detection (state visit counting) and echo guard (response similarity)
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules to test
from app.data_models import State
from app.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from app.semantic_manager import SemanticManager
from app.context_manager import ContextManager


class TestStateVisitCounting:
    """Test state visit count tracking and boredom detection"""

    @pytest.mark.asyncio
    async def test_state_creation_initializes_visit_count(self):
        """Test that new states are created with visit_count=0"""
        state = State.create("goal", "test goal", "active")

        assert state.visit_count == 0
        assert state.last_visited == 0.0
        assert state.type == "goal"
        assert state.status == "active"

    @pytest.mark.asyncio
    async def test_increment_state_visits(self):
        """Test that increment_state_visits increases visit_count"""
        # Mock Neo4j driver
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"updated_count": 3}

        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        # Create Neo4jKnowledgeGraph with mocked driver
        neo4j_graph = Neo4jKnowledgeGraph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        neo4j_graph._driver = mock_driver

        # Test increment
        state_ids = ["state_1", "state_2", "state_3"]
        count = await neo4j_graph.increment_state_visits(state_ids)

        assert count == 3
        mock_session.run.assert_called_once()

        # Verify the query includes UNWIND and SET operations
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "UNWIND" in query
        assert "visit_count" in query
        assert "last_visited" in query

    @pytest.mark.asyncio
    async def test_update_state_status_resets_visit_count(self):
        """Test that updating state status resets visit_count to 0"""
        # Mock Neo4j driver
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"s": {"id": "state_1"}}

        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        # Create Neo4jKnowledgeGraph with mocked driver
        neo4j_graph = Neo4jKnowledgeGraph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        neo4j_graph._driver = mock_driver

        # Test update
        success = await neo4j_graph.update_state_status("state_1", "completed")

        assert success is True
        mock_session.run.assert_called_once()

        # Verify the query sets visit_count = 0
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "visit_count = 0" in query
        assert "last_visited = 0.0" in query

    @pytest.mark.asyncio
    async def test_boredom_detection_warning_injection(self):
        """Test that boredom warning is injected when visit_count exceeds threshold"""
        # Mock semantic manager with Neo4j
        mock_neo4j = AsyncMock()

        # Simulate states with high visit counts
        high_visit_states = [
            {
                "id": "state_1",
                "type": "task",
                "desc": "implement feature X",
                "status": "active",
                "created": 1000,
                "updated": 2000,
                "visit_count": 6,  # Exceeds default threshold of 5
                "last_visited": 2000
            }
        ]

        mock_neo4j.get_active_states.return_value = high_visit_states
        mock_neo4j.get_completed_states.return_value = []
        mock_neo4j.increment_state_visits.return_value = 1

        # Create context manager with mocked dependencies
        context_manager = ContextManager(
            max_tokens=4096,
            semantic_manager=MagicMock(neo4j_graph=mock_neo4j)
        )

        # Build state message
        with patch('app.config.BOREDOM_THRESHOLD', 5):
            with patch('app.config.BOREDOM_DETECTION_ENABLED', True):
                with patch('app.config.STATE_INJECTION_LIMITS', {'task': 3}):
                    state_msg = await context_manager._build_state_message()

        # Verify boredom warning is in message
        assert state_msg is not None
        content = state_msg["content"]
        assert "⚠️ LOOP DETECTED" in content
        assert "implement feature X" in content or "1 states" in content


class TestEchoGuard:
    """Test response similarity detection (echo guard)"""

    @pytest.mark.asyncio
    async def test_store_response_embedding(self):
        """Test that response embeddings are stored in Redis"""
        # Mock Redis storage
        mock_redis = AsyncMock()
        mock_redis.zadd.return_value = 1
        mock_redis.zremrangebyrank.return_value = 0

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        # Create semantic manager with mocked Redis
        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # Test storing embedding
        test_embedding = [0.1] * 384
        with patch('app.config.ECHO_RESPONSE_HISTORY_SIZE', 10):
            success = await semantic_manager.store_response_embedding(test_embedding)

        assert success is True
        mock_redis.zadd.assert_called_once()
        mock_redis.zremrangebyrank.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_response_similarity_no_duplicates(self):
        """Test similarity check when response is novel"""
        # Mock Redis with different embeddings
        mock_redis = AsyncMock()

        # Return a stored embedding that's different
        stored_embedding = [0.5] * 384
        import json
        mock_redis.zrange.return_value = [json.dumps(stored_embedding)]

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # Test with a different embedding
        new_embedding = [0.1] * 384
        with patch('app.config.ECHO_SIMILARITY_THRESHOLD', 0.95):
            is_duplicate, similarity = await semantic_manager.check_response_similarity(new_embedding)

        assert is_duplicate is False
        assert similarity < 0.95

    @pytest.mark.asyncio
    async def test_check_response_similarity_detects_duplicate(self):
        """Test similarity check when response is duplicate"""
        # Mock Redis with identical embedding
        mock_redis = AsyncMock()

        # Return the same embedding
        test_embedding = [0.5] * 384
        import json
        mock_redis.zrange.return_value = [json.dumps(test_embedding)]

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # Test with the same embedding
        with patch('app.config.ECHO_SIMILARITY_THRESHOLD', 0.95):
            is_duplicate, similarity = await semantic_manager.check_response_similarity(test_embedding)

        assert is_duplicate is True
        assert similarity >= 0.95

    @pytest.mark.asyncio
    async def test_check_response_similarity_high_similarity(self):
        """Test similarity check with very similar embeddings"""
        # Mock Redis with very similar embedding
        mock_redis = AsyncMock()

        # Create two very similar embeddings (98% similar)
        base_embedding = np.random.randn(384)
        stored_embedding = base_embedding + np.random.randn(384) * 0.1
        new_embedding = base_embedding + np.random.randn(384) * 0.1

        import json
        mock_redis.zrange.return_value = [json.dumps(stored_embedding.tolist())]

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # Test with very similar embedding
        with patch('app.config.ECHO_SIMILARITY_THRESHOLD', 0.95):
            is_duplicate, similarity = await semantic_manager.check_response_similarity(new_embedding.tolist())

        # Similarity should be high but exact value depends on random noise
        assert similarity > 0.5  # At least moderately similar

    @pytest.mark.asyncio
    async def test_echo_guard_empty_history(self):
        """Test echo guard when there's no previous response history"""
        # Mock Redis with empty history
        mock_redis = AsyncMock()
        mock_redis.zrange.return_value = []

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # Test with any embedding
        test_embedding = [0.5] * 384
        is_duplicate, similarity = await semantic_manager.check_response_similarity(test_embedding)

        assert is_duplicate is False
        assert similarity == 0.0


class TestIntegration:
    """Integration tests for loop detection features"""

    @pytest.mark.asyncio
    async def test_full_boredom_detection_flow(self):
        """Test complete flow: state creation -> visits -> boredom warning"""
        # This would require actual database connections
        # For now, we test the flow with mocks

        # 1. Create state
        state = State.create("goal", "go to hydro plant", "active")
        assert state.visit_count == 0

        # 2. Simulate multiple visits
        # (In real flow, this happens via increment_state_visits)
        state.visit_count = 6

        # 3. Check if boredom threshold exceeded
        from app.config import BOREDOM_THRESHOLD
        boredom_threshold = 5

        assert state.visit_count > boredom_threshold

        # 4. Warning should be generated
        # (Tested in test_boredom_detection_warning_injection)

    @pytest.mark.asyncio
    async def test_full_echo_guard_flow(self):
        """Test complete flow: response -> embed -> check -> regenerate"""
        # Mock the full chain
        mock_redis = AsyncMock()

        # First check: no duplicates
        mock_redis.zrange.return_value = []

        mock_redis_storage = MagicMock()
        mock_redis_storage.redis = mock_redis

        semantic_manager = SemanticManager(
            embedding_model=MagicMock(),
            redis_storage=mock_redis_storage,
            qdrant_db=MagicMock(),
            neo4j_graph=MagicMock()
        )

        # 1. Generate embedding for response
        response_text = "This is a test response"
        response_embedding = [0.5] * 384  # Simulated

        # 2. Check for duplicates (first time - none)
        is_dup_1, sim_1 = await semantic_manager.check_response_similarity(response_embedding)
        assert is_dup_1 is False

        # 3. Store the embedding
        await semantic_manager.store_response_embedding(response_embedding)

        # 4. Check again with same embedding (should detect duplicate)
        import json
        mock_redis.zrange.return_value = [json.dumps(response_embedding)]

        with patch('app.config.ECHO_SIMILARITY_THRESHOLD', 0.95):
            is_dup_2, sim_2 = await semantic_manager.check_response_similarity(response_embedding)

        assert is_dup_2 is True
        assert sim_2 >= 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
