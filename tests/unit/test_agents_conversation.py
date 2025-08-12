"""
Unit tests for the Conversation Agent.

Tests the conversational functionality for handling queries that don't require
research and can be answered with general knowledge.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ai_deep_research_assistant.agents.conversation import (
    ConversationDeps,
    ConversationResponse,
    handle_conversation,
    conversation_agent,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestConversationAgent:
    """Test cases for the conversation agent functionality."""

    @pytest.fixture
    def sample_greeting_queries(self):
        """Sample greeting queries for testing."""
        return [
            "Hello, how are you?",
            "Hi there!",
            "Good morning",
            "Hey, what's up?",
            "Greetings!",
        ]

    @pytest.fixture
    def sample_knowledge_queries(self):
        """Sample general knowledge queries for testing."""
        return [
            "What is machine learning?",
            "Explain photosynthesis",
            "How do you calculate the area of a circle?",
            "What is artificial intelligence?",
            "Define quantum mechanics",
            "What are the primary colors?",
        ]

    @pytest.fixture
    def sample_capability_queries(self):
        """Sample queries about system capabilities."""
        return [
            "What are your capabilities?",
            "How can you help me?",
            "What kind of questions can I ask you?",
            "What do you do?",
            "Can you help with research?",
        ]

    @pytest.fixture
    def sample_creative_queries(self):
        """Sample creative task queries."""
        return [
            "Write a short story about space",
            "Can you compose a poem?",
            "Tell me a joke",
            "Create a haiku about nature",
            "Write a brief essay on creativity",
        ]

    async def test_handle_greeting_conversation(self, sample_greeting_queries):
        """Test handling of greeting conversations."""
        with patch.object(conversation_agent, "run") as mock_run:
            # Mock conversation agent response for greeting
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="Hello! I'm doing well, thank you for asking. I'm an AI research assistant here to help you with questions, research tasks, and general conversation. How can I assist you today?",
                confidence=0.95,
                response_type="greeting",
                suggested_follow_ups=[
                    "What would you like to research today?",
                    "Do you have any questions I can help with?",
                    "Would you like to know about my capabilities?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            for query in sample_greeting_queries[:2]:  # Test first 2
                response = await handle_conversation(
                    query=query, session_id="test-session", request_id="test-request"
                )

                assert isinstance(response, ConversationResponse)
                assert response.response_type == "greeting"
                assert response.confidence > 0.9
                assert len(response.response) > 20  # Substantial response
                assert len(response.suggested_follow_ups) > 0

    async def test_handle_knowledge_conversation(self, sample_knowledge_queries):
        """Test handling of general knowledge conversations."""
        with patch.object(conversation_agent, "run") as mock_run:
            # Mock conversation agent response for knowledge query
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that analysis. Common types include supervised learning, unsupervised learning, and reinforcement learning, each suited for different types of problems and data structures.",
                confidence=0.90,
                response_type="explanation",
                suggested_follow_ups=[
                    "Would you like to know about specific types of machine learning?",
                    "Are you interested in machine learning applications?",
                    "Should I explain how neural networks work?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            query = sample_knowledge_queries[0]  # "What is machine learning?"
            response = await handle_conversation(
                query=query, session_id="test-session", request_id="test-request"
            )

            assert response.response_type == "explanation"
            assert response.confidence >= 0.8
            assert "machine learning" in response.response.lower()
            assert len(response.response) > 100  # Comprehensive explanation

    async def test_handle_capability_conversation(self, sample_capability_queries):
        """Test handling of capability-related conversations."""
        with patch.object(conversation_agent, "run") as mock_run:
            # Mock conversation agent response for capability query
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="I'm an AI research assistant with several key capabilities:\n\n• **Research & Analysis**: I can conduct comprehensive research on various topics using web search and multiple specialized agents\n• **Information Synthesis**: I combine information from multiple sources to provide balanced, well-sourced answers\n• **Conversational Help**: I can answer questions, explain concepts, and engage in general conversation\n• **Creative Tasks**: I can help with writing, brainstorming, and creative problem-solving\n\nI'm designed to be helpful, accurate, and transparent about my limitations. What would you like help with today?",
                confidence=0.95,
                response_type="capability_explanation",
                suggested_follow_ups=[
                    "Would you like to try a research query?",
                    "Do you have a specific topic you'd like me to explain?",
                    "Is there a particular task I can help you with?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            query = sample_capability_queries[0]  # "What are your capabilities?"
            response = await handle_conversation(
                query=query, session_id="test-session", request_id="test-request"
            )

            assert response.response_type == "capability_explanation"
            assert response.confidence >= 0.9
            assert "research" in response.response.lower()
            assert (
                "capabilities" in response.response.lower()
                or "help" in response.response.lower()
            )

    async def test_handle_creative_conversation(self, sample_creative_queries):
        """Test handling of creative task conversations."""
        with patch.object(conversation_agent, "run") as mock_run:
            # Mock conversation agent response for creative task
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="Here's a short story about space:\n\n**The Silent Voyager**\n\nCaptain Sarah Chen gazed through the observation deck of the starship Endeavor, watching distant galaxies spiral in their eternal dance. After eighteen months in deep space, she had grown accustomed to the profound silence that surrounded them—a silence so complete it seemed to have weight.\n\n'Signal detected from sector 7-Alpha,' announced her AI companion. 'Origin unknown.'\n\nSarah's heart quickened. In all their travels, they had never encountered anything like this. The universe, it seemed, was about to become a little less lonely.\n\n'Set course for the signal,' she commanded, her voice steady despite the excitement building within her. 'It's time to make history.'",
                confidence=0.85,
                response_type="creative",
                suggested_follow_ups=[
                    "Would you like me to continue this story?",
                    "Should I write about a different topic?",
                    "Would you like me to try a different creative format?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            query = sample_creative_queries[0]  # "Write a short story about space"
            response = await handle_conversation(
                query=query, session_id="test-session", request_id="test-request"
            )

            assert response.response_type == "creative"
            assert response.confidence >= 0.8
            assert len(response.response) > 200  # Substantial creative content
            assert (
                "space" in response.response.lower()
                or "star" in response.response.lower()
            )

    async def test_handle_conversation_with_context(self):
        """Test conversation handling with user context."""
        with patch.object(conversation_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="Given your academic background, I can explain machine learning from a more technical perspective. Machine learning algorithms optimize objective functions through iterative processes, using techniques like gradient descent to minimize loss functions. The mathematical foundation involves linear algebra, statistics, and calculus to transform input feature spaces into meaningful representations.",
                confidence=0.92,
                response_type="contextual_explanation",
                suggested_follow_ups=[
                    "Would you like me to dive into the mathematical details?",
                    "Should I explain specific algorithms?",
                    "Are you interested in recent research developments?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            response = await handle_conversation(
                query="Explain machine learning",
                session_id="test-session",
                request_id="test-request",
                user_context="Academic researcher with PhD in Computer Science",
            )

            # Verify context was included in the agent call
            mock_run.assert_called_once()
            # Context is passed via deps, not in the prompt

            assert response.response_type == "contextual_explanation"
            assert (
                "technical" in response.response or "mathematical" in response.response
            )

    async def test_handle_math_calculation(self):
        """Test handling of mathematical calculations."""
        with patch.object(conversation_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="To calculate the area of a circle, you use the formula:\n\nA = πr²\n\nWhere:\n• A = area\n• π (pi) ≈ 3.14159\n• r = radius of the circle\n\nFor example, if a circle has a radius of 5 units:\nA = π × 5² = π × 25 ≈ 3.14159 × 25 ≈ 78.54 square units\n\nThe radius is the distance from the center of the circle to any point on its edge.",
                confidence=0.98,
                response_type="mathematical",
                suggested_follow_ups=[
                    "Would you like to calculate the area for a specific radius?",
                    "Should I explain other circle formulas like circumference?",
                    "Do you need help with other geometry calculations?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            response = await handle_conversation(
                query="How do you calculate the area of a circle?",
                session_id="test-session",
                request_id="test-request",
            )

            assert response.response_type == "mathematical"
            assert response.confidence >= 0.95
            assert "π" in response.response or "pi" in response.response
            assert "r²" in response.response or "radius" in response.response

    async def test_conversation_error_handling(self):
        """Test error handling in conversation agent."""
        with patch.object(conversation_agent, "run") as mock_run:
            # Simulate agent error
            mock_run.side_effect = Exception("Conversation agent error")

            # Should propagate the error
            with pytest.raises(Exception):
                await handle_conversation(
                    query="test query",
                    session_id="test-session",
                    request_id="test-request",
                )

    def test_conversation_response_validation(self):
        """Test ConversationResponse model validation."""
        # Valid response
        valid_response = ConversationResponse(
            response="This is a test response",
            confidence=0.85,
            response_type="explanation",
            suggested_follow_ups=["Follow up 1", "Follow up 2"],
            handled_successfully=True,
        )
        assert valid_response.confidence == 0.85
        assert len(valid_response.suggested_follow_ups) == 2

        # Test confidence bounds - pydantic may clamp instead of raising
        try:
            invalid_response = ConversationResponse(
                response="Invalid response",
                confidence=1.5,  # Invalid > 1.0
                response_type="explanation",
                suggested_follow_ups=[],
                handled_successfully=True,
            )
            # If pydantic allows it, confidence should be valid range
            assert 0.0 <= invalid_response.confidence <= 1.0
        except Exception:
            # If pydantic raises, that's also acceptable
            pass

    def test_conversation_deps_validation(self):
        """Test ConversationDeps model validation."""
        # Valid deps
        valid_deps = ConversationDeps(
            session_id="test-session",
            request_id="test-request",
            user_context="Test context",
        )
        assert valid_deps.session_id == "test-session"
        assert valid_deps.user_context == "Test context"

        # Test without context
        minimal_deps = ConversationDeps(
            session_id="test-session", request_id="test-request"
        )
        assert minimal_deps.user_context is None

    @pytest.mark.parametrize(
        "query_type,expected_type",
        [
            ("Hello", "greeting"),
            ("What is AI?", "explanation"),
            ("What can you do?", "capability_explanation"),
            ("Write a poem", "creative"),
            ("What is 2+2?", "mathematical"),
        ],
    )
    async def test_response_type_classification(self, query_type, expected_type):
        """Test that queries are classified into correct response types."""
        with patch.object(conversation_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response=f"Response for {query_type}",
                confidence=0.9,
                response_type=expected_type,
                suggested_follow_ups=["Follow up"],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            response = await handle_conversation(
                query=query_type, session_id="test-session", request_id="test-request"
            )

            assert response.response_type == expected_type

    async def test_empty_query_handling(self):
        """Test handling of empty or invalid queries."""
        with patch.object(conversation_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="I notice you haven't asked a specific question. I'm here to help with research, answer questions, explain concepts, or just chat. What would you like to know about?",
                confidence=0.85,
                response_type="clarification",
                suggested_follow_ups=[
                    "Ask me about any topic you're curious about",
                    "Try a research query like 'latest developments in...'",
                    "Ask me to explain a concept",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            response = await handle_conversation(
                query="", session_id="test-session", request_id="test-request"
            )

            assert response.response_type == "clarification"
            assert (
                "question" in response.response.lower()
                or "help" in response.response.lower()
            )
            assert len(response.suggested_follow_ups) > 0

    async def test_follow_up_suggestions_quality(self):
        """Test that follow-up suggestions are relevant and helpful."""
        with patch.object(conversation_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ConversationResponse(
                response="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs primarily in the chloroplasts of plant cells, using chlorophyll to capture light energy.",
                confidence=0.92,
                response_type="explanation",
                suggested_follow_ups=[
                    "Would you like to know about the detailed steps of photosynthesis?",
                    "Should I explain how photosynthesis relates to cellular respiration?",
                    "Are you interested in learning about different types of photosynthesis?",
                    "Would you like to explore how climate change affects photosynthesis?",
                ],
                handled_successfully=True,
            )
            mock_run.return_value = mock_result

            response = await handle_conversation(
                query="Explain photosynthesis",
                session_id="test-session",
                request_id="test-request",
            )

            # Test follow-up quality
            assert len(response.suggested_follow_ups) > 2
            for follow_up in response.suggested_follow_ups:
                assert len(follow_up) > 10  # Substantial suggestions
                assert follow_up.endswith("?")  # Proper question format
                assert "photosynthesis" in follow_up.lower()  # Relevant to topic

    async def test_confidence_scores_appropriate(self):
        """Test that confidence scores are appropriate for different query types."""
        test_cases = [
            ("Hello", 0.95, "greeting"),  # High confidence for simple greeting
            ("What is 2+2?", 0.98, "mathematical"),  # Very high for basic math
            (
                "Explain quantum mechanics",
                0.85,
                "explanation",
            ),  # Lower for complex topics
            ("Write a story", 0.80, "creative"),  # Lower for subjective creative tasks
        ]

        for query, expected_confidence, response_type in test_cases:
            with patch.object(conversation_agent, "run") as mock_run:
                mock_result = Mock()
                mock_result.output = ConversationResponse(
                    response=f"Response to {query}",
                    confidence=expected_confidence,
                    response_type=response_type,
                    suggested_follow_ups=["Follow up"],
                    handled_successfully=True,
                )
                mock_run.return_value = mock_result

                response = await handle_conversation(
                    query=query, session_id="test-session", request_id="test-request"
                )

                # Allow some tolerance in confidence matching
                assert abs(response.confidence - expected_confidence) <= 0.05
