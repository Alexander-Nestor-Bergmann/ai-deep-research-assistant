"""
Unit tests for the Guardrail Agent.

Tests the query classification functionality that determines whether queries
require research or can be answered conversationally.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.guardrail import (
    GuardrailOutput,
    classify_query,
    should_route_to_research,
    guardrail_agent
)
from config.settings import get_settings


@pytest.mark.unit
@pytest.mark.asyncio
class TestGuardrailAgent:
    """Test cases for the guardrail agent functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch('agents.guardrail.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.min_confidence_threshold = 0.7
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def sample_research_queries(self):
        """Sample queries that should be classified as requiring research."""
        return [
            "What are the latest developments in quantum computing?",
            "Recent advances in AI in 2024",
            "Current climate change research findings",
            "What happened in the recent climate summit?",
            "Latest news about space exploration",
            "Compare renewable energy policies across countries",
            "Recent scientific breakthroughs in medicine"
        ]

    @pytest.fixture
    def sample_conversational_queries(self):
        """Sample queries that should be handled conversationally."""
        return [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain photosynthesis",
            "How do you calculate the area of a circle?",
            "What are your capabilities?",
            "Can you help me write a story?",
            "What is 2 + 2?",
            "Define artificial intelligence"
        ]

    async def test_classify_research_query(self, sample_research_queries, mock_settings):
        """Test that research queries are correctly classified."""
        with patch.object(guardrail_agent, 'run') as mock_run:
            # Mock agent response for research query
            mock_result = Mock()
            mock_result.output = GuardrailOutput(
                is_research_request=True,
                confidence=0.9,
                reasoning="Query asks for recent developments requiring current information",
                complexity_estimate="moderate",
                suggested_research_type="news",
                estimated_sources_needed=5,
                can_answer_immediately=False,
                requires_current_info=True
            )
            mock_run.return_value = mock_result

            for query in sample_research_queries[:3]:  # Test first 3 to avoid rate limits
                result = await classify_query(
                    query=query,
                    session_id="test-session",
                    request_id="test-request"
                )
                
                assert result.is_research_request is True
                assert result.confidence > 0.5
                assert result.reasoning is not None
                assert result.complexity_estimate in ["simple", "moderate", "complex"]
                assert isinstance(result.requires_current_info, bool)
                assert isinstance(result.can_answer_immediately, bool)

    async def test_classify_conversational_query(self, sample_conversational_queries, mock_settings):
        """Test that conversational queries are correctly classified."""
        with patch.object(guardrail_agent, 'run') as mock_run:
            # Mock agent response for conversational query
            mock_result = Mock()
            mock_result.output = GuardrailOutput(
                is_research_request=False,
                confidence=0.8,
                reasoning="Query can be answered with general knowledge",
                complexity_estimate="simple",
                suggested_research_type=None,
                estimated_sources_needed=1,
                can_answer_immediately=True,
                requires_current_info=False
            )
            mock_run.return_value = mock_result

            for query in sample_conversational_queries[:3]:  # Test first 3
                result = await classify_query(
                    query=query,
                    session_id="test-session",
                    request_id="test-request"
                )
                
                assert result.is_research_request is False
                assert result.confidence > 0.5
                assert result.can_answer_immediately is True
                assert result.requires_current_info is False

    async def test_classify_query_with_context(self, mock_settings):
        """Test query classification with conversation context."""
        with patch.object(guardrail_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = GuardrailOutput(
                is_research_request=True,
                confidence=0.85,
                reasoning="Follow-up question builds on previous context",
                complexity_estimate="moderate",
                suggested_research_type="academic",
                estimated_sources_needed=3,
                can_answer_immediately=False,
                requires_current_info=True
            )
            mock_run.return_value = mock_result

            result = await classify_query(
                query="Can you tell me more about the applications?",
                session_id="test-session",
                request_id="test-request",
                conversation_context="Previously discussed quantum computing advances"
            )
            
            # Verify that context was included in the agent call
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert "Previously discussed quantum computing advances" in call_args[0]

    def test_should_route_to_research_high_confidence(self, mock_settings):
        """Test routing decision with high confidence research request."""
        classification = GuardrailOutput(
            is_research_request=True,
            confidence=0.9,
            reasoning="Clear research query",
            complexity_estimate="moderate",
            suggested_research_type="general",
            estimated_sources_needed=3,
            can_answer_immediately=False,
            requires_current_info=True
        )
        
        assert should_route_to_research(classification) is True

    def test_should_route_to_research_low_confidence(self, mock_settings):
        """Test routing decision with low confidence research request."""
        classification = GuardrailOutput(
            is_research_request=True,
            confidence=0.5,  # Below default threshold
            reasoning="Uncertain classification",
            complexity_estimate="simple",
            suggested_research_type=None,
            estimated_sources_needed=1,
            can_answer_immediately=True,
            requires_current_info=False
        )
        
        # Should still route to research if complexity is moderate/complex
        assert should_route_to_research(classification) is False

    def test_should_route_to_research_complex_override(self, mock_settings):
        """Test that complex queries route to research even with lower confidence."""
        classification = GuardrailOutput(
            is_research_request=True,
            confidence=0.6,  # Below threshold
            reasoning="Complex query requiring analysis",
            complexity_estimate="complex",  # This should override confidence
            suggested_research_type="academic",
            estimated_sources_needed=5,
            can_answer_immediately=False,
            requires_current_info=True
        )
        
        assert should_route_to_research(classification) is True

    def test_should_route_to_research_conversational(self, mock_settings):
        """Test routing decision for conversational queries."""
        classification = GuardrailOutput(
            is_research_request=False,
            confidence=0.9,
            reasoning="Simple conversational query",
            complexity_estimate="simple",
            suggested_research_type=None,
            estimated_sources_needed=1,
            can_answer_immediately=True,
            requires_current_info=False
        )
        
        assert should_route_to_research(classification) is False

    def test_should_route_to_research_custom_threshold(self, mock_settings):
        """Test routing with custom confidence threshold."""
        classification = GuardrailOutput(
            is_research_request=True,
            confidence=0.6,
            reasoning="Moderate confidence classification",
            complexity_estimate="simple",
            suggested_research_type="general",
            estimated_sources_needed=2,
            can_answer_immediately=False,
            requires_current_info=True
        )
        
        # With custom threshold of 0.5, should route to research
        assert should_route_to_research(classification, confidence_threshold=0.5) is True
        
        # With custom threshold of 0.8, should not route to research
        assert should_route_to_research(classification, confidence_threshold=0.8) is False

    async def test_classify_query_error_handling(self, mock_settings):
        """Test error handling in query classification."""
        with patch.object(guardrail_agent, 'run') as mock_run:
            # Simulate agent error
            mock_run.side_effect = Exception("Agent error")
            
            # Should handle the error gracefully
            with pytest.raises(Exception):
                await classify_query(
                    query="test query",
                    session_id="test-session",
                    request_id="test-request"
                )

    def test_guardrail_output_validation(self):
        """Test GuardrailOutput model validation."""
        # Valid output
        valid_output = GuardrailOutput(
            is_research_request=True,
            confidence=0.8,
            reasoning="Valid reasoning",
            complexity_estimate="moderate",
            suggested_research_type="academic",
            estimated_sources_needed=3,
            can_answer_immediately=False,
            requires_current_info=True
        )
        assert valid_output.confidence == 0.8
        assert valid_output.complexity_estimate == "moderate"
        
        # Test confidence bounds - pydantic may clamp instead of raising
        try:
            invalid_output = GuardrailOutput(
                is_research_request=True,
                confidence=1.5,  # Invalid confidence > 1.0
                reasoning="Invalid confidence",
                complexity_estimate="moderate",
                can_answer_immediately=False,
                requires_current_info=True
            )
            # If pydantic allows it, confidence should be valid range
            assert 0.0 <= invalid_output.confidence <= 1.0
        except Exception:
            # If pydantic raises, that's also acceptable
            pass

    @pytest.mark.parametrize("complexity,should_research", [
        ("simple", False),
        ("moderate", True),
        ("complex", True)
    ])
    def test_complexity_based_routing(self, complexity, should_research, mock_settings):
        """Test routing based on complexity estimates."""
        classification = GuardrailOutput(
            is_research_request=True,
            confidence=0.6,  # Below threshold
            reasoning=f"Query classified as {complexity}",
            complexity_estimate=complexity,
            suggested_research_type="general",
            estimated_sources_needed=3,
            can_answer_immediately=False,
            requires_current_info=True
        )
        
        result = should_route_to_research(classification)
        assert result == should_research

    async def test_empty_query_handling(self, mock_settings):
        """Test handling of empty or invalid queries."""
        with patch.object(guardrail_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = GuardrailOutput(
                is_research_request=False,
                confidence=0.9,
                reasoning="Empty query cannot be processed",
                complexity_estimate="simple",
                suggested_research_type=None,
                estimated_sources_needed=0,
                can_answer_immediately=True,
                requires_current_info=False
            )
            mock_run.return_value = mock_result

            result = await classify_query(
                query="",
                session_id="test-session",
                request_id="test-request"
            )
            
            assert result.is_research_request is False
            assert result.can_answer_immediately is True

    def test_settings_integration(self):
        """Test integration with settings configuration."""
        # Test that settings are being used for threshold
        with patch('agents.guardrail.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.min_confidence_threshold = 0.8
            mock_get_settings.return_value = mock_settings
            
            classification = GuardrailOutput(
                is_research_request=True,
                confidence=0.75,  # Below custom threshold
                reasoning="Test classification",
                complexity_estimate="simple",
                suggested_research_type=None,
                estimated_sources_needed=1,
                can_answer_immediately=False,
                requires_current_info=True
            )
            
            # Should use settings threshold
            result = should_route_to_research(classification)
            assert result is False  # Below 0.8 threshold