"""
Pytest configuration and fixtures for the research assistant test suite.

This module provides shared fixtures, test data, and configuration for all tests
including mock objects, test states, and performance benchmarks.
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.guardrail import GuardrailOutput
from agents.planner import ResearchPlan, ResearchTask
from agents.researcher import ResearchOutput, ResearchFinding
from agents.synthesizer import SynthesisOutput
from agents.conversation import ConversationResponse
from graph.workflow import EnhancedResearchState, create_enhanced_initial_state

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer than 30 seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test location
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)


# =============================================================================
# EVENT LOOP AND ASYNC FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_async_sleep():
    """Mock asyncio.sleep to speed up tests."""
    with patch('asyncio.sleep', side_effect=lambda x: None):
        yield


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_research_query():
    """Sample research query for testing."""
    return "What are the latest developments in quantum computing?"


@pytest.fixture
def sample_conversational_query():
    """Sample conversational query for testing."""
    return "Hello, how are you today?"


@pytest.fixture
def sample_complex_query():
    """Sample complex research query for testing."""
    return "Analyze the comparative effectiveness of renewable energy policies across different countries, focusing on wind and solar adoption rates, economic impacts, and environmental outcomes over the past decade."


@pytest.fixture
def sample_guardrail_output():
    """Sample guardrail agent output."""
    return GuardrailOutput(
        is_research_request=True,
        confidence=0.85,
        reasoning="Query contains specific research elements requiring comprehensive analysis",
        complexity_estimate="moderate",
        suggested_research_type="academic",
        estimated_sources_needed=5,
        can_answer_immediately=False,
        requires_current_info=True
    )


@pytest.fixture
def sample_research_plan():
    """Sample research plan for testing."""
    return ResearchPlan(
        original_query="What are recent developments in AI?",
        research_objective="Comprehensive analysis of AI developments",
        tasks=[
            ResearchTask(
                task_description="Academic research on AI advances",
                agent_type="academic_research",
                priority=9,
                keywords=["AI", "artificial intelligence", "recent", "developments"],
                expected_outcome="Peer-reviewed research findings"
            ),
            ResearchTask(
                task_description="News research on AI developments", 
                agent_type="news_research",
                priority=8,
                keywords=["AI news", "artificial intelligence", "2024"],
                expected_outcome="Current news coverage"
            )
        ],
        estimated_duration_minutes=60,
        parallel_execution=True,
        success_criteria=[
            "Academic depth and accuracy",
            "Current relevance and timeliness"
        ],
        reasoning="Multi-agent approach for comprehensive coverage"
    )


@pytest.fixture
def sample_research_findings():
    """Sample research findings for testing."""
    return [
        ResearchFinding(
            claim="Large language models have achieved significant performance improvements in 2024",
            evidence_urls=[
                "https://arxiv.org/abs/2024.1234",
                "https://openai.com/research/gpt-4-improvements"
            ],
            evidence_titles=[
                "Advances in Large Language Model Architecture",
                "GPT-4 Performance Improvements"
            ],
            evidence_snippets=[
                "Novel attention mechanisms show 15% improvement in reasoning tasks",
                "GPT-4 demonstrates enhanced mathematical and coding capabilities"
            ],
            confidence=0.90,
            keywords=["large language models", "performance", "improvements"],
            source_count=2
        ),
        ResearchFinding(
            claim="AI safety research has gained increased funding and attention",
            evidence_urls=[
                "https://anthropic.com/safety-research-2024"
            ],
            evidence_titles=[
                "AI Safety Research Progress Report"
            ],
            evidence_snippets=[
                "Investment in AI alignment research increased by 40% in 2024"
            ],
            confidence=0.85,
            keywords=["AI safety", "funding", "alignment research"],
            source_count=1
        )
    ]


@pytest.fixture
def sample_research_output(sample_research_findings):
    """Sample research output for testing."""
    return ResearchOutput(
        findings=sample_research_findings,
        agent_type="academic_research",
        sources_searched=8,
        search_queries_used=["AI developments 2024", "large language models recent"],
        confidence_score=0.88,
        agent_notes="High-quality academic sources with peer review"
    )


@pytest.fixture
def sample_synthesis_output():
    """Sample synthesis output for testing."""
    return SynthesisOutput(
        final_answer="Artificial intelligence has experienced remarkable progress in 2024, with significant advances in large language models showing 15% performance improvements in reasoning tasks. GPT-4 has demonstrated enhanced mathematical and coding capabilities, while AI safety research has gained increased attention with 40% more funding allocated to alignment research. These developments indicate a maturing field balancing capability advancement with safety considerations.",
        key_findings=[
            "Large language models achieved 15% performance improvements in reasoning",
            "GPT-4 shows enhanced mathematical and coding capabilities", 
            "AI safety research funding increased by 40% in 2024"
        ],
        source_urls=[
            "https://arxiv.org/abs/2024.1234",
            "https://openai.com/research/gpt-4-improvements",
            "https://anthropic.com/safety-research-2024"
        ],
        source_titles=[
            "Advances in Large Language Model Architecture",
            "GPT-4 Performance Improvements",
            "AI Safety Research Progress Report"
        ],
        confidence_score=0.87,
        limitations=[
            "Limited to publicly available research",
            "Some commercial developments may not be disclosed"
        ],
        follow_up_questions=[
            "What specific AI applications have benefited most from these improvements?",
            "How do different companies' AI safety approaches compare?",
            "What are the projected timelines for AI capability milestones?"
        ],
        research_summary="Synthesized findings from academic research and industry reports",
        total_sources=3,
        agents_used=["academic_research", "news_research"]
    )


@pytest.fixture
def sample_conversation_response():
    """Sample conversation response for testing."""
    return ConversationResponse(
        response="Hello! I'm doing well, thank you for asking. I'm an AI research assistant designed to help with questions, research tasks, and general conversation. I can conduct comprehensive research using multiple specialized agents, explain complex topics, and engage in friendly dialogue. How can I assist you today?",
        confidence=0.95,
        reasoning="Friendly greeting with capability explanation",
        response_type="greeting",
        suggested_follow_ups=[
            "What would you like to research today?",
            "Do you have any questions I can help answer?",
            "Would you like to know more about my research capabilities?"
        ]
    )


@pytest.fixture
def sample_initial_state(sample_research_query):
    """Sample initial workflow state."""
    return create_enhanced_initial_state(
        query=sample_research_query,
        session_id="test-session-123",
        request_id="test-request-456"
    )


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_brave_response():
    """Mock Brave search API response."""
    return {
        "web": {
            "results": [
                {
                    "url": "https://example.com/quantum-computing-2024",
                    "title": "Quantum Computing Advances in 2024",
                    "description": "Recent breakthroughs in quantum computing include improved error correction and new algorithms for practical applications.",
                    "published_date": "2024-03-15",
                    "age": "2 months ago"
                },
                {
                    "url": "https://nature.com/quantum-research", 
                    "title": "Nature: Quantum Error Correction Methods",
                    "description": "Peer-reviewed research demonstrates significant improvements in quantum error correction with 99.9% fidelity achieved in laboratory conditions.",
                    "published_date": "2024-02-01",
                    "age": "3 months ago"
                },
                {
                    "url": "https://arxiv.org/abs/2024.5678",
                    "title": "Quantum Algorithms for Optimization Problems",
                    "description": "Novel quantum algorithms show exponential speedup for specific optimization problems compared to classical approaches.",
                    "published_date": "2024-01-20",
                    "age": "4 months ago"
                }
            ]
        }
    }


@pytest.fixture
def mock_settings():
    """Mock settings configuration."""
    mock_settings = Mock()
    mock_settings.brave_api_key = "test-brave-api-key"
    mock_settings.max_search_results = 8
    mock_settings.min_confidence_threshold = 0.7
    mock_settings.synthesis_confidence_threshold = 0.8
    mock_settings.model_choice = "anthropic/claude-3-5-sonnet-20241022"
    mock_settings.model_choice_small = "openai/gpt-4o-mini"
    return mock_settings


@pytest.fixture
def mock_agent_responses():
    """Mock responses for different agent types."""
    return {
        'guardrail': GuardrailOutput(
            is_research_request=True,
            confidence=0.85,
            reasoning="Query requires research analysis",
            complexity_estimate="moderate",
            suggested_research_type="academic",
            estimated_sources_needed=5,
            can_answer_immediately=False,
            requires_current_info=True
        ),
        'conversation': ConversationResponse(
            response="Test conversational response",
            confidence=0.9,
            reasoning="Simple conversational query",
            response_type="explanation",
            suggested_follow_ups=["Follow up question?"]
        )
    }


@pytest.fixture  
def mock_search_tool():
    """Mock web search tool."""
    async def mock_web_search(query: str, count: int = 5):
        return {
            "results": [
                {
                    "url": f"https://example.com/result-{i}",
                    "title": f"Test Result {i}",
                    "description": f"Test description for result {i}",
                    "published_date": "2024-01-01"
                }
                for i in range(count)
            ],
            "query_used": query,
            "search_type": "general",
            "source_count": count
        }
    
    return mock_web_search


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

@pytest.fixture
def assertion_helpers():
    """Helper functions for test assertions."""
    class AssertionHelpers:
        @staticmethod
        def assert_valid_guardrail_output(output: GuardrailOutput):
            """Assert that guardrail output is valid."""
            assert isinstance(output.is_research_request, bool)
            assert 0.0 <= output.confidence <= 1.0
            assert len(output.reasoning) > 0
            assert output.complexity_estimate in ["simple", "moderate", "complex"]
            assert isinstance(output.can_answer_immediately, bool)
            assert isinstance(output.requires_current_info, bool)
        
        @staticmethod
        def assert_valid_research_plan(plan: ResearchPlan):
            """Assert that research plan is valid."""
            assert len(plan.original_query) > 0
            assert len(plan.research_objective) > 0
            assert len(plan.tasks) > 0
            assert plan.estimated_duration_minutes > 0
            assert isinstance(plan.parallel_execution, bool)
            assert len(plan.success_criteria) > 0
            
            for task in plan.tasks:
                assert len(task.task_description) > 0
                assert task.agent_type in ["general_research", "academic_research", "news_research"]
                assert 1 <= task.priority <= 10
                assert len(task.keywords) > 0
        
        @staticmethod
        def assert_valid_research_output(output: ResearchOutput):
            """Assert that research output is valid."""
            assert output.agent_type in ["general_research", "academic_research", "news_research"]
            assert output.sources_searched >= 0
            assert len(output.search_queries_used) > 0
            assert 0.0 <= output.confidence_score <= 1.0
            
            for finding in output.findings:
                assert len(finding.claim) > 0
                assert len(finding.evidence_urls) > 0
                assert len(finding.evidence_titles) == len(finding.evidence_urls)
                assert len(finding.evidence_snippets) == len(finding.evidence_urls)
                assert 0.0 <= finding.confidence <= 1.0
                assert finding.source_count == len(finding.evidence_urls)
        
        @staticmethod
        def assert_valid_synthesis_output(output: SynthesisOutput):
            """Assert that synthesis output is valid."""
            assert len(output.final_answer) > 100  # Substantial response
            assert len(output.key_findings) > 0
            assert len(output.source_urls) > 0
            assert len(output.source_titles) == len(output.source_urls)
            assert 0.0 <= output.confidence_score <= 1.0
            assert output.total_sources == len(output.source_urls)
            assert len(output.agents_used) > 0
        
        @staticmethod
        def assert_valid_conversation_response(response: ConversationResponse):
            """Assert that conversation response is valid."""
            assert len(response.response) > 0
            assert 0.0 <= response.confidence <= 1.0
            assert len(response.reasoning) > 0
            assert response.response_type in [
                "greeting", "explanation", "capability_explanation", 
                "creative", "mathematical", "contextual_explanation", "clarification"
            ]
            assert isinstance(response.suggested_follow_ups, list)
        
        @staticmethod
        def assert_valid_workflow_state(state: Dict[str, Any]):
            """Assert that workflow state is valid."""
            required_fields = ["query", "session_id", "request_id"]
            for field in required_fields:
                assert field in state, f"State missing required field: {field}"
            
            # Check enhanced state fields
            if "classification" in state:
                assert state["classification"] is None or isinstance(state["classification"], dict)
            if "skip_research" in state:
                assert isinstance(state["skip_research"], bool)
    
    return AssertionHelpers()


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for benchmarking tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
        
        def record_metric(self, name: str, value: Any):
            self.metrics[name] = value
        
        def get_summary(self):
            return {
                'duration': self.duration,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
    
    return PerformanceMonitor()


# =============================================================================
# TEMPORARY FILES AND DIRECTORIES
# =============================================================================

@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'test_timeout': 30,
        'max_test_queries': 5,
        'performance_thresholds': {
            'guardrail_max_time': 2.0,
            'planning_max_time': 5.0,
            'research_max_time': 30.0,
            'synthesis_max_time': 10.0,
            'total_workflow_max_time': 60.0
        },
        'quality_thresholds': {
            'min_confidence': 0.6,
            'min_response_length': 100,
            'min_sources': 2,
            'max_response_length': 5000
        }
    }


if __name__ == "__main__":
    """Test fixture definitions when run directly."""
    print("🧪 Research Assistant Test Fixtures")
    print("Available fixtures:")
    
    fixtures = [
        'sample_research_query', 'sample_conversational_query', 'sample_complex_query',
        'sample_guardrail_output', 'sample_research_plan', 'sample_research_findings',
        'sample_research_output', 'sample_synthesis_output', 'sample_conversation_response',
        'sample_initial_state', 'mock_brave_response', 'mock_settings',
        'mock_agent_responses', 'mock_search_tool', 'assertion_helpers',
        'performance_monitor', 'temp_directory', 'test_config'
    ]
    
    for i, fixture in enumerate(fixtures, 1):
        print(f"{i:2d}. {fixture}")
    
    print(f"\nTotal: {len(fixtures)} fixtures available")
    print("Use with: pytest tests/")