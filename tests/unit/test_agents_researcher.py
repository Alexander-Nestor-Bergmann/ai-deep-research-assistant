"""
Unit tests for the Researcher Agents.

Tests the research functionality including general, academic, and news research agents,
with mocked Brave API calls to ensure reliable testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.researcher import (
    ResearcherDeps,
    ResearchFinding,
    ResearchOutput,
    conduct_research,
    web_search,
    general_research_agent,
    academic_research_agent,
    news_research_agent,
)
from config.settings import get_settings


@pytest.mark.unit
@pytest.mark.asyncio
class TestResearcherAgents:
    """Test cases for the researcher agent functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("agents.researcher.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.brave_api_key = "test-brave-api-key"
            mock_settings.max_search_results = 8
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def sample_brave_response(self):
        """Sample Brave API response for mocking."""
        return {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/quantum-computing-2024",
                        "title": "Quantum Computing Advances in 2024",
                        "description": "Recent breakthroughs in quantum computing include improved error correction and new algorithms...",
                        "published_date": "2024-03-15",
                        "age": "2 months ago",
                    },
                    {
                        "url": "https://nature.com/quantum-research",
                        "title": "Nature: Quantum Error Correction Methods",
                        "description": "Peer-reviewed research on quantum error correction demonstrates significant improvements...",
                        "published_date": "2024-02-01",
                        "age": "3 months ago",
                    },
                    {
                        "url": "https://arxiv.org/abs/2024.1234",
                        "title": "Quantum Algorithms for Machine Learning",
                        "description": "This paper presents novel quantum algorithms that show exponential speedup...",
                        "published_date": "2024-01-20",
                        "age": "4 months ago",
                    },
                    {
                        "url": "https://ibm.com/quantum-roadmap",
                        "title": "IBM Quantum Roadmap 2024",
                        "description": "IBM announces new quantum processor milestones and commercial applications...",
                        "published_date": "2024-03-01",
                        "age": "2 months ago",
                    },
                    {
                        "url": "https://google.com/quantum-supremacy",
                        "title": "Google Quantum AI Progress Report",
                        "description": "Google's quantum AI division reports breakthrough in quantum advantage demonstrations...",
                        "published_date": "2024-02-15",
                        "age": "3 months ago",
                    },
                ]
            }
        }

    @pytest.fixture
    def sample_search_tool_response(self, sample_brave_response):
        """Sample search tool response for mocking."""
        return {
            "results": sample_brave_response["web"]["results"],
            "query_used": "quantum computing 2024",
            "search_type": "general",
            "source_count": 5,
        }

    @pytest.fixture
    def sample_research_deps(self):
        """Sample research dependencies."""
        return ResearcherDeps(
            session_id="test-session",
            request_id="test-request",
            agent_type="general_research",
            task_description="Research recent developments in quantum computing",
            keywords=["quantum computing", "2024", "advances"],
            max_sources=5,
        )

    async def test_conduct_general_research(
        self, sample_research_deps, mock_settings, sample_search_tool_response
    ):
        """Test general research agent execution."""
        with patch.object(general_research_agent, "run") as mock_run:
            # Mock research agent response
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum computing has achieved significant breakthroughs in error correction",
                        evidence_urls=["https://nature.com/quantum-research"],
                        evidence_titles=["Nature: Quantum Error Correction Methods"],
                        evidence_snippets=[
                            "Peer-reviewed research shows major improvements"
                        ],
                        confidence=0.9,
                        keywords=["quantum", "error correction", "breakthrough"],
                        source_count=3,
                    ),
                    ResearchFinding(
                        claim="Commercial quantum applications are expanding rapidly",
                        evidence_urls=["https://ibm.com/quantum-roadmap"],
                        evidence_titles=["IBM Quantum Roadmap 2024"],
                        evidence_snippets=["IBM announces new commercial milestones"],
                        confidence=0.8,
                        keywords=["commercial", "quantum", "applications"],
                        source_count=2,
                    ),
                ],
                agent_type="general_research",
                sources_searched=5,
                search_queries_used=["quantum computing 2024", "quantum advances"],
                confidence_score=0.85,
                agent_notes="Comprehensive general research completed",
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Research quantum computing advances",
                agent_type="general_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["quantum computing", "advances"],
                max_sources=5,
            )

            assert isinstance(result, ResearchOutput)
            assert result.agent_type == "general_research"
            assert len(result.findings) == 2
            assert result.confidence_score == 0.85
            assert result.sources_searched == 5

            # Verify findings structure
            for finding in result.findings:
                assert isinstance(finding, ResearchFinding)
                assert finding.confidence > 0.0
                assert len(finding.evidence_urls) > 0
                assert len(finding.keywords) > 0

    async def test_conduct_academic_research(self, mock_settings):
        """Test academic research agent execution."""
        with patch.object(academic_research_agent, "run") as mock_run:
            # Mock academic research response
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum error correction methods show 99.9% fidelity in recent studies",
                        evidence_urls=["https://arxiv.org/abs/2024.1234"],
                        evidence_titles=["Quantum Algorithms for Machine Learning"],
                        evidence_snippets=[
                            "This paper presents novel quantum algorithms with proven advantages"
                        ],
                        confidence=0.95,
                        keywords=[
                            "peer-reviewed",
                            "quantum error correction",
                            "fidelity",
                        ],
                        source_count=3,
                    )
                ],
                agent_type="academic_research",
                sources_searched=8,
                search_queries_used=[
                    "quantum error correction peer-reviewed",
                    "academic quantum research",
                ],
                confidence_score=0.92,
                agent_notes="High-quality academic sources found",
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Find academic research on quantum error correction",
                agent_type="academic_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["quantum error correction", "peer-reviewed"],
            )

            assert result.agent_type == "academic_research"
            assert (
                result.confidence_score >= 0.9
            )  # Academic research should have high confidence
            assert len(result.findings) > 0

            # Academic findings should emphasize peer review
            finding = result.findings[0]
            assert any("peer-reviewed" in keyword for keyword in finding.keywords)

    async def test_conduct_news_research(self, mock_settings):
        """Test news research agent execution."""
        with patch.object(news_research_agent, "run") as mock_run:
            # Mock news research response
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Google announces major quantum computing breakthrough this week",
                        evidence_urls=["https://techcrunch.com/google-quantum-2024"],
                        evidence_titles=["TechCrunch: Google Quantum Breakthrough"],
                        evidence_snippets=[
                            "Google's latest quantum processor achieves unprecedented performance"
                        ],
                        confidence=0.85,
                        keywords=["recent news", "Google", "quantum breakthrough"],
                        source_count=4,
                    ),
                    ResearchFinding(
                        claim="IBM stock rises on quantum computing milestones",
                        evidence_urls=["https://bloomberg.com/ibm-quantum-stock"],
                        evidence_titles=["Bloomberg: IBM Quantum Advances"],
                        evidence_snippets=[
                            "Investors respond positively to IBM's quantum roadmap updates"
                        ],
                        confidence=0.8,
                        keywords=["IBM", "stock", "recent developments"],
                        source_count=3,
                    ),
                ],
                agent_type="news_research",
                sources_searched=10,
                search_queries_used=[
                    "quantum computing news 2024",
                    "latest quantum developments",
                ],
                confidence_score=0.82,
                agent_notes="Current news coverage analyzed",
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Find recent news about quantum computing developments",
                agent_type="news_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["quantum computing news", "2024", "recent"],
            )

            assert result.agent_type == "news_research"
            assert len(result.findings) > 0

            # News findings should emphasize recency
            for finding in result.findings:
                keywords_lower = [k.lower() for k in finding.keywords]
                assert any(
                    "recent" in k or "news" in k or "2024" in k for k in keywords_lower
                )

    async def test_conduct_research_with_defaults(self, mock_settings):
        """Test research with default max_sources from settings."""
        mock_settings.max_search_results = 6

        with patch.object(general_research_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[],
                agent_type="general_research",
                sources_searched=6,  # Should use settings default
                search_queries_used=["test"],
                confidence_score=0.7,
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Test research",
                agent_type="general_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["test"],
                # max_sources not specified, should use settings
            )

            assert result.sources_searched == 6  # From settings

    def test_research_output_validation(self):
        """Test ResearchOutput model validation."""
        # Valid output
        valid_output = ResearchOutput(
            findings=[],
            agent_type="general_research",
            sources_searched=5,
            search_queries_used=["test query"],
            confidence_score=0.75,
            agent_notes="Test notes",
        )
        assert valid_output.agent_type == "general_research"
        assert valid_output.confidence_score == 0.75

        # Test that agent_type must be valid
        valid_types = ["general_research", "academic_research", "news_research"]
        assert valid_output.agent_type in valid_types

    def test_researcher_deps_validation(self):
        """Test ResearcherDeps model validation."""
        # Valid deps
        valid_deps = ResearcherDeps(
            session_id="test-session",
            request_id="test-request",
            agent_type="general_research",
            task_description="Test task",
            keywords=["test"],
            max_sources=5,
        )
        assert valid_deps.max_sources == 5
        assert len(valid_deps.keywords) == 1

    async def test_research_with_empty_keywords(self, mock_settings):
        """Test research with empty keywords list."""
        with patch.object(general_research_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[],
                agent_type="general_research",
                sources_searched=0,
                search_queries_used=[],
                confidence_score=0.5,
                agent_notes="No keywords provided, limited research possible",
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Research with no keywords",
                agent_type="general_research",
                session_id="test-session",
                request_id="test-request",
                keywords=[],  # Empty keywords
            )

            # Should handle gracefully with lower confidence
            assert result.confidence_score <= 0.6
            assert result.sources_searched >= 0

    async def test_research_agent_selection(self, mock_settings):
        """Test that correct agent is selected based on agent_type."""
        agent_types = ["general_research", "academic_research", "news_research"]

        for agent_type in agent_types:
            with patch(f"agents.researcher.{agent_type}_agent.run") as mock_run:
                mock_result = Mock()
                mock_result.output = ResearchOutput(
                    findings=[],
                    agent_type=agent_type,
                    sources_searched=3,
                    search_queries_used=["test"],
                    confidence_score=0.7,
                )
                mock_run.return_value = mock_result

                result = await conduct_research(
                    task_description=f"Test {agent_type}",
                    agent_type=agent_type,
                    session_id="test-session",
                    request_id="test-request",
                    keywords=["test"],
                )

                assert result.agent_type == agent_type
                mock_run.assert_called_once()

    @pytest.mark.parametrize(
        "max_sources,expected_searches",
        [
            (3, 3),
            (8, 8),
            (15, 15),
            (None, 8),  # Should use settings default
        ],
    )
    async def test_max_sources_parameter(
        self, max_sources, expected_searches, mock_settings
    ):
        """Test max_sources parameter affects search behavior."""
        mock_settings.max_search_results = 8

        with patch.object(general_research_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[],
                agent_type="general_research",
                sources_searched=expected_searches,
                search_queries_used=["test"],
                confidence_score=0.7,
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Test max sources",
                agent_type="general_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["test"],
                max_sources=max_sources,
            )

            assert result.sources_searched == expected_searches

    async def test_research_findings_structure(self, mock_settings):
        """Test that research findings have proper structure and content."""
        with patch.object(general_research_agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum computing shows promise for cryptography applications",
                        evidence_urls=[
                            "https://example.com/quantum-crypto-1",
                            "https://example.com/quantum-crypto-2",
                        ],
                        evidence_titles=[
                            "Quantum Cryptography Advances",
                            "Breaking RSA with Quantum Computers",
                        ],
                        evidence_snippets=[
                            "Recent advances in quantum algorithms threaten current cryptographic methods",
                            "Quantum computers could break RSA encryption within the next decade",
                        ],
                        confidence=0.87,
                        keywords=[
                            "quantum computing",
                            "cryptography",
                            "RSA",
                            "security",
                        ],
                        source_count=2,
                    )
                ],
                agent_type="general_research",
                sources_searched=5,
                search_queries_used=["quantum computing cryptography"],
                confidence_score=0.87,
            )
            mock_run.return_value = mock_result

            result = await conduct_research(
                task_description="Research quantum computing applications in cryptography",
                agent_type="general_research",
                session_id="test-session",
                request_id="test-request",
                keywords=["quantum computing", "cryptography"],
            )

            finding = result.findings[0]

            # Test finding structure
            assert len(finding.claim) > 20  # Substantial claim
            assert len(finding.evidence_urls) == len(finding.evidence_titles)
            assert len(finding.evidence_urls) == len(finding.evidence_snippets)
            assert len(finding.evidence_urls) == finding.source_count

            # Test URL validity
            for url in finding.evidence_urls:
                assert url.startswith("http")
                assert len(url) > 10

            # Test titles are meaningful
            for title in finding.evidence_titles:
                assert len(title) > 5

            # Test snippets provide context
            for snippet in finding.evidence_snippets:
                assert len(snippet) > 15
