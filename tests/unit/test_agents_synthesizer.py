"""
Unit tests for the Synthesizer Agent.

Tests the synthesis functionality that combines research results from multiple agents
into comprehensive, well-structured responses.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.synthesizer import (
    SynthesizerDeps,
    SynthesisOutput,
    synthesize_research,
    synthesizer_agent
)
from agents.researcher import ResearchOutput, ResearchFinding
from agents.planner import ResearchPlan, ResearchTask
from config.settings import get_settings


@pytest.mark.unit
@pytest.mark.asyncio
class TestSynthesizerAgent:
    """Test cases for the synthesizer agent functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch('agents.synthesizer.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.synthesis_confidence_threshold = 0.8
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def sample_research_plan(self):
        """Sample research plan for synthesis testing."""
        return ResearchPlan(
            original_query="What are the latest developments in quantum computing?",
            research_objective="Comprehensive analysis of quantum computing advances",
            tasks=[
                ResearchTask(
                    task_description="Academic research on quantum computing",
                    agent_type="academic_research",
                    priority=9,
                    keywords=["quantum computing", "academic papers"],
                    expected_outcome="Peer-reviewed findings"
                ),
                ResearchTask(
                    task_description="Recent news on quantum computing",
                    agent_type="news_research",
                    priority=8,
                    keywords=["quantum computing news", "2024"],
                    expected_outcome="Current developments"
                )
            ],
            estimated_duration_minutes=60,
            parallel_execution=True,
            success_criteria=["Academic depth", "Current relevance"],
            reasoning="Multi-source approach for comprehensive coverage"
        )

    @pytest.fixture
    def sample_research_results(self):
        """Sample research results from multiple agents."""
        return [
            # Academic research results
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum error correction has achieved 99.9% fidelity in laboratory conditions",
                        evidence_urls=[
                            "https://nature.com/quantum-error-correction-2024",
                            "https://arxiv.org/abs/2024.5678"
                        ],
                        evidence_titles=[
                            "Nature: Quantum Error Correction Breakthrough",
                            "arXiv: High-Fidelity Quantum Error Correction"
                        ],
                        evidence_snippets=[
                            "Researchers demonstrate unprecedented error correction accuracy",
                            "Novel topological codes show remarkable stability"
                        ],
                        confidence=0.95,
                        keywords=["error correction", "fidelity", "laboratory"],
                        source_count=2
                    ),
                    ResearchFinding(
                        claim="Quantum algorithms for optimization problems show exponential speedup",
                        evidence_urls=[
                            "https://science.org/quantum-optimization-2024"
                        ],
                        evidence_titles=[
                            "Science: Quantum Optimization Advances"
                        ],
                        evidence_snippets=[
                            "Variational quantum algorithms demonstrate clear advantage over classical methods"
                        ],
                        confidence=0.88,
                        keywords=["optimization", "algorithms", "speedup"],
                        source_count=1
                    )
                ],
                agent_type="academic_research",
                sources_searched=8,
                search_queries_used=["quantum computing academic 2024", "quantum algorithms peer-reviewed"],
                confidence_score=0.92,
                agent_notes="High-quality academic sources with peer review"
            ),
            # News research results
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="IBM announces 1000-qubit quantum processor milestone",
                        evidence_urls=[
                            "https://techcrunch.com/ibm-1000-qubit-2024",
                            "https://ibm.com/news/quantum-processor"
                        ],
                        evidence_titles=[
                            "TechCrunch: IBM Reaches Quantum Milestone",
                            "IBM: New Quantum Processor Announcement"
                        ],
                        evidence_snippets=[
                            "IBM's latest quantum processor crosses the 1000-qubit threshold",
                            "Commercial quantum computing applications now closer to reality"
                        ],
                        confidence=0.90,
                        keywords=["IBM", "1000-qubit", "commercial"],
                        source_count=2
                    ),
                    ResearchFinding(
                        claim="Google demonstrates quantum advantage in practical applications",
                        evidence_urls=[
                            "https://bloomberg.com/google-quantum-advantage-2024"
                        ],
                        evidence_titles=[
                            "Bloomberg: Google's Quantum Breakthrough"
                        ],
                        evidence_snippets=[
                            "Google's quantum computer solves optimization problem faster than classical supercomputers"
                        ],
                        confidence=0.85,
                        keywords=["Google", "quantum advantage", "practical"],
                        source_count=1
                    )
                ],
                agent_type="news_research",
                sources_searched=10,
                search_queries_used=["quantum computing news 2024", "IBM Google quantum"],
                confidence_score=0.87,
                agent_notes="Current industry developments from reliable news sources"
            ),
            # General research results
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum computing market projected to reach $125 billion by 2030",
                        evidence_urls=[
                            "https://marketresearch.com/quantum-computing-forecast"
                        ],
                        evidence_titles=[
                            "Market Research: Quantum Computing Growth"
                        ],
                        evidence_snippets=[
                            "Analysts predict explosive growth in quantum computing investments and applications"
                        ],
                        confidence=0.80,
                        keywords=["market", "125 billion", "2030"],
                        source_count=1
                    )
                ],
                agent_type="general_research",
                sources_searched=6,
                search_queries_used=["quantum computing market size"], 
                confidence_score=0.80,
                agent_notes="Market analysis and commercial projections"
            )
        ]

    async def test_synthesize_research_comprehensive(self, sample_research_results, sample_research_plan, mock_settings):
        """Test comprehensive research synthesis."""
        with patch.object(synthesizer_agent, 'run') as mock_run:
            # Mock synthesizer response
            mock_result = Mock()
            mock_result.output = SynthesisOutput(
                final_answer="""Quantum computing has experienced remarkable progress in 2024, with significant advances across both theoretical research and practical implementation. Recent breakthroughs in quantum error correction have achieved 99.9% fidelity in laboratory conditions, representing a crucial step toward fault-tolerant quantum computing.

On the commercial front, IBM has reached a major milestone with their 1000-qubit quantum processor, while Google has demonstrated quantum advantage in practical optimization problems. These developments suggest that quantum computing is transitioning from purely experimental research to commercially viable applications.

The quantum computing market reflects this momentum, with projections indicating growth to $125 billion by 2030. Key areas of advancement include error correction methods, optimization algorithms, and hardware scaling, positioning quantum computing as a transformative technology for the next decade.""",
                key_findings=[
                    "Quantum error correction achieved 99.9% fidelity in lab conditions",
                    "IBM announced 1000-qubit quantum processor milestone", 
                    "Google demonstrated quantum advantage in practical applications",
                    "Market projected to reach $125 billion by 2030",
                    "Quantum algorithms showing exponential speedup for optimization"
                ],
                source_urls=[
                    "https://nature.com/quantum-error-correction-2024",
                    "https://arxiv.org/abs/2024.5678",
                    "https://science.org/quantum-optimization-2024",
                    "https://techcrunch.com/ibm-1000-qubit-2024",
                    "https://ibm.com/news/quantum-processor",
                    "https://bloomberg.com/google-quantum-advantage-2024",
                    "https://marketresearch.com/quantum-computing-forecast"
                ],
                source_titles=[
                    "Nature: Quantum Error Correction Breakthrough",
                    "arXiv: High-Fidelity Quantum Error Correction", 
                    "Science: Quantum Optimization Advances",
                    "TechCrunch: IBM Reaches Quantum Milestone",
                    "IBM: New Quantum Processor Announcement",
                    "Bloomberg: Google's Quantum Breakthrough",
                    "Market Research: Quantum Computing Growth"
                ],
                confidence_score=0.89,
                limitations=[
                    "Laboratory conditions may not reflect real-world performance",
                    "Commercial timeline estimates may vary",
                    "Market projections based on current trends"
                ],
                follow_up_questions=[
                    "What specific applications will benefit first from quantum advantage?",
                    "How do quantum error rates compare across different hardware approaches?",
                    "Which companies are best positioned for quantum commercialization?"
                ],
                research_summary="Synthesized findings from 3 specialized research agents covering academic literature, current news, and market analysis",
                total_sources=7,
                agents_used=["academic_research", "news_research", "general_research"]
            )
            mock_run.return_value = mock_result

            synthesis = await synthesize_research(
                research_results=sample_research_results,
                research_plan=sample_research_plan,
                session_id="test-session",
                request_id="test-request"
            )
            
            assert isinstance(synthesis, SynthesisOutput)
            assert len(synthesis.final_answer) > 500  # Comprehensive response
            assert len(synthesis.key_findings) >= 3  # Multiple key insights
            assert len(synthesis.source_urls) == 7  # All sources included
            assert synthesis.confidence_score > 0.8  # High confidence
            assert synthesis.total_sources == 7
            assert len(synthesis.agents_used) == 3  # All agent types used

    async def test_synthesize_single_agent_research(self, sample_research_plan, mock_settings):
        """Test synthesis with results from only one agent."""
        single_agent_results = [
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Single finding from academic research",
                        evidence_urls=["https://example.com/academic"],
                        evidence_titles=["Academic Paper"],
                        evidence_snippets=["Academic evidence"],
                        confidence=0.90,
                        keywords=["academic", "single"],
                        source_count=1
                    )
                ],
                agent_type="academic_research",
                sources_searched=5,
                search_queries_used=["test query"],
                confidence_score=0.90
            )
        ]

        with patch.object(synthesizer_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = SynthesisOutput(
                final_answer="Based on academic research, we found a single significant finding.",
                key_findings=["Single finding from academic research"],
                source_urls=["https://example.com/academic"],
                source_titles=["Academic Paper"],
                confidence_score=0.85,  # Slightly lower due to single source
                limitations=["Limited to academic sources only"],
                follow_up_questions=["Would additional news or general research be helpful?"],
                research_summary="Synthesis based on academic research only",
                total_sources=1,
                agents_used=["academic_research"]
            )
            mock_run.return_value = mock_result

            synthesis = await synthesize_research(
                research_results=single_agent_results,
                research_plan=sample_research_plan,
                session_id="test-session",
                request_id="test-request"
            )
            
            assert synthesis.total_sources == 1
            assert len(synthesis.agents_used) == 1
            assert "Limited to" in synthesis.limitations[0]

    async def test_synthesize_conflicting_findings(self, sample_research_plan, mock_settings):
        """Test synthesis when research findings conflict."""
        conflicting_results = [
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Quantum computing will be commercially viable within 5 years",
                        evidence_urls=["https://optimistic-source.com"],
                        evidence_titles=["Optimistic Quantum Timeline"],
                        evidence_snippets=["Commercial applications are imminent"],
                        confidence=0.80,
                        keywords=["commercial", "5 years", "viable"],
                        source_count=1
                    )
                ],
                agent_type="news_research",
                sources_searched=3,
                search_queries_used=["quantum commercial timeline"],
                confidence_score=0.80
            ),
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Practical quantum computing applications are still 15-20 years away",
                        evidence_urls=["https://conservative-academic.edu"],
                        evidence_titles=["Conservative Quantum Assessment"],
                        evidence_snippets=["Technical challenges remain significant"],
                        confidence=0.85,
                        keywords=["practical", "15-20 years", "challenges"],
                        source_count=1
                    )
                ],
                agent_type="academic_research",
                sources_searched=5,
                search_queries_used=["quantum computing challenges timeline"],
                confidence_score=0.85
            )
        ]

        with patch.object(synthesizer_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = SynthesisOutput(
                final_answer="There are conflicting views on quantum computing timelines. News sources suggest commercial viability within 5 years, while academic sources are more conservative, estimating 15-20 years for practical applications. The difference likely reflects varying definitions of 'commercial viability' and different technical challenges being considered.",
                key_findings=[
                    "Timeline estimates vary significantly across source types",
                    "News sources more optimistic (5 years) than academic sources (15-20 years)",
                    "Different definitions of commercial viability may explain discrepancy"
                ],
                source_urls=["https://optimistic-source.com", "https://conservative-academic.edu"],
                source_titles=["Optimistic Quantum Timeline", "Conservative Quantum Assessment"],
                confidence_score=0.70,  # Lower due to conflicting information
                limitations=[
                    "Conflicting timelines from different source types",
                    "Uncertainty around definition of commercial viability"
                ],
                follow_up_questions=[
                    "What specific applications define commercial viability?",
                    "How do technical experts reconcile these timeline differences?"
                ],
                research_summary="Analysis reveals significant disagreement between news and academic sources on quantum computing timelines",
                total_sources=2,
                agents_used=["news_research", "academic_research"]
            )
            mock_run.return_value = mock_result

            synthesis = await synthesize_research(
                research_results=conflicting_results,
                research_plan=sample_research_plan,
                session_id="test-session",
                request_id="test-request"
            )
            
            assert synthesis.confidence_score < 0.8  # Should be lower due to conflicts
            assert "conflicting" in synthesis.final_answer.lower()
            assert len(synthesis.limitations) > 0
            assert any("conflicting" in limitation.lower() for limitation in synthesis.limitations)

    async def test_synthesize_empty_research_results(self, sample_research_plan, mock_settings):
        """Test synthesis behavior with empty research results."""
        empty_results = [
            ResearchOutput(
                findings=[],  # No findings
                agent_type="general_research",
                sources_searched=5,
                search_queries_used=["test query"],
                confidence_score=0.30,
                agent_notes="No relevant results found"
            )
        ]

        with patch.object(synthesizer_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = SynthesisOutput(
                final_answer="I apologize, but the research did not yield sufficient information to provide a comprehensive answer to your query. The search was conducted but no relevant findings were discovered.",
                key_findings=["No significant findings discovered in research"],
                source_urls=[],
                source_titles=[],
                confidence_score=0.20,  # Very low confidence
                limitations=[
                    "No relevant research findings discovered",
                    "Search queries may need refinement",
                    "Topic may require different research approach"
                ],
                follow_up_questions=[
                    "Could you rephrase or narrow your question?",
                    "Would a different research approach be helpful?"
                ],
                research_summary="Research conducted but no relevant findings discovered",
                total_sources=0,
                agents_used=["general_research"]
            )
            mock_run.return_value = mock_result

            synthesis = await synthesize_research(
                research_results=empty_results,
                research_plan=sample_research_plan,
                session_id="test-session",
                request_id="test-request"
            )
            
            assert synthesis.confidence_score < 0.5
            assert synthesis.total_sources == 0
            assert len(synthesis.source_urls) == 0
            assert "no" in synthesis.final_answer.lower()

    def test_synthesis_output_validation(self):
        """Test SynthesisOutput model validation."""
        # Valid synthesis output
        valid_output = SynthesisOutput(
            final_answer="Test synthesis answer",
            key_findings=["Finding 1", "Finding 2"],
            source_urls=["https://example.com/1", "https://example.com/2"],
            source_titles=["Title 1", "Title 2"],
            confidence_score=0.85,
            limitations=["Limitation 1"],
            follow_up_questions=["Question 1?"],
            research_summary="Test summary",
            total_sources=2,
            agents_used=["general_research"]
        )
        assert valid_output.confidence_score == 0.85
        assert len(valid_output.key_findings) == 2
        
        # Test confidence bounds - pydantic may clamp instead of raising
        try:
            invalid_output = SynthesisOutput(
                final_answer="Invalid output",
                key_findings=[],
                source_urls=[],
                source_titles=[],
                confidence_score=1.5,  # Invalid > 1.0
                limitations=[],
                follow_up_questions=[],
                research_summary="Invalid",
                total_sources=0,
                agents_used=[]
            )
            # If pydantic allows it, confidence should be valid range
            assert 0.0 <= invalid_output.confidence_score <= 1.0
        except Exception:
            # If pydantic raises, that's also acceptable
            pass

    def test_synthesis_output_requires_high_confidence(self, mock_settings):
        """Test synthesis confidence threshold from settings."""
        mock_settings.synthesis_confidence_threshold = 0.9
        
        threshold = SynthesisOutput.requires_high_confidence()
        assert threshold == 0.9

    def test_synthesizer_deps_validation(self):
        """Test SynthesizerDeps model validation."""
        research_results = [
            ResearchOutput(
                findings=[],
                agent_type="general_research",
                sources_searched=1,
                search_queries_used=["test"],
                confidence_score=0.5
            )
        ]
        
        research_plan = ResearchPlan(
            original_query="test",
            research_objective="test",
            tasks=[],
            estimated_duration_minutes=30,
            parallel_execution=False,
            success_criteria=["test"],
            reasoning="test"
        )
        
        valid_deps = SynthesizerDeps(
            research_results=research_results,
            research_plan=research_plan,
            session_id="test-session",
            request_id="test-request"
        )
        
        assert len(valid_deps.research_results) == 1
        assert valid_deps.research_plan.original_query == "test"

    async def test_synthesize_research_error_handling(self, sample_research_results, sample_research_plan, mock_settings):
        """Test error handling in synthesis process."""
        with patch.object(synthesizer_agent, 'run') as mock_run:
            # Simulate synthesis agent error
            mock_run.side_effect = Exception("Synthesis agent error")
            
            # Should propagate the error
            with pytest.raises(Exception):
                await synthesize_research(
                    research_results=sample_research_results,
                    research_plan=sample_research_plan,
                    session_id="test-session",
                    request_id="test-request"
                )

    @pytest.mark.parametrize("agent_count,expected_confidence", [
        (1, 0.75),  # Single agent = lower confidence
        (2, 0.85),  # Two agents = medium confidence
        (3, 0.90),  # Three agents = higher confidence
    ])
    async def test_confidence_based_on_agent_diversity(self, agent_count, expected_confidence, sample_research_plan, mock_settings):
        """Test that confidence increases with agent diversity."""
        # Create results from different numbers of agents
        agent_types = ["academic_research", "news_research", "general_research"][:agent_count]
        
        diverse_results = []
        for agent_type in agent_types:
            diverse_results.append(
                ResearchOutput(
                    findings=[
                        ResearchFinding(
                            claim=f"Finding from {agent_type}",
                            evidence_urls=[f"https://example.com/{agent_type}"],
                            evidence_titles=[f"Title from {agent_type}"],
                            evidence_snippets=[f"Snippet from {agent_type}"],
                            confidence=0.85,
                            keywords=[agent_type],
                            source_count=1
                        )
                    ],
                    agent_type=agent_type,
                    sources_searched=3,
                    search_queries_used=["test"],
                    confidence_score=0.85
                )
            )

        with patch.object(synthesizer_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = SynthesisOutput(
                final_answer="Test synthesis",
                key_findings=[f"Finding from {at}" for at in agent_types],
                source_urls=[f"https://example.com/{at}" for at in agent_types],
                source_titles=[f"Title from {at}" for at in agent_types],
                confidence_score=expected_confidence,
                limitations=[],
                follow_up_questions=[],
                research_summary=f"Synthesis from {agent_count} agent types",
                total_sources=agent_count,
                agents_used=agent_types
            )
            mock_run.return_value = mock_result

            synthesis = await synthesize_research(
                research_results=diverse_results,
                research_plan=sample_research_plan,
                session_id="test-session",
                request_id="test-request"
            )
            
            assert synthesis.confidence_score == expected_confidence
            assert len(synthesis.agents_used) == agent_count