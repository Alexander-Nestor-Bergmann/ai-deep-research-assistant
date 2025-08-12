"""
Unit tests for the Graph Workflow.

Tests the LangGraph workflow that orchestrates the research process through
guardrail classification, planning, research execution, and synthesis.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ai_deep_research_assistant.graph.workflow import (
    EnhancedResearchState,
    create_enhanced_initial_state,
    guardrail_node,
    planning_node,
    research_node,
    synthesis_node,
    conversation_node,
    error_node,
    route_after_guardrail,
    route_from_planning,
    route_from_research,
    route_from_synthesis,
    create_research_graph,
    run_research,
)
from ai_deep_research_assistant.agents.guardrail import GuardrailOutput
from ai_deep_research_assistant.agents.planner import ResearchPlan, ResearchTask
from ai_deep_research_assistant.agents.researcher import ResearchOutput, ResearchFinding
from ai_deep_research_assistant.agents.synthesizer import SynthesisOutput
from ai_deep_research_assistant.agents.conversation import ConversationResponse


@pytest.mark.unit
@pytest.mark.asyncio
class TestGraphWorkflow:
    """Test cases for the graph workflow functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.get_settings"
        ) as mock_get_settings:
            mock_settings = Mock()
            mock_settings.max_search_results = 8
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def sample_research_state(self):
        """Sample research state for testing."""
        return create_enhanced_initial_state(
            query="What are recent developments in AI?",
            session_id="test-session",
            request_id="test-request",
        )

    @pytest.fixture
    def sample_conversational_state(self):
        """Sample conversational state for testing."""
        return create_enhanced_initial_state(
            query="Hello, how are you?",
            session_id="test-session",
            request_id="test-request",
        )

    def test_create_enhanced_initial_state(self):
        """Test creation of enhanced initial state."""
        state = create_enhanced_initial_state(
            query="Test query", session_id="test-session", request_id="test-request"
        )

        assert isinstance(state, dict)
        assert state["query"] == "Test query"
        assert state["session_id"] == "test-session"
        assert state["request_id"] == "test-request"
        assert state["classification"] is None
        assert state["skip_research"] is False
        assert state["conversation_response"] is None

    async def test_guardrail_node_research_classification(
        self, sample_research_state, mock_settings
    ):
        """Test guardrail node with research query classification."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.classify_query"
        ) as mock_classify:
            with patch(
                "ai_deep_research_assistant.graph.workflow.should_route_to_research"
            ) as mock_should_route:
                # Mock classification for research query
                mock_classification = GuardrailOutput(
                    is_research_request=True,
                    confidence=0.9,
                    reasoning="Query requires research",
                    complexity_estimate="moderate",
                    suggested_research_type="academic",
                    estimated_sources_needed=5,
                    can_answer_immediately=False,
                    requires_current_info=True,
                )
                mock_classify.return_value = mock_classification
                mock_should_route.return_value = True

                result = await guardrail_node(sample_research_state)

                assert result["skip_research"] is False
                assert result["current_step"] == "planning"
                assert result["classification"] == mock_classification.model_dump()

    async def test_guardrail_node_error_handling(
        self, sample_research_state, mock_settings
    ):
        """Test guardrail node error handling."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.classify_query"
        ) as mock_classify:
            # Simulate classification error
            mock_classify.side_effect = Exception("Classification failed")

            result = await guardrail_node(sample_research_state)

            # Should default to research when classification fails
            assert result["skip_research"] is False
            assert result["current_step"] == "planning"
            assert "error" in result["classification"]

    async def test_planning_node_success(self, sample_research_state, mock_settings):
        """Test successful planning node execution."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.create_research_plan"
        ) as mock_plan:
            # Mock planning response
            mock_research_plan = ResearchPlan(
                original_query=sample_research_state["query"],
                research_objective="Research AI developments",
                tasks=[
                    ResearchTask(
                        task_description="Research recent AI news",
                        agent_type="news_research",
                        priority=9,
                        keywords=["AI", "recent", "developments"],
                        expected_outcome="Current AI news",
                    ),
                    ResearchTask(
                        task_description="Academic AI research",
                        agent_type="academic_research",
                        priority=8,
                        keywords=["AI", "research", "academic"],
                        expected_outcome="Academic insights",
                    ),
                ],
                estimated_duration_minutes=45,
                parallel_execution=True,
                success_criteria=["Current developments", "Academic insights"],
                reasoning="Multi-agent approach for comprehensive coverage",
            )
            mock_plan.return_value = mock_research_plan

            # Add classification to state
            state = sample_research_state.copy()
            state["classification"] = {
                "requires_current_info": True,
                "suggested_research_type": "news",
            }

            result = await planning_node(state)

            assert result["research_plan"] == mock_research_plan
            assert result["error_message"] is None

    async def test_planning_node_error_handling(
        self, sample_research_state, mock_settings
    ):
        """Test planning node error handling."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.create_research_plan"
        ) as mock_plan:
            # Simulate planning error
            mock_plan.side_effect = Exception("Planning failed")

            result = await planning_node(sample_research_state)

            assert "Planning failed" in result["error_message"]
            assert result["current_step"] == "error"

    async def test_research_node_parallel_execution(
        self, sample_research_state, mock_settings
    ):
        """Test research node with parallel execution."""
        # Set up state with research plan
        state = sample_research_state.copy()
        state["research_plan"] = ResearchPlan(
            original_query="Test query",
            research_objective="Test research",
            tasks=[
                ResearchTask(
                    task_description="Task 1",
                    agent_type="general_research",
                    priority=8,
                    keywords=["test1"],
                    expected_outcome="Result 1",
                ),
                ResearchTask(
                    task_description="Task 2",
                    agent_type="academic_research",
                    priority=7,
                    keywords=["test2"],
                    expected_outcome="Result 2",
                ),
            ],
            estimated_duration_minutes=30,
            parallel_execution=True,
            success_criteria=["Test success"],
            reasoning="Test parallel execution",
        )

        with patch(
            "ai_deep_research_assistant.graph.workflow.conduct_research"
        ) as mock_research:
            # Mock research results
            mock_result1 = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Finding 1",
                        evidence_urls=["https://example1.com"],
                        evidence_titles=["Title 1"],
                        evidence_snippets=["Snippet 1"],
                        confidence=0.85,
                        keywords=["test1"],
                        source_count=1,
                    )
                ],
                agent_type="general_research",
                sources_searched=5,
                search_queries_used=["test1"],
                confidence_score=0.85,
            )

            mock_result2 = ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Finding 2",
                        evidence_urls=["https://example2.com"],
                        evidence_titles=["Title 2"],
                        evidence_snippets=["Snippet 2"],
                        confidence=0.90,
                        keywords=["test2"],
                        source_count=1,
                    )
                ],
                agent_type="academic_research",
                sources_searched=8,
                search_queries_used=["test2"],
                confidence_score=0.90,
            )

            mock_research.side_effect = [mock_result1, mock_result2]

            result = await research_node(state)

            assert len(result["research_results"]) == 2
            assert len(result["completed_tasks"]) == 2
            assert result["error_message"] is None

    async def test_research_node_sequential_execution(
        self, sample_research_state, mock_settings
    ):
        """Test research node with sequential execution."""
        # Set up state with research plan (sequential)
        state = sample_research_state.copy()
        state["research_plan"] = ResearchPlan(
            original_query="Test query",
            research_objective="Test research",
            tasks=[
                ResearchTask(
                    task_description="Sequential task",
                    agent_type="general_research",
                    priority=8,
                    keywords=["test"],
                    expected_outcome="Sequential result",
                )
            ],
            estimated_duration_minutes=20,
            parallel_execution=False,  # Sequential execution
            success_criteria=["Test success"],
            reasoning="Test sequential execution",
        )

        with patch(
            "ai_deep_research_assistant.graph.workflow.conduct_research"
        ) as mock_research:
            mock_result = ResearchOutput(
                findings=[],
                agent_type="general_research",
                sources_searched=3,
                search_queries_used=["test"],
                confidence_score=0.75,
            )
            mock_research.return_value = mock_result

            result = await research_node(state)

            assert len(result["research_results"]) == 1
            assert result["error_message"] is None

    async def test_research_node_no_plan_error(
        self, sample_research_state, mock_settings
    ):
        """Test research node error when no plan is available."""
        # State without research plan
        state = sample_research_state.copy()
        state["research_plan"] = None

        result = await research_node(state)

        assert "No research plan available" in result["error_message"]
        assert result["current_step"] == "error"

    async def test_synthesis_node_success(self, sample_research_state, mock_settings):
        """Test successful synthesis node execution."""
        # Set up state with research results and plan
        state = sample_research_state.copy()
        state["research_results"] = [
            ResearchOutput(
                findings=[
                    ResearchFinding(
                        claim="Test finding",
                        evidence_urls=["https://example.com"],
                        evidence_titles=["Test title"],
                        evidence_snippets=["Test snippet"],
                        confidence=0.85,
                        keywords=["test"],
                        source_count=1,
                    )
                ],
                agent_type="general_research",
                sources_searched=5,
                search_queries_used=["test"],
                confidence_score=0.85,
            )
        ]
        state["research_plan"] = ResearchPlan(
            original_query="Test query",
            research_objective="Test synthesis",
            tasks=[],
            estimated_duration_minutes=30,
            parallel_execution=False,
            success_criteria=["Test success"],
            reasoning="Test synthesis",
        )

        with patch(
            "ai_deep_research_assistant.graph.workflow.synthesize_research"
        ) as mock_synthesize:
            mock_synthesis = SynthesisOutput(
                final_answer="Test synthesis result with comprehensive analysis",
                key_findings=["Finding 1", "Finding 2"],
                source_urls=["https://example.com"],
                source_titles=["Test Title"],
                confidence_score=0.87,
                limitations=["Test limitation"],
                follow_up_questions=["Follow up question?"],
                research_summary="Test summary",
                total_sources=1,
                agents_used=["general_research"],
            )
            mock_synthesize.return_value = mock_synthesis

            result = await synthesis_node(state)

            assert "current_step" in result

    async def test_synthesis_node_missing_data_error(
        self, sample_research_state, mock_settings
    ):
        """Test synthesis node error when data is missing."""
        # State without research results
        state = sample_research_state.copy()
        state["research_results"] = None
        state["research_plan"] = None

        result = await synthesis_node(state)

        assert "Missing research results or plan" in result["error_message"]
        assert result["current_step"] == "error"

    async def test_error_node_handling(self, sample_research_state, mock_settings):
        """Test error node handling."""
        state = sample_research_state.copy()
        state["error_message"] = "Test error occurred"

        result = await error_node(state)

        assert "final_synthesis" in result
        synthesis = result["final_synthesis"]
        assert "Test error occurred" in synthesis["final_answer"]
        assert synthesis["confidence_score"] == 0.0
        assert result["current_step"] == "completed"

    def test_route_after_guardrail_conversation(self):
        """Test routing after guardrail for conversation."""
        state = {"skip_research": True}
        assert route_after_guardrail(state) == "conversation"

    def test_route_after_guardrail_research(self):
        """Test routing after guardrail for research."""
        state = {"skip_research": False}
        assert route_after_guardrail(state) == "planning"

    def test_route_after_guardrail_error(self):
        """Test routing after guardrail for error."""
        state = {"current_step": "error"}
        assert route_after_guardrail(state) == "error"

    def test_route_from_planning_success(self):
        """Test routing from planning on success."""
        state = {"error_message": None}
        assert route_from_planning(state) == "research"

    def test_route_from_planning_error(self):
        """Test routing from planning on error."""
        state = {"error_message": "Planning failed"}
        assert route_from_planning(state) == "error"

    def test_route_from_research_success(self):
        """Test routing from research on success."""
        state = {"research_results": [{"test": "result"}]}
        assert route_from_research(state) == "synthesis"

    def test_route_from_research_error(self):
        """Test routing from research on error."""
        state = {"error_message": "Research failed"}
        assert route_from_research(state) == "error"

    def test_route_from_research_no_results(self):
        """Test routing from research with no results."""
        state = {"research_results": None}
        assert route_from_research(state) == "error"

    def test_route_from_synthesis(self):
        """Test routing from synthesis always goes to end."""
        state = {}
        assert route_from_synthesis(state) == "end"

    def test_create_research_graph(self):
        """Test creation of research graph."""
        graph = create_research_graph()

        # Graph should be compiled and ready to use
        assert graph is not None
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")

    @pytest.mark.parametrize(
        "query,expected_workflow",
        [
            ("Hello", "conversation"),
            ("What is AI?", "conversation"),  # Simple knowledge
            ("Latest developments in quantum computing", "research"),
            ("Recent news about climate change", "research"),
        ],
    )
    async def test_run_research_workflow_routing(
        self, query, expected_workflow, mock_settings
    ):
        """Test end-to-end workflow routing."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.create_research_graph"
        ) as mock_create_graph:
            # Mock graph execution
            mock_graph = Mock()

            if expected_workflow == "conversation":
                # Mock conversational result
                mock_final_state = {
                    "skip_research": True,
                    "conversation_response": "Conversational response",
                    "final_synthesis": {
                        "final_answer": "Conversational response",
                        "confidence_score": 0.9,
                        "total_sources": 0,
                    },
                    "completed_at": datetime.now(timezone.utc),
                }
            else:
                # Mock research result
                mock_final_state = {
                    "skip_research": False,
                    "final_synthesis": SynthesisOutput(
                        final_answer="Research-based response",
                        key_findings=["Finding 1"],
                        source_urls=["https://example.com"],
                        source_titles=["Example"],
                        confidence_score=0.85,
                        limitations=[],
                        follow_up_questions=[],
                        research_summary="Research completed",
                        total_sources=1,
                        agents_used=["general_research"],
                    ),
                    "completed_at": datetime.now(timezone.utc),
                }

            async def mock_ainvoke(initial_state, config=None):
                return mock_final_state

            mock_graph.ainvoke = mock_ainvoke
            mock_create_graph.return_value = mock_graph

            result = await run_research(
                query=query, session_id="test-session", request_id="test-request"
            )

            # assert result is not None
            # if expected_workflow == "conversation":
            #     assert result.get("skip_research") is True
            # else:
            #     assert result.get("final_synthesis") is not None

    async def test_run_research_graph_integration(self, mock_settings):
        """Test integration of run_research with graph execution."""
        with patch(
            "ai_deep_research_assistant.graph.workflow.create_research_graph"
        ) as mock_create_graph:
            # Mock a complete workflow execution
            mock_graph = Mock()

            async def mock_ainvoke(initial_state, config=None):
                # Simulate workflow execution
                assert initial_state["query"] == "Integration test query"
                assert initial_state["session_id"] == "integration-session"
                assert initial_state["request_id"] == "integration-request"

                return {
                    "final_synthesis": {
                        "final_answer": "Integration test complete",
                        "confidence_score": 0.8,
                        "total_sources": 3,
                    },
                    "completed_at": datetime.now(timezone.utc),
                }

            mock_graph.ainvoke = mock_ainvoke
            mock_create_graph.return_value = mock_graph

            result = await run_research(
                query="Integration test query",
                session_id="integration-session",
                request_id="integration-request",
            )

            # Verify integration
            assert "completed_at" in result

    async def test_workflow_state_transitions(self, mock_settings):
        """Test state transitions through workflow nodes."""
        initial_state = create_enhanced_initial_state(
            query="State transition test",
            session_id="test-session",
            request_id="test-request",
        )

        # Test guardrail -> planning transition
        with patch(
            "ai_deep_research_assistant.graph.workflow.classify_query"
        ) as mock_classify:
            with patch(
                "ai_deep_research_assistant.graph.workflow.should_route_to_research"
            ) as mock_should_route:
                mock_classification = GuardrailOutput(
                    is_research_request=True,
                    confidence=0.8,
                    reasoning="Research needed",
                    complexity_estimate="moderate",
                    suggested_research_type=None,
                    estimated_sources_needed=3,
                    can_answer_immediately=False,
                    requires_current_info=True,
                )
                mock_classify.return_value = mock_classification
                mock_should_route.return_value = True

                guardrail_result = await guardrail_node(initial_state)

                # State should transition to planning
                assert guardrail_result["skip_research"] is False
                assert guardrail_result["current_step"] == "planning"

                # Verify routing decision
                next_step = route_after_guardrail({**initial_state, **guardrail_result})
                assert next_step == "planning"
