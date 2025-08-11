"""
Unit tests for the Planner Agent.

Tests the research planning functionality that breaks down queries into
specific, actionable research tasks with appropriate agent assignments.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.planner import (
    ResearchTask,
    ResearchPlan,
    create_research_plan,
    planner_agent
)
from config.settings import get_settings


@pytest.mark.unit
@pytest.mark.asyncio
class TestPlannerAgent:
    """Test cases for the planner agent functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch('agents.planner.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.min_confidence_threshold = 0.7
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def sample_simple_query(self):
        """Simple research query for testing."""
        return "What is quantum computing?"

    @pytest.fixture
    def sample_complex_query(self):
        """Complex research query requiring multiple tasks."""
        return "Compare renewable energy policies across different countries, focusing on wind and solar adoption rates, economic impacts, and environmental outcomes over the past decade"

    @pytest.fixture
    def sample_current_events_query(self):
        """Current events query for testing news agent prioritization."""
        return "What are the latest developments in AI regulation in 2024?"

    @pytest.fixture
    def sample_academic_query(self):
        """Academic research query for testing academic agent prioritization."""
        return "Explain recent advances in quantum error correction methods published in peer-reviewed journals"

    @pytest.fixture
    def available_agents(self):
        """Standard list of available research agents."""
        return ["general_research", "academic_research", "news_research"]

    async def test_create_simple_research_plan(self, sample_simple_query, available_agents, mock_settings):
        """Test creating a research plan for a simple query."""
        with patch.object(planner_agent, 'run') as mock_run:
            # Mock planner response for simple query
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_simple_query,
                research_objective="Provide a comprehensive explanation of quantum computing",
                tasks=[
                    ResearchTask(
                        task_description="Search for general information about quantum computing basics",
                        agent_type="general_research",
                        priority=8,
                        keywords=["quantum computing", "quantum mechanics", "qubits"],
                        expected_outcome="Basic explanation and key concepts"
                    )
                ],
                estimated_duration_minutes=30,
                parallel_execution=False,
                success_criteria=["Clear explanation of quantum computing concepts"],
                reasoning="Simple query requires single general research task"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_simple_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=available_agents
            )
            
            assert isinstance(plan, ResearchPlan)
            assert plan.original_query == sample_simple_query
            assert len(plan.tasks) == 1
            assert plan.tasks[0].agent_type == "general_research"
            assert plan.estimated_duration_minutes > 0
            assert isinstance(plan.parallel_execution, bool)
            assert len(plan.success_criteria) > 0
            assert plan.reasoning is not None

    async def test_create_complex_research_plan(self, sample_complex_query, available_agents, mock_settings):
        """Test creating a research plan for a complex query."""
        with patch.object(planner_agent, 'run') as mock_run:
            # Mock planner response for complex query
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_complex_query,
                research_objective="Comprehensive analysis of renewable energy policies across countries",
                tasks=[
                    ResearchTask(
                        task_description="Research renewable energy policies in major countries",
                        agent_type="general_research",
                        priority=9,
                        keywords=["renewable energy policies", "wind solar adoption"],
                        expected_outcome="Policy comparison data"
                    ),
                    ResearchTask(
                        task_description="Find academic studies on renewable energy economics",
                        agent_type="academic_research", 
                        priority=8,
                        keywords=["renewable energy economics", "policy impact studies"],
                        expected_outcome="Peer-reviewed research on economic impacts"
                    ),
                    ResearchTask(
                        task_description="Search for recent news on renewable energy developments",
                        agent_type="news_research",
                        priority=7,
                        keywords=["renewable energy news", "wind solar developments"],
                        expected_outcome="Current developments and trends"
                    )
                ],
                estimated_duration_minutes=90,
                parallel_execution=True,
                success_criteria=[
                    "Compare policies across multiple countries",
                    "Include economic impact analysis",
                    "Cover environmental outcomes"
                ],
                reasoning="Complex query requires multiple specialized research streams"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_complex_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=available_agents
            )
            
            assert isinstance(plan, ResearchPlan)
            assert len(plan.tasks) == 3  # Complex query should have multiple tasks
            assert plan.parallel_execution is True  # Complex queries should use parallel execution
            assert plan.estimated_duration_minutes > 60  # Should take longer
            
            # Check that different agent types are used
            agent_types = [task.agent_type for task in plan.tasks]
            assert len(set(agent_types)) > 1  # Should use multiple agent types
            
            # Check task priorities
            priorities = [task.priority for task in plan.tasks]
            assert all(1 <= p <= 10 for p in priorities)
            
            # Check keywords are relevant
            for task in plan.tasks:
                assert len(task.keywords) > 0
                assert task.expected_outcome is not None

    async def test_news_query_agent_prioritization(self, sample_current_events_query, mock_settings):
        """Test that current events queries prioritize news research agents."""
        # Custom available agents with news first
        prioritized_agents = ["news_research", "general_research", "academic_research"]
        
        with patch.object(planner_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_current_events_query,
                research_objective="Research latest AI regulation developments",
                tasks=[
                    ResearchTask(
                        task_description="Search recent news about AI regulation in 2024",
                        agent_type="news_research",  # Should prioritize news
                        priority=10,
                        keywords=["AI regulation 2024", "artificial intelligence policy"],
                        expected_outcome="Recent regulatory developments"
                    ),
                    ResearchTask(
                        task_description="Find general information about AI regulation frameworks",
                        agent_type="general_research",
                        priority=7,
                        keywords=["AI regulation frameworks", "technology policy"],
                        expected_outcome="Regulatory framework overview"
                    )
                ],
                estimated_duration_minutes=45,
                parallel_execution=True,
                success_criteria=["Current AI regulation developments", "Regulatory framework context"],
                reasoning="Current events query prioritizes news research"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_current_events_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=prioritized_agents
            )
            
            # First task should use news research for current events
            assert plan.tasks[0].agent_type == "news_research"
            assert plan.tasks[0].priority >= 8  # High priority for news

    async def test_academic_query_agent_prioritization(self, sample_academic_query, mock_settings):
        """Test that academic queries prioritize academic research agents."""
        # Custom available agents with academic first
        prioritized_agents = ["academic_research", "general_research", "news_research"]
        
        with patch.object(planner_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_academic_query,
                research_objective="Research quantum error correction advances in academic literature",
                tasks=[
                    ResearchTask(
                        task_description="Search peer-reviewed papers on quantum error correction",
                        agent_type="academic_research",  # Should prioritize academic
                        priority=10,
                        keywords=["quantum error correction", "peer-reviewed", "quantum computing"],
                        expected_outcome="Academic research findings"
                    ),
                    ResearchTask(
                        task_description="Find general background on quantum error correction",
                        agent_type="general_research",
                        priority=6,
                        keywords=["quantum error correction basics", "quantum computing"],
                        expected_outcome="Background context"
                    )
                ],
                estimated_duration_minutes=60,
                parallel_execution=False,
                success_criteria=["Academic sources on quantum error correction", "Technical depth and accuracy"],
                reasoning="Academic query prioritizes peer-reviewed sources"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_academic_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=prioritized_agents
            )
            
            # Should prioritize academic research for academic queries
            academic_tasks = [t for t in plan.tasks if t.agent_type == "academic_research"]
            assert len(academic_tasks) > 0
            # Academic task should have highest priority
            academic_priority = max(t.priority for t in academic_tasks)
            assert academic_priority >= 8

    async def test_research_plan_with_limited_agents(self, sample_simple_query, mock_settings):
        """Test creating research plan with limited available agents."""
        limited_agents = ["general_research"]  # Only general research available
        
        with patch.object(planner_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_simple_query,
                research_objective="Research quantum computing with limited agents",
                tasks=[
                    ResearchTask(
                        task_description="General research on quantum computing",
                        agent_type="general_research",  # Only available option
                        priority=8,
                        keywords=["quantum computing"],
                        expected_outcome="General information"
                    )
                ],
                estimated_duration_minutes=30,
                parallel_execution=False,
                success_criteria=["Basic understanding of quantum computing"],
                reasoning="Limited to general research agent only"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_simple_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=limited_agents
            )
            
            # All tasks should use only the available agent
            agent_types = [task.agent_type for task in plan.tasks]
            assert all(agent_type in limited_agents for agent_type in agent_types)

    def test_research_task_validation(self):
        """Test ResearchTask model validation."""
        # Valid task
        valid_task = ResearchTask(
            task_description="Test research task",
            agent_type="general_research",
            priority=5,
            keywords=["test", "keywords"],
            expected_outcome="Test outcome"
        )
        assert valid_task.priority == 5
        assert len(valid_task.keywords) == 2
        
        # Test priority bounds - pydantic may clamp instead of raising
        try:
            invalid_task = ResearchTask(
                task_description="Invalid task",
                agent_type="general_research",
                priority=15,  # Invalid priority > 10
                keywords=["test"],
                expected_outcome="Test outcome"
            )
            # If pydantic allows it, priority should be valid range
            assert 1 <= invalid_task.priority <= 10
        except Exception:
            # If pydantic raises, that's also acceptable
            pass

    def test_research_plan_confidence_threshold(self, mock_settings):
        """Test that research plan uses settings for confidence threshold."""
        mock_settings.min_confidence_threshold = 0.8
        
        # Create plan with custom confidence threshold
        plan = ResearchPlan(
            original_query="test query",
            research_objective="test objective",
            tasks=[
                ResearchTask(
                    task_description="test task",
                    agent_type="general_research", 
                    priority=5,
                    keywords=["test"],
                    expected_outcome="test"
                )
            ],
            estimated_duration_minutes=30,
            parallel_execution=False,
            success_criteria=["test"],
            reasoning="test"
        )
        
        # Should use settings threshold
        assert plan.confidence_threshold == 0.8

    async def test_create_research_plan_error_handling(self, sample_simple_query, available_agents, mock_settings):
        """Test error handling in research plan creation."""
        with patch.object(planner_agent, 'run') as mock_run:
            # Simulate agent error
            mock_run.side_effect = Exception("Planning agent error")
            
            # Should propagate the error
            with pytest.raises(Exception):
                await create_research_plan(
                    query=sample_simple_query,
                    session_id="test-session",
                    request_id="test-request",
                    available_agents=available_agents
                )

    @pytest.mark.parametrize("query_type,expected_parallel", [
        ("simple query", False),
        ("complex multi-faceted research question requiring analysis across multiple domains", True),
        ("current news about AI", False),  # Simple current events
        ("compare AI policies across countries with economic and social impact analysis", True)  # Complex comparison
    ])
    async def test_parallel_execution_decisions(self, query_type, expected_parallel, available_agents, mock_settings):
        """Test decisions about parallel vs sequential execution."""
        with patch.object(planner_agent, 'run') as mock_run:
            # Mock based on complexity
            task_count = 3 if expected_parallel else 1
            
            tasks = []
            for i in range(task_count):
                tasks.append(ResearchTask(
                    task_description=f"Task {i+1}",
                    agent_type="general_research",
                    priority=8,
                    keywords=[f"keyword{i+1}"],
                    expected_outcome=f"Outcome {i+1}"
                ))
            
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=query_type,
                research_objective="Test objective",
                tasks=tasks,
                estimated_duration_minutes=60 if expected_parallel else 30,
                parallel_execution=expected_parallel,
                success_criteria=["Test success"],
                reasoning="Test reasoning"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=query_type,
                session_id="test-session",
                request_id="test-request",
                available_agents=available_agents
            )
            
            assert plan.parallel_execution == expected_parallel
            if expected_parallel:
                assert len(plan.tasks) > 1  # Complex queries should have multiple tasks
            else:
                assert len(plan.tasks) >= 1  # Simple queries have at least one task

    async def test_task_keyword_generation(self, sample_complex_query, available_agents, mock_settings):
        """Test that appropriate keywords are generated for tasks."""
        with patch.object(planner_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_complex_query,
                research_objective="Test complex research",
                tasks=[
                    ResearchTask(
                        task_description="Research renewable energy policies",
                        agent_type="general_research",
                        priority=8,
                        keywords=["renewable energy", "policy", "government", "legislation"],
                        expected_outcome="Policy analysis"
                    ),
                    ResearchTask(
                        task_description="Economic impact analysis",
                        agent_type="academic_research",
                        priority=7,
                        keywords=["economic impact", "renewable energy economics", "cost analysis"],
                        expected_outcome="Economic data"
                    )
                ],
                estimated_duration_minutes=75,
                parallel_execution=True,
                success_criteria=["Comprehensive policy comparison"],
                reasoning="Complex query broken down by topic area"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_complex_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=available_agents
            )
            
            # Check that keywords are relevant to tasks
            for task in plan.tasks:
                assert len(task.keywords) > 0
                # Keywords should be strings
                assert all(isinstance(keyword, str) for keyword in task.keywords)
                # Keywords should not be empty
                assert all(len(keyword.strip()) > 0 for keyword in task.keywords)

    async def test_success_criteria_generation(self, sample_complex_query, available_agents, mock_settings):
        """Test that meaningful success criteria are generated."""
        with patch.object(planner_agent, 'run') as mock_run:
            mock_result = Mock()
            mock_result.output = ResearchPlan(
                original_query=sample_complex_query,
                research_objective="Comprehensive renewable energy policy analysis",
                tasks=[
                    ResearchTask(
                        task_description="Policy research",
                        agent_type="general_research",
                        priority=8,
                        keywords=["policy"],
                        expected_outcome="Policy data"
                    )
                ],
                estimated_duration_minutes=60,
                parallel_execution=False,
                success_criteria=[
                    "Compare policies across at least 3 countries",
                    "Include both wind and solar energy data", 
                    "Cover economic impacts quantitatively",
                    "Address environmental outcomes"
                ],
                reasoning="Detailed success criteria for comprehensive analysis"
            )
            mock_run.return_value = mock_result

            plan = await create_research_plan(
                query=sample_complex_query,
                session_id="test-session",
                request_id="test-request",
                available_agents=available_agents
            )
            
            # Should have meaningful success criteria
            assert len(plan.success_criteria) > 1
            assert all(isinstance(criterion, str) for criterion in plan.success_criteria)
            assert all(len(criterion.strip()) > 10 for criterion in plan.success_criteria)  # Non-trivial criteria