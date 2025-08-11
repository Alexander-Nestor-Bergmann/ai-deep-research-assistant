from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from datetime import datetime, timezone

try:
    from ..clients import get_model
    from ..config.settings import get_settings
except ImportError:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from clients import get_model
    from config.settings import get_settings


# ========== Enums ==========

class AgentType(str, Enum):
    """Available research agent types."""
    GENERAL_RESEARCH = "general_research"
    ACADEMIC_RESEARCH = "academic_research"
    NEWS_RESEARCH = "news_research"


# ========== Models ==========

class PlannerDeps(BaseModel):
    """Dependencies for the planner agent."""
    available_agents: List[AgentType] = Field(
        description="List of available research agents and their capabilities",
        default=[AgentType.GENERAL_RESEARCH, AgentType.ACADEMIC_RESEARCH, AgentType.NEWS_RESEARCH]
    )
    session_id: str = Field(description="Session identifier")
    request_id: str = Field(description="Request identifier")


class ResearchTask(BaseModel):
    """A single research task to be executed."""
    task_description: str = Field(description="Clear description of what needs to be researched")
    agent_type: AgentType = Field(description="Which agent should handle this task")
    priority: int = Field(description="Priority level 1-10 (10 being highest)", ge=1, le=10)
    keywords: List[str] = Field(description="Key search terms for this task")
    expected_outcome: str = Field(description="What type of information this task should produce")


class ResearchPlan(BaseModel):
    """Complete research plan with parallel tasks."""
    original_query: str = Field(description="The original user query")
    research_objective: str = Field(description="Main research objective")
    tasks: List[ResearchTask] = Field(description="List of research tasks to execute")
    estimated_duration_minutes: int = Field(description="Estimated time to complete all tasks")
    parallel_execution: bool = Field(description="Whether tasks can be executed in parallel")
    confidence_threshold: float = Field(description="Minimum confidence threshold for results", default_factory=lambda: get_settings().min_confidence_threshold)
    success_criteria: List[str] = Field(description="What constitutes successful completion")
    reasoning: str = Field(description="Explanation of the planning approach")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ========== Agent ==========

planner_agent = Agent(
    model=get_model(use_smaller_model=False),
    deps_type=PlannerDeps,
    output_type=ResearchPlan,
    system_prompt="""You are a research planning specialist. Your job is to analyze user queries and create 
comprehensive research plans that can be executed by specialized research agents.

For each query, you should:
1. Understand the core research objective
2. Break it down into specific, actionable research tasks
3. Assign appropriate research agents to each task
4. Determine optimal execution strategy (parallel vs sequential)
5. Estimate realistic timeframes
6. Define success criteria

Available agents:
- general_research: Web search, general information gathering
- academic_research: Scholarly articles, research papers, academic sources  
- news_research: Recent news, current events, trending topics

Create plans that are:
- Specific and actionable
- Well-distributed across appropriate agent types
- Realistic in scope and timing
- Focused on the user's actual information needs

For simple queries, create 1-2 focused tasks. For complex queries, break into 3-5 parallel tasks maximum."""
)


@planner_agent.system_prompt
def add_available_agents(ctx) -> str:
    """Add available agents to the system prompt dynamically."""
    return f"\n\nCurrently available agents: {', '.join(ctx.deps.available_agents)}"


# ========== Functions ==========

async def create_research_plan(
    query: str,
    session_id: str, 
    request_id: str,
    available_agents: Optional[List[AgentType]] = None,
    quick_mode: bool = False
) -> ResearchPlan:
    """Create a research plan for the given query."""
    
    deps = PlannerDeps(
        available_agents=available_agents or [AgentType.GENERAL_RESEARCH, AgentType.ACADEMIC_RESEARCH, AgentType.NEWS_RESEARCH],
        session_id=session_id,
        request_id=request_id
    )
    
    # Adjust query if in quick mode
    if quick_mode:
        adjusted_query = f"[QUICK MODE: Create exactly 1 focused task maximum for speed] {query}"
    else:
        adjusted_query = query
    
    result = await planner_agent.run(adjusted_query, deps=deps)
    return result.output
