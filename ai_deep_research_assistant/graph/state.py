from typing import TypedDict, List, Optional
from datetime import datetime, timezone

try:
    from ..agents.planner import ResearchPlan
    from ..agents.researcher import ResearchOutput
    from ..agents.synthesizer import SynthesisOutput
except ImportError:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from agents.planner import ResearchPlan
    from agents.researcher import ResearchOutput
    from agents.synthesizer import SynthesisOutput


# ========== State ==========

class ResearchState(TypedDict):
    """State for the research workflow."""
    
    # Input
    query: str
    session_id: str
    request_id: str
    
    # Planning phase
    research_plan: Optional[ResearchPlan]
    
    # Research phase
    research_results: List[ResearchOutput]
    completed_tasks: List[str]
    
    # Synthesis phase
    final_synthesis: Optional[SynthesisOutput]
    
    # Workflow metadata
    current_step: str
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


def create_initial_state(query: str, session_id: str, request_id: str) -> ResearchState:
    """Create initial state for the research workflow."""
    return ResearchState(
        query=query,
        session_id=session_id,
        request_id=request_id,
        research_plan=None,
        research_results=[],
        completed_tasks=[],
        final_synthesis=None,
        current_step="planning",
        error_message=None,
        started_at=datetime.now(timezone.utc),
        completed_at=None
    )
