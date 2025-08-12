from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from datetime import datetime, timezone

try:
    from ..clients import get_model
    from ..config.settings import get_settings
    from .researcher import ResearchOutput
    from .planner import ResearchPlan
except ImportError:
    import sys
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from clients import get_model
    from config.settings import get_settings
    from agents.researcher import ResearchOutput
    from agents.planner import ResearchPlan


# ========== Models ==========


class SynthesizerDeps(BaseModel):
    """Dependencies for the synthesizer agent."""

    research_results: List[ResearchOutput] = Field(
        description="Research results to synthesize"
    )
    research_plan: ResearchPlan = Field(description="Original research plan")
    session_id: str = Field(description="Session identifier")
    request_id: str = Field(description="Request identifier")


class SynthesisOutput(BaseModel):
    """Final synthesized response."""

    final_answer: str = Field(description="Comprehensive answer to the original query")
    key_findings: List[str] = Field(description="Main findings from the research")
    source_urls: List[str] = Field(description="URLs of all sources used")
    source_titles: List[str] = Field(description="Titles of all sources used")
    confidence_score: float = Field(
        description="Overall confidence in the synthesis (0.0-1.0)"
    )
    limitations: List[str] = Field(
        description="Known limitations or gaps in the research"
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")
    research_summary: str = Field(description="Brief summary of the research process")
    total_sources: int = Field(description="Total number of unique sources consulted")
    agents_used: List[str] = Field(
        description="Types of research agents that contributed"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def requires_high_confidence(cls) -> float:
        """Get the synthesis confidence threshold from settings."""
        return get_settings().synthesis_confidence_threshold


# ========== Agent ==========

synthesizer_agent = Agent(
    model=get_model(use_smaller_model=False),  # Use full model for complex synthesis
    deps_type=SynthesizerDeps,
    output_type=SynthesisOutput,
    system_prompt="""You are a research synthesis specialist. Your job is to take multiple research 
results from different agents and create a comprehensive, well-structured response.

Your synthesis process:
1. Analyze all research findings and identify key themes
2. Cross-reference claims across different sources
3. Resolve any conflicts between sources
4. Create a coherent narrative that addresses the original query
5. Assess overall confidence based on source quality and consistency
6. Identify limitations and suggest follow-up questions

For your synthesis:
- Prioritize claims supported by multiple sources
- Be transparent about conflicting information
- Include diverse perspectives when relevant
- Maintain appropriate confidence levels
- Cite sources naturally in your final answer
- Structure your response to be clear and comprehensive

The final_answer should be a complete, well-written response that directly addresses 
the user's original query. Include specific details and examples from your research.""",
)


def prepare_synthesis_prompt(deps: SynthesizerDeps) -> str:
    """Prepare the synthesis prompt with research data."""
    prompt = f"""Original Query: {deps.research_plan.original_query}

Research Objective: {deps.research_plan.research_objective}

Research Results to Synthesize:
"""

    for i, result in enumerate(deps.research_results, 1):
        prompt += f"\n--- {result.agent_type.title()} Research Results ---\n"
        prompt += f"Sources searched: {result.sources_searched}\n"
        prompt += f"Queries used: {', '.join(result.search_queries_used)}\n"
        prompt += f"Overall confidence: {result.confidence_score}\n\n"

        for j, finding in enumerate(result.findings, 1):
            prompt += f"Finding {j}: {finding.claim}\n"
            prompt += f"Confidence: {finding.confidence}\n"
            prompt += f"Sources ({finding.source_count}): {', '.join(finding.evidence_titles)}\n"
            if finding.evidence_snippets:
                prompt += f"Key evidence: {finding.evidence_snippets[0][:200]}...\n"
            prompt += "\n"

        if result.agent_notes:
            prompt += f"Agent notes: {result.agent_notes}\n"
        prompt += "\n"

    return prompt


# ========== Functions ==========


async def synthesize_research(
    research_results: List[ResearchOutput],
    research_plan: ResearchPlan,
    session_id: str,
    request_id: str,
    message_history: Optional[List[ModelMessage]] = None,
) -> SynthesisOutput:
    """Synthesize research results into a comprehensive response."""

    deps = SynthesizerDeps(
        research_results=research_results,
        research_plan=research_plan,
        session_id=session_id,
        request_id=request_id,
    )

    # Prepare the synthesis prompt with all research data
    synthesis_prompt = prepare_synthesis_prompt(deps)

    result = await synthesizer_agent.run(
        synthesis_prompt, deps=deps, message_history=message_history or []
    )
    return result.output
