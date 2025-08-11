from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from datetime import datetime, timezone

try:
    from ..clients import get_model
    from ..tools.brave_tools import search_web_tool, search_with_retry
    from ..config.settings import get_settings
    from .planner import AgentType
except ImportError:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from clients import get_model
    from tools.brave_tools import search_with_retry
    from config.settings import get_settings
    from agents.planner import AgentType


# ========== Models ==========

class ResearcherDeps(BaseModel):
    """Dependencies for research agents."""
    session_id: str = Field(description="Session identifier")
    request_id: str = Field(description="Request identifier")
    agent_type: AgentType = Field(description="Type of research agent")
    task_description: str = Field(description="Specific task description")
    keywords: List[str] = Field(description="Key search terms")
    max_sources: int = Field(description="Maximum number of sources to find", default_factory=lambda: get_settings().max_search_results)


class ResearchFinding(BaseModel):
    """A single research finding with its evidence."""
    claim: str = Field(description="Main claim or finding")
    evidence_urls: List[str] = Field(description="URLs of supporting sources")
    evidence_titles: List[str] = Field(description="Titles of supporting sources")
    evidence_snippets: List[str] = Field(description="Key excerpts from sources")
    confidence: float = Field(description="Confidence in this finding (0.0-1.0)")
    keywords: List[str] = Field(description="Key terms related to this finding")
    source_count: int = Field(description="Number of sources supporting this finding")


class ResearchOutput(BaseModel):
    """Output from a research agent."""
    findings: List[ResearchFinding] = Field(description="List of research findings")
    agent_type: AgentType = Field(description="Which agent produced this research")
    sources_searched: int = Field(description="Number of sources searched")
    search_queries_used: List[str] = Field(description="Actual search queries used")
    confidence_score: float = Field(description="Overall confidence in the research (0.0-1.0)")
    agent_notes: str = Field(description="Additional notes from the agent", default="")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ========== Agents ==========

general_research_agent = Agent(
    model=get_model(),
    deps_type=ResearcherDeps,
    output_type=ResearchOutput,
    system_prompt="""You are a general research specialist. You excel at finding comprehensive information 
about any topic using web search and general knowledge sources.

Your research process:
1. Analyze the task and identify key information needs
2. Use the web_search tool to find relevant sources
3. Evaluate source credibility and relevance
4. Extract key claims and supporting evidence
5. Synthesize findings with appropriate confidence levels

For each finding, provide:
- Clear, factual claims
- URLs, titles, and snippets from supporting sources
- Realistic confidence assessments (0.0-1.0)
- Relevant keywords

Focus on accuracy, credibility, and comprehensiveness. Be honest about limitations."""
)


academic_research_agent = Agent(
    model=get_model(),
    deps_type=ResearcherDeps, 
    output_type=ResearchOutput,
    system_prompt="""You are an academic research specialist. You focus on scholarly sources, 
research papers, institutional publications, and authoritative academic content.

Your research approach:
1. Use web_search to find academic and institutional sources
2. Prioritize peer-reviewed and institutional sources
3. Look for research papers, studies, and academic publications
4. Focus on evidence-based claims with proper methodology
5. Evaluate research quality and statistical significance

For academic findings:
- Emphasize peer-reviewed sources when available
- Include methodology context where relevant
- Higher confidence for well-established academic consensus
- Note limitations and areas of ongoing research

Maintain high standards for evidence quality and source credibility."""
)


news_research_agent = Agent(
    model=get_model(),
    deps_type=ResearcherDeps,
    output_type=ResearchOutput, 
    system_prompt="""You are a news and current events research specialist. You focus on recent 
developments, breaking news, trending topics, and timely information.

Your research focus:
1. Use web_search to find recent and current information
2. Look for reputable news sources and journalism
3. Track developing stories and emerging trends
4. Provide temporal context and timeline information
5. Balance multiple perspectives on current events

For news findings:
- Emphasize recency and timeliness
- Include multiple source perspectives when possible
- Note the evolving nature of breaking news
- Provide appropriate context for developing situations
- Be cautious about unverified claims in breaking news

Focus on credible journalism and established news sources."""
)


# ========== Tools ==========

@general_research_agent.tool
@academic_research_agent.tool  
@news_research_agent.tool
async def web_search(
    ctx: RunContext[ResearcherDeps],
    query: str,
    count: int = 5
) -> Dict[str, Any]:
    """Search the web for information related to the research task."""
    
    settings = get_settings()
    if not settings.brave_api_key:
        return {
            "error": "No Brave API key configured",
            "results": [],
            "query_used": query
        }
    
    try:
        # Determine search type based on agent
        search_type = "general"
        if ctx.deps.agent_type == AgentType.NEWS_RESEARCH:
            search_type = "news"
        elif ctx.deps.agent_type == AgentType.ACADEMIC_RESEARCH:
            search_type = "academic"
        
        results = await search_with_retry(
            api_key=settings.brave_api_key,
            query=query,
            count=count,
            search_type=search_type,
            max_retries=2
        )
        
        return {
            "results": results,
            "query_used": query,
            "search_type": search_type,
            "source_count": len(results)
        }
        
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "results": [],
            "query_used": query
        }


# ========== Functions ==========

async def conduct_research(
    task_description: str,
    agent_type: AgentType,
    session_id: str,
    request_id: str,
    keywords: List[str],
    max_sources: Optional[int] = None,
    quick_mode: bool = False
) -> ResearchOutput:
    """Conduct research using the specified agent type."""
    
    # Use settings default if not provided, but adjust for quick mode
    if max_sources is None:
        if quick_mode:
            max_sources = 3  # Limit to 3 sources in quick mode
        else:
            max_sources = get_settings().max_search_results
    
    deps = ResearcherDeps(
        session_id=session_id,
        request_id=request_id,
        agent_type=agent_type,
        task_description=task_description,
        keywords=keywords,
        max_sources=max_sources
    )
    
    # Select appropriate agent
    if agent_type == AgentType.ACADEMIC_RESEARCH:
        agent = academic_research_agent
    elif agent_type == AgentType.NEWS_RESEARCH:
        agent = news_research_agent
    else:
        agent = general_research_agent
    
    result = await agent.run(
        f"Research task: {task_description}\n\nKey terms: {', '.join(keywords)}",
        deps=deps
    )
    
    return result.output
