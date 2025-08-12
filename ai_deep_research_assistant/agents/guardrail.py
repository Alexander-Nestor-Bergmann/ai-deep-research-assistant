from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ai_deep_research_assistant.clients import get_model
from ai_deep_research_assistant.config.settings import get_settings

# ========== Models ==========


class GuardrailDeps(BaseModel):
    """Dependencies for guardrail agent."""

    session_id: str = Field(description="Session identifier")
    request_id: str = Field(description="Request identifier")
    conversation_context: Optional[str] = Field(
        description="Previous conversation context", default=None
    )


class GuardrailOutput(BaseModel):
    """Output from guardrail classification."""

    is_research_request: bool = Field(
        description="Whether this requires research or can be answered conversationally"
    )
    confidence: float = Field(description="Confidence in classification (0.0-1.0)")
    reasoning: str = Field(description="Explanation of the classification decision")
    complexity_estimate: str = Field(
        description="Estimated complexity: simple, moderate, complex"
    )
    suggested_research_type: Optional[str] = Field(
        description="Suggested research approach if applicable", default=None
    )
    estimated_sources_needed: int = Field(
        description="Estimated number of sources needed", default=1
    )
    can_answer_immediately: bool = Field(
        description="Whether this can be answered without web search"
    )
    requires_current_info: bool = Field(
        description="Whether current/recent information is needed"
    )


# ========== Agent ==========

guardrail_agent = Agent(
    model=get_model(use_smaller_model=True),  # Fast model for quick routing
    deps_type=GuardrailDeps,
    output_type=GuardrailOutput,
    system_prompt="""You are a query classification specialist. Your job is to quickly determine 
whether user queries require web research or can be answered conversationally.

Classification criteria:

REQUIRES RESEARCH (is_research_request = true):
- Questions about current events, recent developments, latest information
- Requests for specific data, statistics, recent studies
- Questions about topics you may not have complete knowledge of
- Comparative analysis requiring multiple sources
- Technical topics requiring up-to-date information
- "What are the latest..." "Recent developments in..." "Current status of..."

CONVERSATIONAL (is_research_request = false):
- General knowledge questions with stable answers
- Math calculations, definitions, explanations of concepts
- Personal questions, greetings, casual conversation
- Questions about your capabilities or system functionality
- Simple how-to questions with well-established answers
- Creative tasks like writing, jokes, stories

For each query, consider:
1. Does this need current/recent information?
2. Could the answer change based on new developments?
3. Do I have sufficient knowledge to answer accurately?
4. Would web search significantly improve the answer?

Be conservative - if uncertain, classify as research to ensure comprehensive answers.

Provide clear reasoning for your classification decision.""",
)


# ========== Functions ==========


async def classify_query(
    query: str,
    session_id: str,
    request_id: str,
    conversation_context: Optional[str] = None,
    message_history: Optional[List[ModelMessage]] = None,
) -> GuardrailOutput:
    """Classify a query as requiring research or conversational response."""

    deps = GuardrailDeps(
        session_id=session_id,
        request_id=request_id,
        conversation_context=conversation_context,
    )

    # Prepare input with context if available
    input_text = query
    if conversation_context:
        input_text = f"Context: {conversation_context}\n\nQuery: {query}"

    result = await guardrail_agent.run(
        input_text, deps=deps, message_history=message_history or []
    )
    return result.output


def should_route_to_research(
    classification: GuardrailOutput, confidence_threshold: Optional[float] = None
) -> bool:
    """Determine if query should be routed to research based on classification."""
    if not classification.is_research_request:
        return False

    # Use settings-based threshold if not provided
    if confidence_threshold is None:
        settings = get_settings()
        confidence_threshold = settings.min_confidence_threshold

    # Route to research if confidence is high enough or complexity suggests it
    return (
        classification.confidence >= confidence_threshold
        or classification.complexity_estimate in ["moderate", "complex"]
    )
