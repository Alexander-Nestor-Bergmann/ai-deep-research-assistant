from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from datetime import datetime, timezone

try:
    from ..clients import get_model
except ImportError:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from clients import get_model


# ========== Models ==========

class ConversationDeps(BaseModel):
    """Dependencies for conversation agent."""
    session_id: str = Field(description="Session identifier")
    request_id: str = Field(description="Request identifier")
    user_context: Optional[str] = Field(description="Any relevant user context", default=None)


class ConversationResponse(BaseModel):
    """Response from conversation agent."""
    response: str = Field(description="The conversational response")
    response_type: str = Field(description="Type of response: answer, clarification, greeting, etc.")
    confidence: float = Field(description="Confidence in the response (0.0-1.0)")
    suggested_follow_ups: list[str] = Field(description="Suggested follow-up questions", default_factory=list)
    handled_successfully: bool = Field(description="Whether the query was handled successfully")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ========== Agent ==========

conversation_agent = Agent(
    model=get_model(use_smaller_model=True),  # Use smaller model for conversation
    deps_type=ConversationDeps,
    output_type=ConversationResponse,
    system_prompt="""You are a helpful, knowledgeable conversation agent. Your role is to handle 
queries that don't require web research - these are typically:

- General knowledge questions with stable answers
- Math calculations and basic problem solving
- Definitions and explanations of well-established concepts
- Creative tasks like writing, brainstorming, jokes
- Personal questions, greetings, and casual conversation
- Questions about your own capabilities and limitations
- Simple how-to questions with well-known answers

For each query, provide:
1. A clear, accurate, and helpful response
2. Appropriate confidence level based on how well-established the information is
3. Suggested follow-up questions when relevant
4. A friendly, conversational tone

Guidelines:
- Be concise but comprehensive
- Acknowledge when you're uncertain about something
- For math problems, show your work when helpful
- For creative tasks, be engaging and helpful
- For greetings, be warm and welcoming
- If you don't know something that seems like it needs research, suggest that

Always be honest about your knowledge cutoff and limitations."""
)


# ========== Functions ==========

async def handle_conversation(
    query: str,
    session_id: str,
    request_id: str,
    user_context: Optional[str] = None,
    message_history: Optional[List[ModelMessage]] = None
) -> ConversationResponse:
    """Handle a conversational query that doesn't require research."""
    
    deps = ConversationDeps(
        session_id=session_id,
        request_id=request_id,
        user_context=user_context
    )
    
    result = await conversation_agent.run(query, deps=deps, message_history=message_history or [])
    return result.output
