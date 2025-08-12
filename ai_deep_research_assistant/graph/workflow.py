import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic_ai.messages import ModelMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ai_deep_research_assistant.agents.guardrail import (
    classify_query,
    should_route_to_research,
)
from ai_deep_research_assistant.agents.planner import (
    create_research_plan,
    AgentType,
)
from ai_deep_research_assistant.agents.researcher import conduct_research
from ai_deep_research_assistant.agents.synthesizer import (
    synthesize_research,
    synthesizer_agent,
    SynthesizerDeps,
    prepare_synthesis_prompt,
)
from ai_deep_research_assistant.agents.conversation import (
    conversation_agent,
    ConversationDeps,
)
from ai_deep_research_assistant.config.settings import get_settings
from ai_deep_research_assistant.graph.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)


# ========== State ==========


class EnhancedResearchState(ResearchState):
    """Enhanced state with guardrail information."""

    classification: Optional[Dict[str, Any]]  # Guardrail classification results
    skip_research: bool  # Whether to skip research based on classification
    conversation_response: Optional[str]  # Direct conversational response
    quick_mode: bool  # Whether to use quick mode for faster research


def create_enhanced_initial_state(
    query: str, session_id: str, request_id: str, quick_mode: bool = False
) -> EnhancedResearchState:
    """Create initial state for the enhanced research workflow."""
    base_state = create_initial_state(query, session_id, request_id)
    return EnhancedResearchState(
        **base_state,
        classification=None,
        skip_research=False,
        conversation_response=None,
        quick_mode=quick_mode,
    )


# ========== Nodes ==========


async def guardrail_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Classify the query to determine if research is needed."""
    logger.info(f"Classifying query: {state['query'][:100]}...")

    try:
        classification = await classify_query(
            query=state["query"],
            session_id=state["session_id"],
            request_id=state["request_id"],
            conversation_context=None,
            message_history=state.get("pydantic_message_history", []),
        )

        should_research = should_route_to_research(classification)

        logger.info(
            f"Classification: {'RESEARCH' if should_research else 'CONVERSATION'} (confidence: {classification.confidence:.2f})"
        )

        # If we can answer immediately without research
        if not should_research and classification.can_answer_immediately:
            result_run = await conversation_agent.run(
                state["query"],
                deps=ConversationDeps(
                    session_id=state["session_id"],
                    request_id=state["request_id"],
                    user_context=None,
                ),
                message_history=state.get("pydantic_message_history", []),
            )
            conversation_result = result_run.output
            new_messages = result_run.new_messages()

            total_messages = state.get("pydantic_message_history", []) + new_messages

            return {
                "classification": classification.model_dump(),
                "skip_research": True,
                "conversation_response": conversation_result.response,
                "conversation_result": conversation_result,
                "current_step": "completed",
                "completed_at": datetime.now(timezone.utc),
                "pydantic_message_history": total_messages,
            }
        else:
            return {
                "classification": classification.model_dump(),
                "skip_research": False,
                "current_step": "planning",
            }

    except Exception as e:
        logger.error(f"Guardrail classification failed: {e}")
        return {
            "classification": {"error": str(e)},
            "skip_research": False,
            "current_step": "planning",
        }


# Removed - now using the proper conversation agent


async def planning_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Create a research plan using classification insights."""
    logger.info(f"Planning research for: {state['query'][:100]}...")

    try:
        # Use classification insights to inform planning
        available_agents = [
            AgentType.GENERAL_RESEARCH,
            AgentType.ACADEMIC_RESEARCH,
            AgentType.NEWS_RESEARCH,
        ]

        # Adjust available agents based on quick mode
        if state.get("quick_mode", False):
            # In quick mode, limit to general research only for speed
            available_agents = [AgentType.GENERAL_RESEARCH]
        elif state.get("classification"):
            # Use classification insights for normal mode
            classification = state["classification"]
            if classification.get("requires_current_info"):
                # Prioritize news research for current info
                available_agents = [
                    AgentType.NEWS_RESEARCH,
                    AgentType.GENERAL_RESEARCH,
                    AgentType.ACADEMIC_RESEARCH,
                ]
            elif classification.get("suggested_research_type") == "academic":
                # Prioritize academic research
                available_agents = [
                    AgentType.ACADEMIC_RESEARCH,
                    AgentType.GENERAL_RESEARCH,
                    AgentType.NEWS_RESEARCH,
                ]

        research_plan = await create_research_plan(
            query=state["query"],
            session_id=state["session_id"],
            request_id=state["request_id"],
            available_agents=available_agents,
            quick_mode=state.get("quick_mode", False),
        )

        logger.info(f"Created plan with {len(research_plan.tasks)} tasks")

        return {"research_plan": research_plan, "error_message": None}

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"error_message": f"Planning failed: {str(e)}", "current_step": "error"}


async def research_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Execute research tasks using classification insights."""
    logger.info("Starting research phase")

    if not state["research_plan"]:
        return {"error_message": "No research plan available", "current_step": "error"}

    try:
        research_results = []
        completed_tasks = []

        # Adjust research approach based on classification
        settings = get_settings()
        max_sources = settings.max_search_results  # Use settings default
        if state.get("classification"):
            classification = state["classification"]
            if classification.get("complexity_estimate") == "complex":
                max_sources = min(
                    settings.max_search_results + 3, 20
                )  # Increase but cap at 20
            elif classification.get("complexity_estimate") == "simple":
                max_sources = max(
                    settings.max_search_results - 2, 3
                )  # Decrease but minimum 3

        tasks = state["research_plan"].tasks

        if state["research_plan"].parallel_execution and len(tasks) > 1:
            # Execute tasks in parallel
            logger.info(f"Executing {len(tasks)} research tasks in parallel")

            research_coroutines = [
                conduct_research(
                    task_description=task.task_description,
                    agent_type=task.agent_type,
                    session_id=state["session_id"],
                    request_id=state["request_id"],
                    keywords=task.keywords,
                    max_sources=max_sources,
                    quick_mode=state.get("quick_mode", False),
                )
                for task in tasks
            ]

            results = await asyncio.gather(*research_coroutines, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Research task {i} failed: {result}")
                    continue

                research_results.append(result)
                completed_tasks.append(tasks[i].task_description)

        else:
            # Execute tasks sequentially
            logger.info(f"Executing {len(tasks)} research tasks sequentially")

            for task in tasks:
                try:
                    result = await conduct_research(
                        task_description=task.task_description,
                        agent_type=task.agent_type,
                        session_id=state["session_id"],
                        request_id=state["request_id"],
                        keywords=task.keywords,
                        max_sources=max_sources,
                        quick_mode=state.get("quick_mode", False),
                    )

                    research_results.append(result)
                    completed_tasks.append(task.task_description)

                except Exception as e:
                    logger.error(f"Research task failed: {e}")
                    continue

        if not research_results:
            return {
                "error_message": "All research tasks failed",
                "current_step": "error",
            }

        logger.info(
            f"Research completed: {len(research_results)} results from {len(completed_tasks)} tasks"
        )

        return {
            "research_results": research_results,
            "completed_tasks": completed_tasks,
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"Research phase failed: {e}")
        return {"error_message": f"Research failed: {str(e)}", "current_step": "error"}


async def synthesis_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Synthesize research results."""
    logger.info("Starting synthesis phase")

    if not state["research_results"] or not state["research_plan"]:
        return {
            "error_message": "Missing research results or plan for synthesis",
            "current_step": "error",
        }

    try:
        # Run synthesis agent and capture messages
        deps = SynthesizerDeps(
            research_results=state["research_results"],
            research_plan=state["research_plan"],
            session_id=state["session_id"],
            request_id=state["request_id"],
        )

        synthesis_prompt = prepare_synthesis_prompt(deps)
        result_run = await synthesizer_agent.run(
            synthesis_prompt,
            deps=deps,
            message_history=state.get("pydantic_message_history", []),
        )

        synthesis = result_run.output
        new_messages = result_run.new_messages()

        logger.info(
            f"Synthesis completed: {len(synthesis.final_answer)} chars, confidence {synthesis.confidence_score:.2f}"
        )

        # Accumulate message history
        msg_history = state.get("pydantic_message_history", [])
        total_messages = msg_history + new_messages

        return {
            "final_synthesis": synthesis,
            "completed_at": datetime.now(timezone.utc),
            "error_message": None,
            "pydantic_message_history": total_messages,
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {"error_message": f"Synthesis failed: {str(e)}", "current_step": "error"}


async def conversation_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Handle conversational queries without research."""
    logger.info("Providing conversational response")

    conversation_result = state.get("conversation_result")
    new_messages = None

    if not conversation_result:
        result_run = await conversation_agent.run(
            state["query"],
            deps=ConversationDeps(
                session_id=state["session_id"],
                request_id=state["request_id"],
                user_context=None,
            ),
            message_history=state.get("pydantic_message_history", []),
        )
        conversation_result = result_run.output
        new_messages = result_run.new_messages()

    conversation_synthesis = {
        "final_answer": conversation_result.response,
        "key_findings": [],
        "source_urls": [],
        "source_titles": [],
        "confidence_score": conversation_result.confidence,
        "limitations": [],
        "follow_up_questions": conversation_result.suggested_follow_ups,
        "research_summary": "Conversational response - no research conducted",
        "total_sources": 0,
        "agents_used": ["conversational"],
        "created_at": datetime.now(timezone.utc),
    }

    result = {
        "final_synthesis": conversation_synthesis,
        "completed_at": datetime.now(timezone.utc),
    }

    if new_messages:
        msg_history = state.get("pydantic_message_history", [])
        total_messages = msg_history + new_messages
        result["pydantic_message_history"] = total_messages

    return result


async def error_node(state: EnhancedResearchState) -> Dict[str, Any]:
    """Handle errors and provide fallback response."""
    logger.error(f"Workflow error: {state.get('error_message', 'Unknown error')}")

    fallback_synthesis = {
        "final_answer": f"I apologize, but I encountered an error while processing your query: '{state['query']}'. Error: {state.get('error_message', 'Unknown error')}",
        "key_findings": [],
        "source_urls": [],
        "source_titles": [],
        "confidence_score": 0.0,
        "limitations": ["Workflow encountered an error"],
        "follow_up_questions": [],
        "research_summary": "Process was interrupted by an error",
        "total_sources": 0,
        "agents_used": [],
        "created_at": datetime.now(timezone.utc),
    }

    return {
        "final_synthesis": fallback_synthesis,
        "current_step": "completed",
        "completed_at": datetime.now(timezone.utc),
    }


# ========== Routing ==========


def route_after_guardrail(state: EnhancedResearchState) -> str:
    """Route after guardrail classification."""
    if state.get("skip_research"):
        return "conversation"
    elif state.get("current_step") == "error":
        return "error"
    else:
        return "planning"


def route_from_planning(state: EnhancedResearchState) -> str:
    """Route from planning step."""
    if state.get("error_message"):
        return "error"
    else:
        return "research"


def route_from_research(state: EnhancedResearchState) -> str:
    """Route from research step."""
    if state.get("error_message"):
        return "error"
    elif state.get("research_results"):
        return "synthesis"
    else:
        return "error"


def route_from_synthesis(state: EnhancedResearchState) -> str:
    """Route from synthesis step."""
    return "end"


# ========== Graph ==========


def create_research_graph() -> StateGraph:
    """Create the research workflow graph with guardrails."""

    workflow = StateGraph(EnhancedResearchState)

    # Add nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("research", research_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("error", error_node)

    # Add edges
    workflow.add_edge(START, "guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"conversation": "conversation", "planning": "planning", "error": "error"},
    )
    workflow.add_edge("conversation", END)
    workflow.add_conditional_edges(
        "planning", route_from_planning, {"research": "research", "error": "error"}
    )
    workflow.add_conditional_edges(
        "research", route_from_research, {"synthesis": "synthesis", "error": "error"}
    )
    workflow.add_conditional_edges("synthesis", route_from_synthesis, {"end": END})
    workflow.add_edge("error", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========== Functions ==========


default_graph = create_research_graph()


async def run_research(
    query: str,
    session_id: str,
    request_id: str,
    quick_mode: bool = False,
    graph: Optional[StateGraph] = None,
    session_thread_id: Optional[str] = None,
) -> EnhancedResearchState:
    """Run the research workflow with guardrails."""

    # Use provided graph or default
    if graph is None:
        graph = default_graph

    # Use session_thread_id for continuity, fall back to request_id for backward compatibility
    thread_id = session_thread_id if session_thread_id else request_id
    config = {"configurable": {"thread_id": thread_id}}

    existing_states = []
    try:
        # Try to get existing state from checkpointer
        checkpointer = graph.checkpointer
        if checkpointer and session_thread_id:
            existing_states = list(checkpointer.list(config, limit=1))
    except Exception as e:
        logger.warning(f"Error checking existing state: {e}")

    # Create initial state
    initial_state = create_enhanced_initial_state(
        query, session_id, request_id, quick_mode
    )

    # If session continuity is enabled and we found existing state, load the message history
    if session_thread_id and existing_states:
        try:
            latest_checkpoint = existing_states[0]
            existing_state = latest_checkpoint.checkpoint.get("channel_values", {})
            existing_msg_history = existing_state.get("pydantic_message_history", [])
            if existing_msg_history:
                initial_state["pydantic_message_history"] = existing_msg_history
        except Exception as e:
            logger.warning(f"Error loading existing message history: {e}")

    # Execute the workflow
    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state
