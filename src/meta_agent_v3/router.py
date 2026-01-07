"""Routing logic for stage transitions."""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from meta_agent_v3.state import MetaAgentState


def route_after_introduction(state: MetaAgentState) -> Literal["rapport_building"] | None:
    """Route after introduction - always go to rapport building.

    Args:
        state: Current agent state

    Returns:
        Next stage name
    """
    # if user sent a message, move to rapport building
    if len(state.messages) > 2 and isinstance(state.messages[2], HumanMessage):
        return "rapport_building"
    return END


def route_after_rapport(state: MetaAgentState) -> Literal["rapport_building", "value_discovery"]:
    """Route after rapport building - check if intent is confirmed.

    Args:
        state: Current agent state

    Returns:
        Next stage name
    """
    if state.intent_confirmed:
        return "value_discovery"
    return "rapport_building"


def route_after_value_discovery(state: MetaAgentState) -> Literal["value_discovery", "value_ranking"]:
    """Route after value discovery - check if we have enough values.

    Args:
        state: Current agent state

    Returns:
        Next stage name
    """
    # If stage has advanced to 3, move to value ranking
    if state.stage >= 3:
        return "value_ranking"
    return "value_discovery"


def route_after_value_ranking(state: MetaAgentState) -> Literal["value_ranking", "action_planning"]:
    """Route after value ranking - check if values are confirmed.

    Args:
        state: Current agent state

    Returns:
        Next stage name
    """
    if state.values_confirmed or state.stage >= 4:
        return "action_planning"
    return "value_ranking"


def route_after_action_planning(state: MetaAgentState) -> Literal["action_planning", "summary_feedback"]:
    """Route after action planning - check if plan is generated.

    Args:
        state: Current agent state

    Returns:
        Next stage name
    """
    if state.plan_generated or state.stage >= 5:
        return "summary_feedback"
    return "action_planning"


def route_after_summary(state: MetaAgentState) -> Literal["summary_feedback", "__end__"]:
    """Route after summary - check if we've collected feedback.

    Args:
        state: Current agent state

    Returns:
        Next stage name or end
    """
    # Check if this is the second turn in summary stage (user has responded)
    from meta_agent_v3.utils import count_user_messages_in_stage

    turn_count = count_user_messages_in_stage(state, 5)

    # After user provides feedback (turn 1+), end the conversation
    if turn_count >= 1 and state.final_feedback:
        return "__end__"

    # Stay in summary to collect feedback
    return "summary_feedback"


def should_assess_completion(state: MetaAgentState) -> bool:
    """Determine if we should assess stage completion.

    Args:
        state: Current agent state

    Returns:
        True if we should assess completion
    """
    from meta_agent_v3.utils import count_user_messages_in_stage

    # Don't assess on introduction or summary stages
    if state.stage in [0, 5]:
        return False

    # Assess after every user message in stages 1-4
    turn_count = count_user_messages_in_stage(state, state.stage)
    return turn_count > 0

