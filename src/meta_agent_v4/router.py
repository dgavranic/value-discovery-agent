"""Routing logic for stage transitions."""

from typing import Literal

from meta_agent_v4.state import MetaAgentState


def route_from_preprocessor(state: MetaAgentState) -> Literal["introduction", "rapport_building", "value_discovery", "action_planning", "summary_feedback", "__end__"]:
    """Route from preprocessor to appropriate stage.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 0:
        return "introduction"
    elif state.stage == 1:
        return "rapport_building"
    elif state.stage == 2:
        return "value_discovery"
    elif state.stage == 3:
        return "action_planning"
    elif state.stage == 4:
        return "summary_feedback"
    else:
        return "__end__"


def route_after_introduction(state: MetaAgentState) -> Literal["rapport_building", "__end__"]:
    """Route after introduction.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 1:
        return "rapport_building"
    return "__end__"


def route_from_reply(state: MetaAgentState) -> Literal["validate_rapport", "validate_value_discovery", "validate_action_planning", "validate_summary", "__end__"]:
    """Route from reply node to appropriate validator based on current stage.

    Args:
        state: Current agent state

    Returns:
        Validator node name
    """
    if state.stage == 1:
        return "validate_rapport"
    elif state.stage == 2:
        return "validate_value_discovery"
    elif state.stage == 3:
        return "validate_action_planning"
    elif state.stage == 4:
        return "validate_summary"
    else:
        return "__end__"


def route_after_validate_rapport(state: MetaAgentState) -> Literal["rapport_building", "value_discovery"]:
    """Route after rapport validation based on stage.

    If stage was updated to 2, go to value_discovery. Otherwise stay in rapport_building.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 2:
        return "value_discovery"
    return "rapport_building"


def route_after_validate_value_discovery(state: MetaAgentState) -> Literal["value_discovery", "action_planning"]:
    """Route after value discovery validation based on stage.

    If stage was updated to 3, go to action_planning. Otherwise stay in value_discovery.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 3:
        return "action_planning"
    return "value_discovery"


def route_after_validate_action_planning(state: MetaAgentState) -> Literal["action_planning", "summary_feedback"]:
    """Route after action planning validation based on stage.

    If stage was updated to 4, go to summary_feedback. Otherwise stay in action_planning.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 4:
        return "summary_feedback"
    return "action_planning"


def route_after_validate_summary(state: MetaAgentState) -> Literal["summary_feedback", "finalize"]:
    """Route after summary validation based on stage.

    If stage was updated to 5, go to finalize. Otherwise stay in summary_feedback.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    if state.stage == 5:
        return "finalize"
    return "summary_feedback"



def route_after_introduction(state: MetaAgentState) -> Literal["rapport_building", "__end__"]:
    """Route after introduction.

    Args:
        state: Current agent state

    Returns:
        Next node
    """
    # Check if user responded
    from langchain_core.messages import HumanMessage

    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]

    if len(user_messages) > 0:
        return "rapport_building"

    return "__end__"


def route_from_reply(state: MetaAgentState) -> Literal["validate_rapport", "validate_value_discovery", "validate_action_planning", "validate_summary", "__end__"]:
    """Route from REPLY node to appropriate validation node based on current stage.

    Args:
        state: Current agent state

    Returns:
        Next node (appropriate validator for current stage)
    """
    if state.stage == 1:
        return "validate_rapport"
    elif state.stage == 2:
        return "validate_value_discovery"
    elif state.stage == 3:
        return "validate_action_planning"
    elif state.stage == 4:
        return "validate_summary"
    else:
        return "__end__"


