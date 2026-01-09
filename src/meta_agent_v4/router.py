"""Routing logic with integrated validation for stage transitions."""

import json
import re
from typing import Literal

from meta_agent_v4.prompts import (
    STAGE_1_VALIDATION_PROMPT,
    STAGE_2_VALIDATION_PROMPT,
    STAGE_3_VALIDATION_PROMPT,
    STAGE_4_VALIDATION_PROMPT,
    format_conversation_history,
)
from meta_agent_v4.state import MetaAgentState
from meta_agent_v4.utils import (
    count_user_messages_in_stage,
    get_conversation_history,
    get_llm,
)
from meta_agent_v4.knowledge import get_knowledge_context


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


def validate_rapport_building(state: MetaAgentState) -> Literal["rapport_building", "value_discovery"]:
    """Validate if rapport building is complete.

    Uses LLM to assess completion criteria.

    Args:
        state: Current agent state

    Returns:
        Next stage (same to loop, or next to advance)
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    conversation_history = format_conversation_history(
        get_conversation_history(state, n_messages=12)
    )
    turn_count = count_user_messages_in_stage(state, 1)

    # Build validation prompt
    validation_prompt = STAGE_1_VALIDATION_PROMPT.format(
        knowledge_context=knowledge_context,
        conversation_history=conversation_history,
        turn_count=turn_count
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
            {"role": "user", "content": validation_prompt}
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "rapport_building")
            return "value_discovery"

        # Stay in rapport building
        return "rapport_building"

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic
        goals_count = len(state.user_profile.goals)

        # Force advancement after 8 turns or if we have goals and 5+ turns
        if turn_count >= 8 or (goals_count >= 1 and turn_count >= 5):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "rapport_building")
            return "value_discovery"

        return "rapport_building"


def validate_value_discovery(state: MetaAgentState) -> Literal["value_discovery", "action_planning"]:
    """Validate if value discovery is complete.

    Uses LLM to assess completion criteria.

    Args:
        state: Current agent state

    Returns:
        Next stage (same to loop, or next to advance)
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    turn_count = count_user_messages_in_stage(state, 2)
    value_count = len(state.user_profile.values)

    # Build validation prompt
    validation_prompt = STAGE_2_VALIDATION_PROMPT.format(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        value_count=value_count
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
            {"role": "user", "content": validation_prompt}
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "value_discovery")
            return "action_planning"

        return "value_discovery"

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic

        # Force advancement after 10 turns or if we have values and 6+ turns
        if turn_count >= 10 or (value_count >= 3 and turn_count >= 6):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "value_discovery")
            return "action_planning"

        return "value_discovery"


def validate_action_planning(state: MetaAgentState) -> Literal["action_planning", "summary_feedback"]:
    """Validate if action planning is complete.

    Uses LLM to assess completion criteria.

    Args:
        state: Current agent state

    Returns:
        Next stage (same to loop, or next to advance)
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    turn_count = count_user_messages_in_stage(state, 3)
    action_count = len(state.user_profile.suggested_actions)

    # Build validation prompt
    validation_prompt = STAGE_3_VALIDATION_PROMPT.format(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        action_count=action_count
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
            {"role": "user", "content": validation_prompt}
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "action_planning")
            return "summary_feedback"

        return "action_planning"

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic

        # Force advancement after 7 turns or if we have actions and 4+ turns with feedback
        if turn_count >= 7 or (action_count >= 3 and turn_count >= 4):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "action_planning")
            return "summary_feedback"

        return "action_planning"


def validate_summary_feedback(state: MetaAgentState) -> Literal["summary_feedback", "__end__"]:
    """Validate if summary & feedback stage is complete.

    Args:
        state: Current agent state

    Returns:
        Next stage (same to loop, or end)
    """
    llm = get_llm(state.model_name, temperature=0.3)

    turn_count = count_user_messages_in_stage(state, 4)
    has_feedback = bool(state.final_feedback)

    # Build validation prompt
    validation_prompt = STAGE_4_VALIDATION_PROMPT.format(
        turn_count=turn_count,
        has_feedback=has_feedback
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
            {"role": "user", "content": validation_prompt}
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should end
        if assessment.get("should_end"):
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "summary_feedback")
            return "__end__"

        return "summary_feedback"

    except (json.JSONDecodeError, Exception):
        # Fallback: end if we have feedback or 2+ turns
        if has_feedback or turn_count >= 2:
            from meta_agent_v4.utils import finalize_stage_metrics
            finalize_stage_metrics(state, "summary_feedback")
            return "__end__"

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

