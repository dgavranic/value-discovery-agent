"""Node implementations for each stage with integrated validation."""

import json
import re

from langchain_core.messages import AIMessage

from meta_agent_v4.knowledge import (
    extract_knowledge,
    update_knowledge_map,
    get_knowledge_context,
    calculate_value_weights,
)
from meta_agent_v4.prompts import (
    SYSTEM_PROMPT,
    INTRODUCTION_MESSAGE,
    STAGE_1_META_PROMPT,
    STAGE_2_META_PROMPT,
    STAGE_3_META_PROMPT,
    STAGE_4_META_PROMPT,
    format_message_analysis,
)
from meta_agent_v4.state import ActionSuggestion, MetaAgentState
from meta_agent_v4.utils import (
    get_llm,
    get_last_user_message,
    create_context_messages,
    update_stage_metrics,
    finalize_stage_metrics,
    estimate_tokens,
)


def preprocessor_node(state: MetaAgentState) -> dict:
    """Process incoming user message and extract knowledge.

    Runs before every stage to maintain knowledge map.

    Args:
        state: Current agent state

    Returns:
        Dictionary with knowledge updates
    """
    # Get last user message
    user_message = get_last_user_message(state)

    if not user_message:
        return {}

    # Extract knowledge
    llm = get_llm(state.model_name, state.temperature)
    extracted = extract_knowledge(user_message, llm)

    # Update knowledge map
    knowledge_update = update_knowledge_map(state, extracted)

    return knowledge_update


# === STAGE 0: INTRODUCTION ===

def introduction_node(state: MetaAgentState) -> dict:
    """Stage 0: Welcome user and explain the process.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    stage_name = "introduction"
    tokens = estimate_tokens(INTRODUCTION_MESSAGE)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    # Mark introduction as complete, move to rapport building
    finalize_stage_metrics(state, stage_name)

    return {
        "messages": [AIMessage(content=INTRODUCTION_MESSAGE)],
        "stage": 1,
        **metrics_update
    }


# === STAGE 1: RAPPORT BUILDING ===

def rapport_building_node(state: MetaAgentState) -> dict:
    """Stage 1: Build rapport and understand the specific problem/goal.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "rapport_building"

    # Get last user message
    user_message = get_last_user_message(state)
    if not user_message:
        return {
            "messages": [AIMessage(content="I didn't receive your response. Could you share what brought you here today?")]
        }

    # Extract knowledge (already done in preprocessor, but get for prompt)
    extracted = extract_knowledge(user_message, llm)

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = STAGE_1_META_PROMPT.format(
        knowledge_context=knowledge_context,
        user_message=user_message,
        message_analysis=format_message_analysis(extracted)
    )

    # Create context messages
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=6)

    # Generate response
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    # Update metrics
    tokens = estimate_tokens(user_message) + estimate_tokens(question) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=question)],
        **metrics_update
    }


# === STAGE 2: VALUE DISCOVERY ===

def value_discovery_node(state: MetaAgentState) -> dict:
    """Stage 2: Explore deeper values, motivations, and prioritization.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "value_discovery"

    # Get last user message
    user_message = get_last_user_message(state)
    if not user_message:
        return {
            "messages": [AIMessage(content="I'd love to hear more about why this matters to you.")]
        }

    # Recalculate value weights
    weight_update = calculate_value_weights(state)

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)
    value_count = len(state.user_profile.values)

    # Build meta-prompt
    meta_prompt = STAGE_2_META_PROMPT.format(
        knowledge_context=knowledge_context,
        user_message=user_message,
        value_count=value_count
    )

    # Create context messages
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=8)

    # Generate response
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    # Update metrics
    tokens = estimate_tokens(user_message) + estimate_tokens(question) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=question)],
        **weight_update,
        **metrics_update
    }


# === STAGE 3: ACTION PLANNING ===

def action_planning_node(state: MetaAgentState) -> dict:
    """Stage 3: Create value-aligned action plan with A/B testing.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "action_planning"

    # Get last user message
    user_message = get_last_user_message(state)

    # Determine plan status based on what we have
    actions_count = len(state.user_profile.suggested_actions)

    if actions_count == 0:
        plan_status = "no actions yet - generate initial suggestions with A/B options"
    elif user_message and any(word in user_message.lower() for word in ["prefer", "like", "choose", "option", "better"]):
        plan_status = "feedback received - refine chosen approach and build out details"
    elif actions_count < 3:
        plan_status = "actions presented - gather preferences and feedback"
    else:
        plan_status = "plan developed - confirm feasibility and alignment"

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = STAGE_3_META_PROMPT.format(
        knowledge_context=knowledge_context,
        user_message=user_message or "[Starting action planning]",
        plan_status=plan_status,
        action_count=actions_count
    )

    # Create context messages
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=8)

    # Generate response
    response = llm.invoke(context_messages)
    action_response = response.content if isinstance(response.content, str) else str(response.content)

    # Try to extract action suggestions from response
    action_items = re.findall(r'(?:\d+\.|\-|\*)\s+([^\n]+(?:\n(?!(?:\d+\.|\-|\*))[^\n]+)*)', action_response)

    # Add new actions to suggested_actions (avoid duplicates)
    for action_text in action_items:
        action_text = action_text.strip()
        if len(action_text) < 20 or len(action_text) > 300:
            continue

        # Try to identify linked values
        linked_values = []
        for value in state.user_profile.values.keys():
            if value.lower() in action_text.lower():
                linked_values.append(value)

        # Check if not duplicate
        if not any(a.description == action_text for a in state.user_profile.suggested_actions):
            action = ActionSuggestion(
                description=action_text,
                linked_values=linked_values,
                fit_score=0.8
            )
            state.user_profile.suggested_actions.append(action)

    # Update metrics
    tokens = estimate_tokens(user_message or "") + estimate_tokens(action_response) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=action_response)],
        "user_profile": state.user_profile,
        **metrics_update
    }


# === STAGE 4: SUMMARY & FEEDBACK ===

def summary_feedback_node(state: MetaAgentState) -> dict:
    """Stage 4: Provide comprehensive summary and collect feedback.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "summary_feedback"

    # Get last user message
    user_message = get_last_user_message(state)

    # If we already have final_summary and user just responded, this is final feedback
    if state.final_summary and user_message:
        closing_message = """Thank you so much for sharing this journey with me! 🌟

Your insights and reflections have been truly valuable. I hope this process has helped you gain clarity on what matters most to you and how to move forward in alignment with your values.

Remember: your values are your compass. When decisions feel difficult, return to what truly matters to you.

Wishing you all the best on your journey ahead! ✨"""

        # Finalize metrics
        finalize_stage_metrics(state, stage_name)

        tokens = estimate_tokens(user_message) + estimate_tokens(closing_message)
        metrics_update = update_stage_metrics(state, stage_name, tokens)

        return {
            "messages": [AIMessage(content=closing_message)],
            "final_feedback": user_message,
            "stage": 5,  # End stage
            **metrics_update
        }

    # Otherwise, generate the summary
    knowledge_context = get_knowledge_context(state)

    meta_prompt = STAGE_4_META_PROMPT.format(
        knowledge_context=knowledge_context
    )

    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=4)

    response = llm.invoke(context_messages)
    summary = response.content if isinstance(response.content, str) else str(response.content)

    # Update metrics
    tokens = estimate_tokens(summary) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=summary)],
        "final_summary": summary,
        **metrics_update
    }

