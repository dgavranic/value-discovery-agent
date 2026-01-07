"""Node implementations for each stage of the value discovery journey."""

import json
import re

from langchain_core.messages import AIMessage

from meta_agent_v3.knowledge import (
    extract_knowledge,
    update_knowledge_map,
    get_knowledge_context,
    calculate_value_weights,
)
from meta_agent_v3.prompts import (
    SYSTEM_PROMPT,
    INTRODUCTION_MESSAGE,
    STAGE_1_META_PROMPT,
    STAGE_1_COMPLETION_PROMPT,
    STAGE_2_META_PROMPT,
    STAGE_2_COMPLETION_PROMPT,
    STAGE_3_META_PROMPT,
    STAGE_3_COMPLETION_PROMPT,
    STAGE_4_META_PROMPT,
    STAGE_4_COMPLETION_PROMPT,
    STAGE_5_META_PROMPT,
    format_message_analysis,
    format_conversation_history,
)
from meta_agent_v3.state import ActionSuggestion, MetaAgentState
from meta_agent_v3.utils import (
    get_llm,
    get_last_user_message,
    get_conversation_history,
    create_context_messages,
    update_stage_metrics,
    finalize_stage_metrics,
    estimate_tokens,
    count_user_messages_in_stage,
)


# === STAGE 0: INTRODUCTION ===

def introduction_node(state: MetaAgentState) -> dict:
    """Stage 0: Welcome user and explain the process.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    # Update metrics
    stage_name = "introduction"
    tokens = estimate_tokens(INTRODUCTION_MESSAGE)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=INTRODUCTION_MESSAGE)],
        "stage": 0,
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

    # Extract knowledge from user message
    extracted = extract_knowledge(user_message, llm)
    knowledge_update = update_knowledge_map(state, extracted)

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
        "stage": 1,
        **knowledge_update,
        **metrics_update
    }


def assess_rapport_completion(state: MetaAgentState) -> dict:
    """Assess if rapport building stage is complete using LLM.

    Args:
        state: Current agent state

    Returns:
        Dictionary with completion assessment and potential stage update
    """
    llm = get_llm(state.model_name, temperature=0.3)  # Lower temperature for assessment

    knowledge_context = get_knowledge_context(state)
    conversation_history = format_conversation_history(get_conversation_history(state, n_messages=10))

    assessment_prompt = STAGE_1_COMPLETION_PROMPT.format(
        knowledge_context=knowledge_context,
        conversation_history=conversation_history
    )

    # Get assessment
    response = llm.invoke([
        {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
        {"role": "user", "content": assessment_prompt}
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # If ready to advance and we have at least 3 exchanges, move forward
        turn_count = count_user_messages_in_stage(state, 1)

        if assessment.get("ready_to_advance") and turn_count >= 3:
            finalize_stage_metrics(state, "rapport_building")
            return {"intent_confirmed": True, "stage": 2}

        # Force advancement after 8 turns to prevent infinite loop
        if turn_count >= 8:
            finalize_stage_metrics(state, "rapport_building")
            return {"intent_confirmed": True, "stage": 2}

        return {}  # Stay in current stage

    except (json.JSONDecodeError, Exception):
        # If assessment fails, use fallback logic
        turn_count = count_user_messages_in_stage(state, 1)
        goals_count = len(state.user_profile.goals)

        # Move forward if we have goals and enough turns
        if goals_count >= 1 and turn_count >= 3:
            finalize_stage_metrics(state, "rapport_building")
            return {"intent_confirmed": True, "stage": 2}

        # Force after 8 turns
        if turn_count >= 8:
            finalize_stage_metrics(state, "rapport_building")
            return {"intent_confirmed": True, "stage": 2}

        return {}


# === STAGE 2: VALUE DISCOVERY ===

def value_discovery_node(state: MetaAgentState) -> dict:
    """Stage 2: Explore deeper values and motivations.

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

    # Extract knowledge
    extracted = extract_knowledge(user_message, llm)
    knowledge_update = update_knowledge_map(state, extracted)

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
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=6)

    # Generate response
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    # Update metrics
    tokens = estimate_tokens(user_message) + estimate_tokens(question) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=question)],
        "stage": 2,
        **knowledge_update,
        **weight_update,
        **metrics_update
    }


def assess_value_discovery_completion(state: MetaAgentState) -> dict:
    """Assess if value discovery stage is complete using LLM.

    Args:
        state: Current agent state

    Returns:
        Dictionary with completion assessment and potential stage update
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    value_count = len(state.user_profile.values)
    goals_count = len(state.user_profile.goals)

    assessment_prompt = STAGE_2_COMPLETION_PROMPT.format(
        knowledge_context=knowledge_context,
        value_count=value_count,
        goals_count=goals_count
    )

    response = llm.invoke([
        {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
        {"role": "user", "content": assessment_prompt}
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        turn_count = count_user_messages_in_stage(state, 2)

        # Ready to advance if LLM says so and we have minimum values
        if assessment.get("ready_to_advance") and value_count >= 3 and turn_count >= 3:
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}

        # Force advancement after 10 turns
        if turn_count >= 10:
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}

        return {}

    except (json.JSONDecodeError, Exception):
        # Fallback logic
        turn_count = count_user_messages_in_stage(state, 2)

        if value_count >= 4 and turn_count >= 4:
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}

        if turn_count >= 10:
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}

        return {}


# === STAGE 3: VALUE RANKING ===

def value_ranking_node(state: MetaAgentState) -> dict:
    """Stage 3: Help user prioritize and rank their values.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "value_ranking"

    # Get last user message
    user_message = get_last_user_message(state)

    # Determine ranking stage based on turn count
    turn_count = count_user_messages_in_stage(state, 3)
    if turn_count == 0:
        ranking_stage = "starting"
    elif turn_count < 3:
        ranking_stage = "mid-ranking"
    else:
        ranking_stage = "finalizing"

    # If we have user message, extract any preference signals
    if user_message:
        extracted = extract_knowledge(user_message, llm)
        knowledge_update = update_knowledge_map(state, extracted)
    else:
        knowledge_update = {}

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = STAGE_3_META_PROMPT.format(
        knowledge_context=knowledge_context,
        user_message=user_message or "[Starting value ranking]",
        ranking_stage=ranking_stage
    )

    # Create context messages
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=6)

    # Generate response
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    # Update metrics
    tokens = estimate_tokens(user_message or "") + estimate_tokens(question) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=question)],
        "stage": 3,
        **knowledge_update,
        **metrics_update
    }


def assess_value_ranking_completion(state: MetaAgentState) -> dict:
    """Assess if value ranking stage is complete using LLM.

    Args:
        state: Current agent state

    Returns:
        Dictionary with completion assessment and potential stage update
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)

    assessment_prompt = STAGE_3_COMPLETION_PROMPT.format(
        knowledge_context=knowledge_context
    )

    response = llm.invoke([
        {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
        {"role": "user", "content": assessment_prompt}
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        turn_count = count_user_messages_in_stage(state, 3)

        # Ready to advance if LLM confirms and enough turns
        if assessment.get("ready_to_advance") and turn_count >= 2:
            finalize_stage_metrics(state, "value_ranking")
            return {"values_confirmed": True, "stage": 4}

        # Force advancement after 6 turns
        if turn_count >= 6:
            finalize_stage_metrics(state, "value_ranking")
            return {"values_confirmed": True, "stage": 4}

        return {}

    except (json.JSONDecodeError, Exception):
        # Fallback
        turn_count = count_user_messages_in_stage(state, 3)

        if turn_count >= 3:
            finalize_stage_metrics(state, "value_ranking")
            return {"values_confirmed": True, "stage": 4}

        if turn_count >= 6:
            finalize_stage_metrics(state, "value_ranking")
            return {"values_confirmed": True, "stage": 4}

        return {}


# === STAGE 4: ACTION PLANNING ===

def action_planning_node(state: MetaAgentState) -> dict:
    """Stage 4: Create value-aligned action plan.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "action_planning"

    # Get last user message
    user_message = get_last_user_message(state)

    # Determine plan status
    turn_count = count_user_messages_in_stage(state, 4)
    actions_count = len(state.user_profile.suggested_actions)

    if turn_count == 0 or actions_count == 0:
        plan_status = "no actions yet - generate initial suggestions"
    elif user_message and ("prefer" in user_message.lower() or "like" in user_message.lower()):
        plan_status = "feedback received - refine and use A/B testing"
    else:
        plan_status = "actions presented - gather feedback"

    # Extract any feedback from user message
    if user_message:
        extracted = extract_knowledge(user_message, llm)
        # Store feedback in suggested actions if applicable
        if state.user_profile.suggested_actions:
            # Update the most recent action with user feedback
            state.user_profile.suggested_actions[-1].user_feedback = user_message[:200]

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = STAGE_4_META_PROMPT.format(
        knowledge_context=knowledge_context,
        user_message=user_message or "[Starting action planning]",
        plan_status=plan_status
    )

    # Create context messages
    context_messages = create_context_messages(state, SYSTEM_PROMPT, meta_prompt, n_history=8)

    # Generate response
    response = llm.invoke(context_messages)
    action_response = response.content if isinstance(response.content, str) else str(response.content)

    # Try to extract action suggestions from response
    # Look for numbered lists or clear action items
    action_items = re.findall(r'\d+\.\s+([^\n]+(?:\n(?!\d+\.)[^\n]+)*)', action_response)

    # Add new actions to suggested_actions
    for action_text in action_items:
        # Try to identify linked values in the action text
        linked_values = []
        for value in state.user_profile.values.keys():
            if value.lower() in action_text.lower():
                linked_values.append(value)

        # Only add if not duplicate
        if not any(a.description == action_text for a in state.user_profile.suggested_actions):
            action = ActionSuggestion(
                description=action_text,
                linked_values=linked_values,
                fit_score=0.8  # Default score
            )
            state.user_profile.suggested_actions.append(action)

    # Update metrics
    tokens = estimate_tokens(user_message or "") + estimate_tokens(action_response) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=action_response)],
        "stage": 4,
        "user_profile": state.user_profile,
        **metrics_update
    }


def assess_action_planning_completion(state: MetaAgentState) -> dict:
    """Assess if action planning stage is complete using LLM.

    Args:
        state: Current agent state

    Returns:
        Dictionary with completion assessment and potential stage update
    """
    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)

    # Format actions for assessment
    actions_text = "\n".join([
        f"- {action.description} (linked to: {', '.join(action.linked_values)})"
        for action in state.user_profile.suggested_actions
    ])

    assessment_prompt = STAGE_4_COMPLETION_PROMPT.format(
        knowledge_context=knowledge_context,
        actions=actions_text
    )

    response = llm.invoke([
        {"role": "system", "content": "You are an assessment system. Return only valid JSON."},
        {"role": "user", "content": assessment_prompt}
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        turn_count = count_user_messages_in_stage(state, 4)
        actions_count = len(state.user_profile.suggested_actions)

        # Ready to advance if plan is good and user confirmed
        if assessment.get("ready_to_advance") and actions_count >= 3 and turn_count >= 2:
            finalize_stage_metrics(state, "action_planning")
            return {"plan_generated": True, "stage": 5}

        # Force advancement after 7 turns
        if turn_count >= 7:
            finalize_stage_metrics(state, "action_planning")
            return {"plan_generated": True, "stage": 5}

        return {}

    except (json.JSONDecodeError, Exception):
        # Fallback
        turn_count = count_user_messages_in_stage(state, 4)
        actions_count = len(state.user_profile.suggested_actions)

        if actions_count >= 3 and turn_count >= 3:
            finalize_stage_metrics(state, "action_planning")
            return {"plan_generated": True, "stage": 5}

        if turn_count >= 7:
            finalize_stage_metrics(state, "action_planning")
            return {"plan_generated": True, "stage": 5}

        return {}


# === STAGE 5: SUMMARY & FEEDBACK ===

def summary_feedback_node(state: MetaAgentState) -> dict:
    """Stage 5: Provide comprehensive summary and collect feedback.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    llm = get_llm(state.model_name, state.temperature)
    stage_name = "summary_feedback"

    # Get last user message (if any - might be first in this stage)
    user_message = get_last_user_message(state)
    turn_count = count_user_messages_in_stage(state, 5)

    # If this is first turn in this stage, generate summary
    if turn_count == 0:
        knowledge_context = get_knowledge_context(state)

        meta_prompt = STAGE_5_META_PROMPT.format(
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
            "stage": 5,
            "final_summary": summary,
            **metrics_update
        }

    # If user has provided feedback, acknowledge and close
    elif user_message:
        closing_message = f"""Thank you so much for sharing this journey with me! 🌟

Your insights and reflections have been truly valuable. I hope this process has helped you gain clarity on what matters most to you and how to move forward in alignment with your values.

Remember: your values are your compass. When decisions feel difficult, return to what truly matters to you.

Wishing you all the best on your journey ahead! ✨"""

        # Store final feedback
        state.final_feedback = user_message

        # Finalize metrics
        finalize_stage_metrics(state, "summary_feedback")

        tokens = estimate_tokens(user_message) + estimate_tokens(closing_message)
        metrics_update = update_stage_metrics(state, stage_name, tokens)

        return {
            "messages": [AIMessage(content=closing_message)],
            "final_feedback": user_message,
            **metrics_update
        }

    # Shouldn't reach here, but just in case
    return {"stage": 5}

