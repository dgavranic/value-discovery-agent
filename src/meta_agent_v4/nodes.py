"""Node implementations for each stage with integrated validation."""

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from meta_agent_v4.knowledge import (
    extract_knowledge,
    update_knowledge_map,
    get_knowledge_context,
    calculate_value_weights,
)
from meta_agent_v4.prompts import (
    render_system_prompt,
    render_introduction,
    render_stage1_meta,
    render_stage2_meta,
    render_stage3_meta,
    render_stage4_meta,
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

    if not user_message or state.stage == 0:
        return {}

    # Extract knowledge
    llm = get_llm(state.model_name, state.temperature)
    extracted = extract_knowledge(user_message, llm)

    # Update knowledge map
    knowledge_update = update_knowledge_map(state, extracted)

    return {"last_extracted_knowledge": extracted, **knowledge_update}


# === STAGE 0: INTRODUCTION ===


def introduction_node(state: MetaAgentState) -> dict:
    """Stage 0: Welcome user and explain the process.

    Args:
        state: Current agent state

    Returns:
        Dictionary with state updates
    """
    stage_name = "introduction"
    introduction_message = render_introduction()
    tokens = estimate_tokens(introduction_message)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    # Mark introduction as complete, move to rapport building
    finalize_stage_metrics(state, stage_name)

    return {
        "messages": [AIMessage(content=introduction_message)],
        "stage": 1,
        **metrics_update,
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
            "messages": [
                AIMessage(
                    content="I didn't receive your response. Could you share what brought you here today?"
                )
            ]
        }

    # Extract knowledge (already done in preprocessor, but get for prompt)
    extracted = state.last_extracted_knowledge or {}

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = render_stage1_meta(
        knowledge_context=knowledge_context,
        user_message=user_message,
        message_analysis=format_message_analysis(extracted),
    )

    # Create context messages
    system_prompt = render_system_prompt()
    context_messages = create_context_messages(
        state, system_prompt, meta_prompt, n_history=6
    )

    # Generate response
    response = llm.invoke(context_messages)
    question = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    # Update metrics
    tokens = (
        estimate_tokens(user_message)
        + estimate_tokens(question)
        + estimate_tokens(meta_prompt)
    )
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {"messages": [AIMessage(content=question)], **metrics_update}


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
            "messages": [
                AIMessage(
                    content="I'd love to hear more about why this matters to you."
                )
            ]
        }

    # Recalculate value weights
    weight_update = calculate_value_weights(state)

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)
    value_count = len(state.user_profile.values)

    # Build meta-prompt
    meta_prompt = render_stage2_meta(
        knowledge_context=knowledge_context,
        user_message=user_message,
        value_count=value_count,
    )

    # Create context messages
    system_prompt = render_system_prompt()
    context_messages = create_context_messages(
        state, system_prompt, meta_prompt, n_history=8
    )

    # Generate response
    response = llm.invoke(context_messages)
    question = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    # Update metrics
    tokens = (
        estimate_tokens(user_message)
        + estimate_tokens(question)
        + estimate_tokens(meta_prompt)
    )
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=question)],
        **weight_update,
        **metrics_update,
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
    elif user_message and any(
        word in user_message.lower()
        for word in ["prefer", "like", "choose", "option", "better"]
    ):
        plan_status = "feedback received - refine chosen approach and build out details"
    elif actions_count < 3:
        plan_status = "actions presented - gather preferences and feedback"
    else:
        plan_status = "plan developed - confirm feasibility and alignment"

    # Get current knowledge context
    knowledge_context = get_knowledge_context(state)

    # Build meta-prompt
    meta_prompt = render_stage3_meta(
        knowledge_context=knowledge_context,
        user_message=user_message or "[Starting action planning]",
        plan_status=plan_status,
        action_count=actions_count,
    )

    # Create context messages
    system_prompt = render_system_prompt()
    context_messages = create_context_messages(
        state, system_prompt, meta_prompt, n_history=8
    )

    # Generate response
    response = llm.invoke(context_messages)
    action_response = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    # Try to extract action suggestions from response
    action_items = re.findall(
        r"(?:\d+\.|[-*])\s+([^\n]+(?:\n(?!(?:\d+\.|[-*]))[^\n]+)*)", action_response
    )

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
        if not any(
            a.description == action_text for a in state.user_profile.suggested_actions
        ):
            action = ActionSuggestion(
                description=action_text, linked_values=linked_values, fit_score=0.8
            )
            state.user_profile.suggested_actions.append(action)

    # Update metrics
    tokens = (
        estimate_tokens(user_message or "")
        + estimate_tokens(action_response)
        + estimate_tokens(meta_prompt)
    )
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=action_response)],
        "user_profile": state.user_profile,
        **metrics_update,
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

    # Ensure we're in the correct stage
    stage_update = {}
    if state.stage != 4:
        stage_update = {"stage": 4}

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

        # No interrupt needed - session is ending
        return {
            "messages": [AIMessage(content=closing_message)],
            "final_feedback": user_message,
            "stage": 5,  # End stage
            **metrics_update,
        }

    # Otherwise, generate the summary
    knowledge_context = get_knowledge_context(state)

    meta_prompt = render_stage4_meta(knowledge_context=knowledge_context)

    system_prompt = render_system_prompt()
    context_messages = create_context_messages(
        state, system_prompt, meta_prompt, n_history=4
    )

    response = llm.invoke(context_messages)
    summary = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    # Update metrics
    tokens = estimate_tokens(summary) + estimate_tokens(meta_prompt)
    metrics_update = update_stage_metrics(state, stage_name, tokens)

    return {
        "messages": [AIMessage(content=summary)],
        "final_summary": summary,
        **metrics_update,
        **stage_update,
    }


def reply_node(state: MetaAgentState) -> dict:
    """REPLY node: Waits for user input via interrupt_before, then passes through.

    This node is interrupted before execution, allowing the user to respond.
    Once user responds, it simply continues execution to the appropriate validator.

    Args:
        state: Current agent state

    Returns:
        Empty dict (no state changes, just passes through)
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "Please provide your response to continue."
    )
    print(f"> Received an input from the interrupt: {answer}")
    return {"messages": [HumanMessage(content=answer)]}


# === VALIDATOR NODES ===


def validate_rapport_node(state: MetaAgentState) -> dict:
    """Validator for rapport building stage - decides if we should loop or advance.

    Uses LLM to assess completion criteria. If validation passes, updates stage to 2.

    Args:
        state: Current agent state

    Returns:
        Dictionary with stage update if validation passes
    """
    import json
    from meta_agent_v4.prompts import (
        render_stage1_validation,
        format_conversation_history,
    )
    from meta_agent_v4.utils import (
        count_user_messages_in_stage,
        get_conversation_history,
    )

    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    conversation_history = format_conversation_history(
        get_conversation_history(state, n_messages=12)
    )
    turn_count = count_user_messages_in_stage(state, 1)

    # Build validation prompt
    validation_prompt = render_stage1_validation(
        knowledge_context=knowledge_context,
        conversation_history=conversation_history,
        turn_count=turn_count,
    )

    try:
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an assessment system. Return only valid JSON.",
                },
                {"role": "user", "content": validation_prompt},
            ]
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            finalize_stage_metrics(state, "rapport_building")
            return {"stage": 2}  # Advance to value discovery

        print("Rapport validation assessment:", assessment)
        return {}  # Stay in rapport building

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic
        goals_count = len(state.user_profile.goals)

        # Force advancement after 8 turns or if we have goals and 5+ turns
        if turn_count >= 8 or (goals_count >= 1 and turn_count >= 5):
            finalize_stage_metrics(state, "rapport_building")
            return {"stage": 2}  # Advance to value discovery

        return {}  # Stay in rapport building


def validate_value_discovery_node(state: MetaAgentState) -> dict:
    """Validator for value discovery stage - decides if we should loop or advance.

    Uses LLM to assess completion criteria. If validation passes, updates stage to 3.

    Args:
        state: Current agent state

    Returns:
        Dictionary with stage update if validation passes
    """
    import json
    from meta_agent_v4.prompts import render_stage2_validation
    from meta_agent_v4.utils import count_user_messages_in_stage

    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    turn_count = count_user_messages_in_stage(state, 2)
    value_count = len(state.user_profile.values)

    # Build validation prompt
    validation_prompt = render_stage2_validation(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        value_count=value_count,
    )

    try:
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an assessment system. Return only valid JSON.",
                },
                {"role": "user", "content": validation_prompt},
            ]
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}  # Advance to action planning

        return {}  # Stay in value discovery

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic
        # Force advancement after 10 turns or if we have values and 6+ turns
        if turn_count >= 10 or (value_count >= 3 and turn_count >= 6):
            finalize_stage_metrics(state, "value_discovery")
            return {"stage": 3}  # Advance to action planning

        return {}  # Stay in value discovery


def validate_action_planning_node(state: MetaAgentState) -> dict:
    """Validator for action planning stage - decides if we should loop or advance.

    Uses LLM to assess completion criteria. If validation passes, updates stage to 4.

    Args:
        state: Current agent state

    Returns:
        Dictionary with stage update if validation passes
    """
    import json
    from meta_agent_v4.prompts import render_stage3_validation
    from meta_agent_v4.utils import count_user_messages_in_stage

    llm = get_llm(state.model_name, temperature=0.3)

    knowledge_context = get_knowledge_context(state)
    turn_count = count_user_messages_in_stage(state, 3)
    action_count = len(state.user_profile.suggested_actions)

    # Build validation prompt
    validation_prompt = render_stage3_validation(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        action_count=action_count,
    )

    try:
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an assessment system. Return only valid JSON.",
                },
                {"role": "user", "content": validation_prompt},
            ]
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should advance
        if assessment.get("should_advance"):
            finalize_stage_metrics(state, "action_planning")
            return {"stage": 4}  # Advance to summary feedback

        return {}  # Stay in action planning

    except (json.JSONDecodeError, Exception) as e:
        # Fallback logic
        # Force advancement after 7 turns or if we have actions and 4+ turns with feedback
        if turn_count >= 7 or (action_count >= 3 and turn_count >= 4):
            finalize_stage_metrics(state, "action_planning")
            return {"stage": 4}  # Advance to summary feedback

        return {}  # Stay in action planning


def validate_summary_node(state: MetaAgentState) -> dict:
    """Validator for summary feedback stage - decides if we should loop or end.

    Uses LLM to assess completion criteria. If validation passes, updates stage to 5.

    Args:
        state: Current agent state

    Returns:
        Dictionary with stage update if validation passes
    """
    import json
    from meta_agent_v4.prompts import render_stage4_validation
    from meta_agent_v4.utils import count_user_messages_in_stage

    llm = get_llm(state.model_name, temperature=0.3)

    turn_count = count_user_messages_in_stage(state, 4)
    has_feedback = bool(state.final_feedback)

    # Build validation prompt
    validation_prompt = render_stage4_validation(
        turn_count=turn_count, has_feedback=has_feedback
    )

    try:
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are an assessment system. Return only valid JSON.",
                },
                {"role": "user", "content": validation_prompt},
            ]
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group(1))
        else:
            assessment = json.loads(content)

        # Check if should end
        if assessment.get("should_end"):
            finalize_stage_metrics(state, "summary_feedback")
            return {"stage": 5}  # End session

        return {}  # Stay in summary feedback

    except (json.JSONDecodeError, Exception):
        # Fallback: end if we have feedback or 2+ turns
        if has_feedback or turn_count >= 2:
            finalize_stage_metrics(state, "summary_feedback")
            return {"stage": 5}  # End session

        return {}  # Stay in summary feedback
