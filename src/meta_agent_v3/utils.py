"""Utility functions for the meta agent."""

from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from meta_agent_v3.state import MetaAgentState, StageMetrics


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Initialize the language model.

    Args:
        model_name: Name of the OpenAI model to use
        temperature: Temperature for generation (0.0-1.0)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


def get_last_user_message(state: MetaAgentState) -> Optional[str]:
    """Extract the last user message from the conversation.

    Args:
        state: Current agent state

    Returns:
        Last user message content or None if no user messages exist
    """
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return None
    return str(user_messages[-1].content)


def get_conversation_history(state: MetaAgentState, n_messages: int = 6) -> list[dict]:
    """Get the last N messages formatted for LLM context.

    Args:
        state: Current agent state
        n_messages: Number of recent messages to include

    Returns:
        List of message dictionaries with role and content
    """
    messages = []
    for msg in state.messages[-n_messages:]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": str(msg.content)})
    return messages


def create_context_messages(
    state: MetaAgentState,
    system_prompt: str,
    meta_prompt: str,
    n_history: int = 6
) -> list[dict]:
    """Create a complete message list for LLM invocation.

    Args:
        state: Current agent state
        system_prompt: System-level instructions
        meta_prompt: Stage-specific meta-prompt with context
        n_history: Number of conversation history messages to include

    Returns:
        Complete list of messages for LLM
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_conversation_history(state, n_history))
    messages.append({"role": "user", "content": meta_prompt})
    return messages


def update_stage_metrics(state: MetaAgentState, stage_name: str, tokens_used: int = 0) -> dict:
    """Update or create metrics for the current stage.

    Args:
        state: Current agent state
        stage_name: Name of the current stage
        tokens_used: Number of tokens used in this turn

    Returns:
        Dictionary with updates to state
    """
    # Find or create metrics for current stage
    current_stage_metrics = None
    for metrics in state.stage_metrics:
        if metrics.stage_name == stage_name and not metrics.end_time:
            current_stage_metrics = metrics
            break

    if current_stage_metrics is None:
        # Create new stage metrics
        current_stage_metrics = StageMetrics(stage_name=stage_name)
        state.stage_metrics.append(current_stage_metrics)

    # Update metrics
    current_stage_metrics.turn_count += 1
    current_stage_metrics.total_tokens += tokens_used

    return {
        "stage_metrics": state.stage_metrics,
        "total_turns": state.total_turns + 1
    }


def finalize_stage_metrics(state: MetaAgentState, stage_name: str) -> None:
    """Mark a stage as complete by setting end time.

    Args:
        state: Current agent state
        stage_name: Name of the stage to finalize
    """
    for metrics in state.stage_metrics:
        if metrics.stage_name == stage_name and not metrics.end_time:
            metrics.end_time = datetime.utcnow().isoformat()
            break


def count_user_messages_in_stage(state: MetaAgentState, stage_number: int) -> int:
    """Count how many user messages have been sent in the current stage.

    Args:
        state: Current agent state
        stage_number: The stage number to count messages for

    Returns:
        Number of user messages in this stage
    """
    # Find the stage metrics
    stage_names = [
        "introduction",
        "rapport_building",
        "value_discovery",
        "value_ranking",
        "action_planning",
        "summary_feedback"
    ]

    if stage_number < 0 or stage_number >= len(stage_names):
        return 0

    stage_name = stage_names[stage_number]

    for metrics in state.stage_metrics:
        if metrics.stage_name == stage_name and not metrics.end_time:
            return metrics.turn_count

    return 0


def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text (1 token ≈ 4 chars).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def get_stage_name(stage_number: int) -> str:
    """Get the name of a stage from its number.

    Args:
        stage_number: Stage number (0-5)

    Returns:
        Stage name
    """
    stage_names = [
        "introduction",
        "rapport_building",
        "value_discovery",
        "value_ranking",
        "action_planning",
        "summary_feedback"
    ]

    if 0 <= stage_number < len(stage_names):
        return stage_names[stage_number]
    return "unknown"

