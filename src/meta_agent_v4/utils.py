"""Utility functions for meta agent v4."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from meta_agent_v4.state import MetaAgentState, StageMetrics


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Get configured LLM instance.

    Args:
        model_name: Name of the model to use
        temperature: Temperature for generation

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


def get_last_user_message(state: MetaAgentState) -> str | None:
    """Get the last user message from state.

    Args:
        state: Current agent state

    Returns:
        Last user message content or None
    """
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return message.content
    return None


def count_user_messages_in_stage(state: MetaAgentState, stage: int) -> int:
    """Count user messages since entering a specific stage.

    Args:
        state: Current agent state
        stage: Stage number to count for

    Returns:
        Number of user messages in that stage
    """
    # Find the stage metric
    stage_names = {
        0: "introduction",
        1: "rapport_building",
        2: "value_discovery",
        3: "action_planning",
        4: "summary_feedback"
    }

    stage_name = stage_names.get(stage)
    if not stage_name:
        return 0

    # Find the metric for this stage
    for metric in state.stage_metrics:
        if metric.stage_name == stage_name and not metric.end_time:
            return metric.turn_count

    return 0


def get_conversation_history(state: MetaAgentState, n_messages: int = 10) -> list[dict]:
    """Get recent conversation history.

    Args:
        state: Current agent state
        n_messages: Number of recent messages to return

    Returns:
        List of message dictionaries with role and content
    """
    history = []
    for message in state.messages[-n_messages:]:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            history.append({"role": "assistant", "content": message.content})
    return history


def create_context_messages(
    state: MetaAgentState,
    system_prompt: str,
    meta_prompt: str,
    n_history: int = 6
) -> list[dict]:
    """Create context messages for LLM including system, meta-prompt, and history.

    Args:
        state: Current agent state
        system_prompt: Base system prompt
        meta_prompt: Stage-specific meta-prompt
        n_history: Number of conversation messages to include

    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"[META-INSTRUCTION]\n{meta_prompt}"}
    ]

    # Add conversation history
    history = get_conversation_history(state, n_messages=n_history)
    messages.extend(history)

    return messages


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def update_stage_metrics(state: MetaAgentState, stage_name: str, tokens: int) -> dict:
    """Update metrics for the current stage.

    Args:
        state: Current agent state
        stage_name: Name of the current stage
        tokens: Number of tokens used in this turn

    Returns:
        Dictionary with metrics updates
    """
    # Find or create metric for this stage
    current_metric = None
    for metric in state.stage_metrics:
        if metric.stage_name == stage_name and not metric.end_time:
            current_metric = metric
            break

    if not current_metric:
        # Create new metric
        current_metric = StageMetrics(stage_name=stage_name)
        state.stage_metrics.append(current_metric)

    # Update metrics
    current_metric.turn_count += 1
    current_metric.total_tokens += tokens

    return {
        "stage_metrics": state.stage_metrics,
        "total_turns": state.total_turns + 1
    }


def finalize_stage_metrics(state: MetaAgentState, stage_name: str):
    """Mark a stage as complete.

    Args:
        state: Current agent state
        stage_name: Name of the stage to finalize
    """
    from datetime import datetime, timezone

    for metric in state.stage_metrics:
        if metric.stage_name == stage_name and not metric.end_time:
            metric.end_time = datetime.now(timezone.utc).isoformat()
            break


def format_conversation_history(history: list[dict]) -> str:
    """Format conversation history for display.

    Args:
        history: List of message dictionaries

    Returns:
        Formatted string
    """
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

