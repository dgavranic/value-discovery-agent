"""Prompts and meta-prompts for each stage of the value discovery agent.

This module uses Jinja2 templates for flexible prompt rendering.
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Set up Jinja2 environment
_template_dir = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(_template_dir),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_system_prompt() -> str:
    """Render the system prompt."""
    template = _jinja_env.get_template("system_prompt.j2")
    return template.render()


def render_introduction() -> str:
    """Render the introduction message."""
    template = _jinja_env.get_template("introduction.j2")
    return template.render()


def render_stage1_meta(knowledge_context: str, user_message: str, message_analysis: str) -> str:
    """Render Stage 1 (Rapport Building) meta-prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        user_message: User's last message
        message_analysis: Analysis of the message
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage1_meta.j2")
    return template.render(
        knowledge_context=knowledge_context,
        user_message=user_message,
        message_analysis=message_analysis
    )


def render_stage2_meta(knowledge_context: str, user_message: str, value_count: int) -> str:
    """Render Stage 2 (Value Discovery) meta-prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        user_message: User's last message
        value_count: Number of values discovered
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage2_meta.j2")
    return template.render(
        knowledge_context=knowledge_context,
        user_message=user_message,
        value_count=value_count
    )


def render_stage3_meta(knowledge_context: str, user_message: str, plan_status: str, action_count: int) -> str:
    """Render Stage 3 (Action Planning) meta-prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        user_message: User's last message
        plan_status: Current status of action planning
        action_count: Number of actions suggested
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage3_meta.j2")
    return template.render(
        knowledge_context=knowledge_context,
        user_message=user_message,
        plan_status=plan_status,
        action_count=action_count
    )


def render_stage4_meta(knowledge_context: str) -> str:
    """Render Stage 4 (Summary & Feedback) meta-prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage4_meta.j2")
    return template.render(knowledge_context=knowledge_context)


def render_stage1_validation(knowledge_context: str, conversation_history: str, turn_count: int) -> str:
    """Render Stage 1 validation prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        conversation_history: Recent conversation turns
        turn_count: Number of turns in this stage
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage1_validation.j2")
    return template.render(
        knowledge_context=knowledge_context,
        conversation_history=conversation_history,
        turn_count=turn_count
    )


def render_stage2_validation(knowledge_context: str, turn_count: int, value_count: int) -> str:
    """Render Stage 2 validation prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        turn_count: Number of turns in this stage
        value_count: Number of values discovered
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage2_validation.j2")
    return template.render(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        value_count=value_count
    )


def render_stage3_validation(knowledge_context: str, turn_count: int, action_count: int) -> str:
    """Render Stage 3 validation prompt.
    
    Args:
        knowledge_context: Current knowledge map summary
        turn_count: Number of turns in this stage
        action_count: Number of actions suggested
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage3_validation.j2")
    return template.render(
        knowledge_context=knowledge_context,
        turn_count=turn_count,
        action_count=action_count
    )


def render_stage4_validation(turn_count: int, has_feedback: bool) -> str:
    """Render Stage 4 validation prompt.
    
    Args:
        turn_count: Number of turns in this stage
        has_feedback: Whether final feedback has been received
        
    Returns:
        Rendered prompt string
    """
    template = _jinja_env.get_template("stage4_validation.j2")
    return template.render(
        turn_count=turn_count,
        has_feedback=has_feedback
    )


# Backward compatibility constants (lazy-loaded)
SYSTEM_PROMPT = None
INTRODUCTION_MESSAGE = None


def _get_system_prompt():
    """Lazy load system prompt."""
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        SYSTEM_PROMPT = render_system_prompt()
    return SYSTEM_PROMPT


def _get_introduction():
    """Lazy load introduction."""
    global INTRODUCTION_MESSAGE
    if INTRODUCTION_MESSAGE is None:
        INTRODUCTION_MESSAGE = render_introduction()
    return INTRODUCTION_MESSAGE


# Legacy compatibility - keep old constants but load them lazily
def __getattr__(name):
    """Lazy load legacy prompt constants."""
    if name == "SYSTEM_PROMPT":
        return _get_system_prompt()
    elif name == "INTRODUCTION_MESSAGE":
        return _get_introduction()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def format_message_analysis(extracted: dict) -> str:
    """Format extracted knowledge for prompt inclusion."""
    return f"""
Message Length: {extracted.get('message_length', 'unknown')}
Engagement Level: {extracted.get('engagement_level', 'unknown')}
Emotional Tone: {extracted.get('emotional_tone', 'neutral')}
Goals Mentioned: {', '.join(extracted.get('goals_mentioned', [])) or 'none'}
Values Mentioned: {', '.join(extracted.get('values_mentioned', [])) or 'none'}
Key Phrases: {' | '.join(extracted.get('key_phrases', [])) or 'none'}
"""


def format_conversation_history(history: list) -> str:
    """Format conversation history for prompts."""
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

