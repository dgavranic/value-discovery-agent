"""State structures for meta agent v4 with simplified interrupt-based flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass
class Value:
    """Represents a user value with weight and supporting rationale."""

    name: str
    weight: float = 0.5
    rationale: list[str] = field(default_factory=list)
    confirmed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Goal:
    """Represents a user goal with associated values and metadata."""

    id: str
    statement: str
    original_phrasing: str = ""
    confirmed: bool = False
    values: list[str] = field(default_factory=list)  # References to value names
    rationale: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class IntentContext:
    """Captures initial context from user."""

    goal_statement: str = ""
    desired_outcome: str = ""
    emotional_tone: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ActionSuggestion:
    """Represents a suggested action with value alignment."""

    description: str
    linked_values: list[str] = field(default_factory=list)
    user_feedback: str = ""
    fit_score: float = 0.0


@dataclass
class UserProfile:
    """Complete user profile with goals, values, and metadata."""

    goals: dict[str, Goal] = field(default_factory=dict)
    values: dict[str, Value] = field(default_factory=dict)
    intent_context: IntentContext = field(default_factory=IntentContext)
    suggested_actions: list[ActionSuggestion] = field(default_factory=list)


@dataclass
class StageMetrics:
    """Metrics tracking for each stage."""

    stage_name: str
    turn_count: int = 0
    total_tokens: int = 0
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str = ""


@dataclass
class MetaAgentState:
    """Complete state for the meta-prompting agent with interrupt-based flow."""

    # Conversation messages
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

    # Current stage (0-4: introduction, rapport, value_discovery, action_planning, summary)
    stage: int = 0

    # User knowledge profile
    user_profile: UserProfile = field(default_factory=UserProfile)

    # Final output
    final_summary: str = ""
    final_feedback: str = ""

    # Metrics tracking (for A/B testing and analysis)
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    total_turns: int = 0
    session_start: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model configuration (for A/B testing)
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7


@dataclass
class InputState:
    """Input state with just messages."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

