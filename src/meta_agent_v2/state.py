"""State structures for the meta-prompting intent-aware conversational agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Goal:
    """Represents a user goal with associated values and metadata."""
    
    id: str
    statement: str
    original_phrasing: str = ""
    confirmed: bool = False
    values: list[str] = field(default_factory=list)  # References to value names
    rationale: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class IntentContext:
    """Captures initial context from user."""
    
    goal_statement: str = ""
    desired_outcome: str = ""
    emotional_tone: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


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
    conversation_history: list[str] = field(default_factory=list)


@dataclass
class MetaAgentState:
    """Complete state for the meta-prompting agent."""
    
    # Conversation messages
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    
    # Current stage (1-7)
    stage: int = 1
    
    # User knowledge profile
    user_profile: UserProfile = field(default_factory=UserProfile)
    
    # Conversation metrics
    trust_level: float = 0.5
    intent_confirmed: bool = False
    value_depth: int = 0
    target_value_depth: int = 3
    values_confirmed: bool = False
    plan_generated: bool = False
    
    # Last generated question (for tracking)
    last_question: str = ""
    
    # Stage completion flags
    stage_1_complete: bool = False
    stage_2_complete: bool = False
    stage_3_complete: bool = False
    stage_4_complete: bool = False
    stage_5_complete: bool = False
    stage_6_complete: bool = False
    stage_7_complete: bool = False
    
    # Final output
    final_summary: str = ""
    final_feedback: str = ""


@dataclass
class InputState:
    """Input state with just messages."""
    
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
