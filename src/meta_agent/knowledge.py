"""Knowledge extraction utilities for parsing user responses into structured data."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from meta_agent.prompts import EXTRACTION_PROMPT
from meta_agent.state import Goal, Value, ActionSuggestion, MetaAgentState


def extract_knowledge(user_response: str, model: ChatOpenAI) -> dict[str, Any]:
    """Extract structured knowledge from user response using LLM.
    
    Args:
        user_response: The user's text response
        model: The LLM to use for extraction
        
    Returns:
        Dictionary with extracted knowledge
    """
    prompt = EXTRACTION_PROMPT.format(user_response=user_response)
    
    response = model.invoke([
        {"role": "system", "content": "You are a precise knowledge extraction system. Always return valid JSON."},
        {"role": "user", "content": prompt}
    ])
    
    content = response.content if isinstance(response.content, str) else str(response.content)
    
    # Try to extract JSON from the response
    try:
        # Look for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try parsing the whole content
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return basic structure
        return {
            "goals_mentioned": [],
            "values_mentioned": [],
            "emotional_tone": "neutral",
            "obstacles_mentioned": [],
            "clarifications": [],
            "key_phrases": [user_response[:100]]
        }


def update_knowledge_graph(state: MetaAgentState, extracted: dict[str, Any]) -> dict[str, Any]:
    """Update the knowledge graph with extracted information.
    
    Args:
        state: Current agent state
        extracted: Extracted knowledge dictionary
        
    Returns:
        Dictionary with updates to apply to state
    """
    updates = {}
    user_profile = state.user_profile
    
    # Update goals
    for goal_text in extracted.get("goals_mentioned", []):
        if goal_text and len(goal_text) > 5:  # Basic validation
            goal_id = f"g{len(user_profile.goals) + 1}"
            if goal_id not in user_profile.goals:
                goal = Goal(
                    id=goal_id,
                    statement=goal_text,
                    original_phrasing=goal_text
                )
                user_profile.goals[goal_id] = goal
    
    # Update values
    for value_text in extracted.get("values_mentioned", []):
        if value_text and len(value_text) > 2:
            # Normalize value name (lowercase, remove extra spaces)
            value_name = value_text.lower().strip()
            
            if value_name not in user_profile.values:
                user_profile.values[value_name] = Value(name=value_name, weight=0.5)
            else:
                # Increase weight when value is mentioned again
                current_value = user_profile.values[value_name]
                current_value.weight = min(1.0, current_value.weight + 0.1)
            
            # Add rationale if available
            if extracted.get("key_phrases"):
                user_profile.values[value_name].rationale.extend(extracted["key_phrases"])
    
    # Update emotional tone in intent context
    if extracted.get("emotional_tone"):
        user_profile.intent_context.emotional_tone = extracted["emotional_tone"]
    
    # Update obstacles for relevant goals
    if extracted.get("obstacles_mentioned"):
        for goal in user_profile.goals.values():
            if not goal.confirmed:
                continue
            goal.obstacles.extend(extracted["obstacles_mentioned"])
    
    # Store conversation turn
    user_profile.conversation_history.append(
        f"User: {extracted.get('key_phrases', [''])[0]}"
    )
    
    updates["user_profile"] = user_profile
    
    return updates


def identify_conflicting_values(state: MetaAgentState) -> tuple[str, str] | None:
    """Identify two values that might conflict based on common value tensions.
    
    Args:
        state: Current agent state
        
    Returns:
        Tuple of two conflicting value names, or None
    """
    values = list(state.user_profile.values.keys())
    
    if len(values) < 2:
        return None
    
    # Common value conflicts
    conflict_patterns = [
        (["freedom", "independence", "autonomy"], ["security", "stability", "safety"]),
        (["speed", "efficiency", "quick"], ["quality", "excellence", "mastery"]),
        (["innovation", "creativity", "new"], ["tradition", "proven", "reliable"]),
        (["wealth", "money", "financial"], ["purpose", "meaning", "fulfillment"]),
        (["career", "achievement", "success"], ["family", "relationships", "connection"]),
        (["control", "power", "leadership"], ["collaboration", "teamwork", "harmony"]),
    ]
    
    # Check for conflicts
    for pattern1, pattern2 in conflict_patterns:
        matches1 = [v for v in values if any(p in v for p in pattern1)]
        matches2 = [v for v in values if any(p in v for p in pattern2)]
        
        if matches1 and matches2:
            return matches1[0], matches2[0]
    
    # If no pattern match, just return first two with highest weights
    sorted_values = sorted(
        state.user_profile.values.items(),
        key=lambda x: x[1].weight,
        reverse=True
    )
    
    if len(sorted_values) >= 2:
        return sorted_values[0][0], sorted_values[1][0]
    
    return None


def format_values_summary(state: MetaAgentState) -> str:
    """Format values for display in prompts.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted string of values
    """
    if not state.user_profile.values:
        return "No values identified yet"
    
    sorted_values = sorted(
        state.user_profile.values.items(),
        key=lambda x: x[1].weight,
        reverse=True
    )
    
    lines = []
    for name, value in sorted_values:
        status = "✓" if value.confirmed else "?"
        lines.append(f"{status} {name.capitalize()} (weight: {value.weight:.2f})")
    
    return "\n".join(lines)


def format_goals_summary(state: MetaAgentState) -> str:
    """Format goals for display in prompts.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted string of goals
    """
    if not state.user_profile.goals:
        return "No goals identified yet"
    
    lines = []
    for goal_id, goal in state.user_profile.goals.items():
        status = "✓" if goal.confirmed else "?"
        lines.append(f"{status} {goal.statement}")
    
    return "\n".join(lines)


def create_final_summary(state: MetaAgentState) -> str:
    """Create final summary of discovered values and goals.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted summary string
    """
    summary_parts = [
        "🎯 **Discovery Summary**",
        "",
        "Based on our conversation, here's what I understand about what matters most to you:",
        "",
    ]
    
    # Values section
    if state.user_profile.values:
        summary_parts.append("**Core Values:**")
        sorted_values = sorted(
            state.user_profile.values.items(),
            key=lambda x: x[1].weight,
            reverse=True
        )
        
        for name, value in sorted_values[:5]:  # Top 5 values
            rationale = value.rationale[0] if value.rationale else "No specific rationale captured"
            summary_parts.append(f"• **{name.capitalize()}** (confidence: {value.weight:.0%})")
            if value.rationale:
                summary_parts.append(f"  ↳ _{rationale[:100]}..._")
        summary_parts.append("")
    
    # Goals section
    if state.user_profile.goals:
        summary_parts.append("**Primary Goals:**")
        for goal in state.user_profile.goals.values():
            summary_parts.append(f"• {goal.statement}")
            if goal.values:
                linked = ", ".join(goal.values)
                summary_parts.append(f"  ↳ Connected to: {linked}")
        summary_parts.append("")
    
    # Action suggestions
    if state.user_profile.suggested_actions:
        summary_parts.append("**Suggested Next Steps:**")
        for i, action in enumerate(state.user_profile.suggested_actions, 1):
            summary_parts.append(f"{i}. {action.description}")
            if action.linked_values:
                summary_parts.append(f"   ↳ Aligns with: {', '.join(action.linked_values)}")
        summary_parts.append("")
    
    # Feedback request
    summary_parts.extend([
        "---",
        "",
        "**Please provide feedback:**",
        "• Does this accurately capture what matters to you?",
        "• What would you change or refine?",
        "• What's missing or misunderstood?"
    ])
    
    return "\n".join(summary_parts)
