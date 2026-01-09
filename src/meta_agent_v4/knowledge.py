"""Knowledge extraction and management for building user profile."""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from meta_agent_v4.state import Goal, Value, MetaAgentState


# === EXTRACTION ===

def extract_knowledge(user_message: str, llm: ChatOpenAI) -> dict[str, Any]:
    """Extract structured knowledge from user message using LLM.

    Args:
        user_message: The user's text response
        llm: Language model for extraction

    Returns:
        Dictionary with extracted knowledge:
        - goals_mentioned: list[str]
        - values_mentioned: list[str]
        - emotional_tone: str
        - obstacles_mentioned: list[str]
        - key_phrases: list[str]
        - context_details: list[str]
        - message_length: str (short/medium/long)
        - engagement_level: str (low/medium/high)
    """
    extraction_prompt = f"""Analyze the user's response and extract structured information.

User's response: "{user_message}"

Extract and return ONLY valid JSON (no markdown, no explanation):
{{
  "goals_mentioned": ["list of any goals or objectives mentioned"],
  "values_mentioned": ["list of underlying values, motivations, or what matters to them"],
  "emotional_tone": "description of emotional tone (e.g., excited, anxious, hopeful, frustrated)",
  "obstacles_mentioned": ["list of any barriers or concerns mentioned"],
  "key_phrases": ["important phrases in their own words, max 3"],
  "context_details": ["specific details about their situation"],
  "message_length": "short|medium|long",
  "engagement_level": "low|medium|high"
}}

Guidelines:
- Only include what is clearly present in the response
- Be thorough but accurate
- Infer values from what they care about
- message_length: short (<20 words), medium (20-50 words), long (>50 words)
- engagement_level: assess based on depth and specificity of response"""

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a precise knowledge extraction system. Always return valid JSON only."},
            {"role": "user", "content": extraction_prompt}
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Try to extract JSON from the response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        return json.loads(content)

    except (json.JSONDecodeError, Exception):
        # Fallback: return basic structure
        words = user_message.split()
        word_count = len(words)

        return {
            "goals_mentioned": [],
            "values_mentioned": [],
            "emotional_tone": "neutral",
            "obstacles_mentioned": [],
            "key_phrases": [user_message[:100]],
            "context_details": [],
            "message_length": "short" if word_count < 20 else ("medium" if word_count < 50 else "long"),
            "engagement_level": "medium"
        }


# === KNOWLEDGE MAP UPDATE ===

def update_knowledge_map(state: MetaAgentState, extracted: dict[str, Any]) -> dict:
    """Update the knowledge map with extracted information.

    Args:
        state: Current agent state
        extracted: Extracted knowledge from user message

    Returns:
        Dictionary with updates to apply to state
    """
    user_profile = state.user_profile

    # Update goals
    for goal_text in extracted.get("goals_mentioned", []):
        if goal_text and len(goal_text) > 5:
            goal_id = f"g{len(user_profile.goals) + 1}"
            # Check if similar goal already exists
            existing = False
            for existing_goal in user_profile.goals.values():
                if goal_text.lower() in existing_goal.statement.lower() or existing_goal.statement.lower() in goal_text.lower():
                    existing = True
                    break

            if not existing:
                goal = Goal(
                    id=goal_id,
                    statement=goal_text,
                    original_phrasing=goal_text
                )
                user_profile.goals[goal_id] = goal

    # Update values
    for value_text in extracted.get("values_mentioned", []):
        if value_text and len(value_text) > 2:
            value_name = value_text.lower().strip()

            if value_name not in user_profile.values:
                user_profile.values[value_name] = Value(name=value_name, weight=0.5)
            else:
                # Increase weight when value is mentioned again
                current_value = user_profile.values[value_name]
                current_value.weight = min(1.0, current_value.weight + 0.1)

            # Add rationale from key phrases
            if extracted.get("key_phrases"):
                for phrase in extracted["key_phrases"]:
                    if phrase and phrase not in user_profile.values[value_name].rationale:
                        user_profile.values[value_name].rationale.append(phrase)

    # Update emotional tone
    if extracted.get("emotional_tone"):
        user_profile.intent_context.emotional_tone = extracted["emotional_tone"]

    # Update obstacles for goals
    if extracted.get("obstacles_mentioned") and user_profile.goals:
        goals_list = list(user_profile.goals.values())
        if goals_list:
            most_recent_goal = goals_list[-1]
            for obstacle in extracted["obstacles_mentioned"]:
                if obstacle not in most_recent_goal.obstacles:
                    most_recent_goal.obstacles.append(obstacle)

    # Update goal statement in intent context if not set
    if not user_profile.intent_context.goal_statement and user_profile.goals:
        goals_list = list(user_profile.goals.values())
        if goals_list:
            user_profile.intent_context.goal_statement = goals_list[0].statement

    return {"user_profile": user_profile}


# === KNOWLEDGE CONTEXT FORMATTING ===

def get_knowledge_context(state: MetaAgentState) -> str:
    """Format current knowledge map as context string for prompt injection.

    Args:
        state: Current agent state

    Returns:
        Formatted string summarizing current knowledge
    """
    profile = state.user_profile
    context_parts = []

    # Goals
    if profile.goals:
        goals_text = "Identified Goals:\n"
        for goal in profile.goals.values():
            status = " [CONFIRMED]" if goal.confirmed else ""
            goals_text += f"  - {goal.statement}{status}\n"
            if goal.obstacles:
                goals_text += f"    Obstacles: {', '.join(goal.obstacles)}\n"
        context_parts.append(goals_text)

    # Values
    if profile.values:
        values_text = "Discovered Values:\n"
        sorted_values = sorted(profile.values.values(), key=lambda v: v.weight, reverse=True)
        for value in sorted_values:
            status = " [CONFIRMED]" if value.confirmed else ""
            values_text += f"  - {value.name} (weight: {value.weight:.2f}){status}\n"
            if value.rationale:
                values_text += f"    Context: {value.rationale[-1]}\n"
        context_parts.append(values_text)

    # Intent context
    if profile.intent_context.emotional_tone or profile.intent_context.desired_outcome:
        intent_text = "Emotional Context:\n"
        if profile.intent_context.emotional_tone:
            intent_text += f"  Tone: {profile.intent_context.emotional_tone}\n"
        if profile.intent_context.desired_outcome:
            intent_text += f"  Desired outcome: {profile.intent_context.desired_outcome}\n"
        context_parts.append(intent_text)

    # Actions if any
    if profile.suggested_actions:
        actions_text = "Suggested Actions:\n"
        for action in profile.suggested_actions:
            actions_text += f"  - {action.description}\n"
            if action.linked_values:
                actions_text += f"    Aligns with: {', '.join(action.linked_values)}\n"
        context_parts.append(actions_text)

    if not context_parts:
        return "No knowledge extracted yet."

    return "\n".join(context_parts)


def calculate_value_weights(state: MetaAgentState) -> dict:
    """Recalculate value weights based on mention frequency and context.

    Args:
        state: Current agent state

    Returns:
        Dictionary with updated user_profile
    """
    profile = state.user_profile

    for value in profile.values.values():
        # Weight based on rationale count (more mentions = higher weight)
        mention_count = len(value.rationale)
        base_weight = min(1.0, 0.3 + (mention_count * 0.15))

        # Boost if confirmed
        if value.confirmed:
            base_weight = min(1.0, base_weight + 0.2)

        value.weight = base_weight

    return {"user_profile": profile}

