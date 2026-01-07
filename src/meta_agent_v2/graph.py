"""Main graph implementation for meta-prompting agent."""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from opik.integrations.langchain import OpikTracer

from meta_agent_v2.knowledge import (
    create_final_summary,
    extract_knowledge,
    format_goals_summary,
    format_values_summary,
    identify_conflicting_values,
    update_knowledge_graph,
)
from meta_agent_v2.opik_logger import log_final_feedback
from meta_agent_v2.prompts import (
    STAGE_1_META_PROMPT,
    STAGE_2_META_PROMPT,
    STAGE_3_META_PROMPT,
    STAGE_4_META_PROMPT,
    STAGE_5_META_PROMPT,
    STAGE_6_META_PROMPT,
    STAGE_7_META_PROMPT,
    SYSTEM_PROMPT,
)
from meta_agent_v2.state import ActionSuggestion, InputState, MetaAgentState


def get_llm() -> ChatOpenAI:
    """Initialize the language model."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# === STAGE NODES ===


# Stage 0: Introduction
def introduction(state: MetaAgentState) -> dict:
    """Stage 0: Introduction - Explain the concept and provide initial prompt."""
    intro_message = """Welcome to your Value Discovery Journey! 🌟

I'm here to help you explore what truly matters to you through a guided conversation. Together, we'll:

• Uncover your core values and what drives you
• Understand the 'why' behind your goals
• Create an action plan that aligns with your authentic self

This is a safe space for reflection. There are no right or wrong answers - only your truth. The journey typically takes about 15-20 minutes, and we'll move at your pace.

**To begin your journey, tell me:** What brings you here today? What would you like to explore or achieve?"""

    return {
        "messages": [AIMessage(content=intro_message)],
        "last_question": intro_message,
        "stage": 0,
    }


# Stage 1: Rapport Building
def rapport_building(state: MetaAgentState) -> dict:
    """Stage 1: Build rapport and gather context."""
    llm = get_llm()

    # Get last user message
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {
            "messages": [AIMessage(content="I didn't receive your response. Could you share what brought you here?")]
        }

    last_user_msg = user_messages[-1].content

    # Extract initial knowledge
    extracted = extract_knowledge(str(last_user_msg), llm)
    updates = update_knowledge_graph(state, extracted)

    # Build context-aware prompt with conversation history
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add conversation history
    for msg in state.messages[-4:]:  # Last 4 messages for context
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add meta-prompt with current state
    state_context = f"""
Current knowledge extracted:
- Goals mentioned: {extracted.get('goals_mentioned', [])}
- Values detected: {extracted.get('values_mentioned', [])}
- Emotional tone: {extracted.get('emotional_tone', 'neutral')}

{STAGE_1_META_PROMPT}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate follow-up question
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 1
    updates["stage_1_complete"] = True

    return updates


# Stage 2: Intent Clarification
def intent_clarification(state: MetaAgentState) -> dict:
    """Stage 2: Reflect back and clarify understanding."""
    llm = get_llm()

    # Get last user message
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="I didn't receive your response. Could you share more?")]}

    last_user_msg = user_messages[-1].content

    # Extract knowledge from user response
    extracted = extract_knowledge(str(last_user_msg), llm)
    updates = update_knowledge_graph(state, extracted)

    # Update intent context
    if extracted.get("goals_mentioned"):
        updates["user_profile"].intent_context.goal_statement = extracted["goals_mentioned"][0]
    if extracted.get("key_phrases"):
        updates["user_profile"].intent_context.desired_outcome = " ".join(extracted["key_phrases"][:2])

    # Build context-aware prompt with conversation history and state
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add conversation history (keep more context for clarification)
    for msg in state.messages[-6:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current understanding:
- Goals identified: {format_goals_summary(updates.get('user_profile', state.user_profile))}
- Intent context: {updates.get('user_profile', state.user_profile).intent_context.goal_statement}
- Emotional tone: {extracted.get('emotional_tone', 'neutral')}

{STAGE_2_META_PROMPT}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate reflection and clarification
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 2
    updates["stage_2_complete"] = True

    return updates


# Stage 3: Value Discovery (Laddering)
def value_discovery(state: MetaAgentState) -> dict:
    """Stage 3: Explore deeper values through 'why' questions."""
    llm = get_llm()

    # Get current goal
    goals = list(state.user_profile.goals.values())
    current_goal = goals[0].statement if goals else state.user_profile.intent_context.goal_statement

    if not current_goal:
        current_goal = "what you shared earlier"

    # Extract and update from previous user response
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_msg = user_messages[-1].content
        extracted = extract_knowledge(str(last_user_msg), llm)
        updates = update_knowledge_graph(state, extracted)

        # Check for confirmation in stage 2
        if state.stage == 2 and any(
                word in str(last_user_msg).lower() for word in ["yes", "correct", "right", "exactly", "accurate"]
        ):
            updates["intent_confirmed"] = True
            # Mark first goal as confirmed
            if state.user_profile.goals:
                goals_list = list(state.user_profile.goals.values())
                if goals_list:
                    goals_list[0].confirmed = True
    else:
        updates = {}

    # Build context-aware prompt with conversation history
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add relevant conversation history
    for msg in state.messages[-6:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current state:
- Confirmed goal: {current_goal}
- Values discovered so far: {format_values_summary(updates.get('user_profile', state.user_profile))}
- Current depth: {state.value_depth}
- Target depth: {state.target_value_depth}

{STAGE_3_META_PROMPT.format(goal=current_goal, value_depth=state.value_depth, target_depth=state.target_value_depth)}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate laddering question
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    # Increment value depth
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 3
    updates["value_depth"] = state.value_depth + 1

    # Mark as complete when target depth is reached
    if state.value_depth + 1 >= state.target_value_depth:
        updates["stage_3_complete"] = True

    return updates


# Stage 4: Value Trade-offs
def value_tradeoffs(state: MetaAgentState) -> dict:
    """Stage 4: Explore value trade-offs and priorities."""
    llm = get_llm()

    # Extract knowledge from previous response
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_msg = user_messages[-1].content
        extracted = extract_knowledge(str(last_user_msg), llm)
        updates = update_knowledge_graph(state, extracted)
    else:
        updates = {}

    # Identify conflicting values
    conflict = identify_conflicting_values(updates.get('user_profile', state.user_profile))

    if not conflict:
        # Skip to next stage if no conflicts found
        updates["stage"] = 4
        updates["stage_4_complete"] = True
        return updates

    value1, value2 = conflict
    values_text = f"{value1} and {value2}"

    # Build context-aware prompt
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent conversation history
    for msg in state.messages[-4:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current state:
- Values identified: {format_values_summary(updates.get('user_profile', state.user_profile))}
- Potential conflict between: {value1} and {value2}

{STAGE_4_META_PROMPT.format(values=values_text)}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate trade-off question
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 4
    updates["stage_4_complete"] = True

    return updates


# Stage 5: Value Confirmation
def value_confirmation(state: MetaAgentState) -> dict:
    """Stage 5: Reflect on and confirm discovered values."""
    llm = get_llm()

    # Extract knowledge from previous response
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_msg = user_messages[-1].content
        extracted = extract_knowledge(str(last_user_msg), llm)
        updates = update_knowledge_graph(state, extracted)

        # Update value weights based on trade-off response
        # (simplified - in production, use more sophisticated analysis)
        for value_name in state.user_profile.values.keys():
            if value_name in str(last_user_msg).lower():
                state.user_profile.values[value_name].weight += 0.15
    else:
        updates = {}

    # Format values summary
    values_summary = format_values_summary(updates.get('user_profile', state.user_profile))

    # Build context-aware prompt
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent conversation history
    for msg in state.messages[-6:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current state:
- Values discovered: {values_summary}
- Conversation depth: {state.value_depth} layers

{STAGE_5_META_PROMPT.format(values_summary=values_summary)}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate reflection question
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 5
    updates["stage_5_complete"] = True

    return updates


# Stage 6: Action Planning
def action_planning(state: MetaAgentState) -> dict:
    """Stage 6: Generate value-aligned action plan."""
    llm = get_llm()

    # Extract knowledge and check for confirmation
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_msg = user_messages[-1].content
        extracted = extract_knowledge(str(last_user_msg), llm)
        updates = update_knowledge_graph(state, extracted)

        # Check for values confirmation
        if any(word in str(last_user_msg).lower() for word in ["yes", "accurate", "right", "correct", "captures"]):
            updates["values_confirmed"] = True
            # Mark top values as confirmed
            for value in state.user_profile.values.values():
                if value.weight > 0.6:
                    value.confirmed = True
    else:
        updates = {}

    # Format goals and values
    goals_text = format_goals_summary(updates.get('user_profile', state.user_profile))
    values_text = format_values_summary(updates.get('user_profile', state.user_profile))

    # Build context-aware prompt
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add relevant conversation history (focusing on goals and values)
    for msg in state.messages[-8:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current state:
- Confirmed goals: {goals_text}
- Confirmed values: {values_text}
- Values confirmed by user: {updates.get('values_confirmed', state.values_confirmed)}

{STAGE_6_META_PROMPT.format(goals=goals_text, values=values_text)}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate plan with meta-prompt
    response = llm.invoke(context_messages)
    plan = response.content if isinstance(response.content, str) else str(response.content)

    # Parse action suggestions (simplified)
    # In production, use structured output or better parsing
    top_values = sorted(state.user_profile.values.items(), key=lambda x: x[1].weight, reverse=True)[:3]

    action = ActionSuggestion(
        description="Action plan generated", linked_values=[v[0] for v in top_values], fit_score=0.8
    )
    state.user_profile.suggested_actions.append(action)

    updates["messages"] = [AIMessage(content=plan)]
    updates["last_question"] = plan
    updates["stage"] = 6
    updates["plan_generated"] = True
    updates["stage_6_complete"] = True
    updates["user_profile"] = state.user_profile

    return updates


# Stage 7: Plan Refinement
def plan_refinement(state: MetaAgentState) -> dict:
    """Stage 7: Refine plan based on feedback."""
    llm = get_llm()

    # Extract knowledge from feedback
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"stage": 7, "stage_7_complete": True}

    last_user_msg = user_messages[-1].content
    extracted = extract_knowledge(str(last_user_msg), llm)
    updates = update_knowledge_graph(state, extracted)

    # Store feedback in action suggestions
    if state.user_profile.suggested_actions:
        state.user_profile.suggested_actions[-1].user_feedback = str(last_user_msg)

    # Build context-aware prompt
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add relevant conversation history (focus on plan and feedback)
    for msg in state.messages[-6:]:
        if isinstance(msg, HumanMessage):
            context_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            context_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add current state and meta-prompt
    state_context = f"""
Current state:
- User feedback on plan: {str(last_user_msg)}
- Obstacles mentioned: {extracted.get('obstacles_mentioned', [])}
- Emotional tone: {extracted.get('emotional_tone', 'neutral')}

{STAGE_7_META_PROMPT.format(feedback=str(last_user_msg))}
"""
    context_messages.append({"role": "user", "content": state_context})

    # Generate refinement question
    response = llm.invoke(context_messages)
    question = response.content if isinstance(response.content, str) else str(response.content)

    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 7
    updates["stage_7_complete"] = True
    updates["user_profile"] = state.user_profile

    return updates


# Stage 8: Summary Generation
def summary_generation(state: MetaAgentState) -> dict:
    """Generate final summary of values and goals."""
    summary = create_final_summary(state)

    return {
        "messages": [AIMessage(content=summary)],
        "final_summary": summary,
        "stage": 8,  # Final stage
    }


# Stage 9: Feedback Collection
def feedback_collection(state: MetaAgentState) -> dict:
    """Process final user feedback."""
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]

    if user_messages:
        feedback = str(user_messages[-1].content)

        # Convert user profile to dict for logging
        user_profile_dict = {
            "goals": {
                gid: {
                    "id": g.id,
                    "statement": g.statement,
                    "confirmed": g.confirmed,
                    "values": g.values,
                    "rationale": g.rationale,
                }
                for gid, g in state.user_profile.goals.items()
            },
            "values": {
                vname: {"name": v.name, "weight": v.weight, "confirmed": v.confirmed, "rationale": v.rationale}
                for vname, v in state.user_profile.values.items()
            },
        }

        # Log to Opik
        log_final_feedback(
            user_profile=user_profile_dict,
            final_summary=state.final_summary,
            final_feedback=feedback,
            conversation_id=None,  # Could extract from config/thread_id
        )

        return {
            "final_feedback": feedback,
            "messages": [AIMessage(content="Thank you for your feedback! Your insights have been recorded.")],
        }

    return {}


# === ROUTING LOGIC - Individual routing functions for each stage ===


def route_entry(
        state: MetaAgentState,
) -> Literal[
    "introduction",
    "rapport_building",
    "intent_clarification",
    "value_discovery",
    "value_tradeoffs",
    "value_confirmation",
    "action_planning",
    "plan_refinement",
    "summary_generation",
    "feedback_collection",
    "__end__",
]:
    """Entry router - determines which stage to start based on current state."""
    # If stage is 0 or not set, start with introduction
    if state.stage == 0 or state.stage == 1:
        # Check if introduction has been shown
        ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]
        if not ai_messages:
            return "introduction"

    # Check if we need to collect feedback after summary
    if state.stage == 8:
        if state.final_summary and not state.final_feedback:
            user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
            ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]
            if len(user_messages) > len(ai_messages):
                return "feedback_collection"
            return "__end__"

    # Route based on current stage
    if state.stage == 1 and state.stage_1_complete:
        return "intent_clarification"
    elif state.stage == 1:
        return "rapport_building"
    elif state.stage == 2 and state.stage_2_complete:
        return "value_discovery"
    elif state.stage == 2:
        return "intent_clarification"
    elif state.stage == 3 and state.stage_3_complete:
        return "value_tradeoffs"
    elif state.stage == 3:
        return "value_discovery"
    elif state.stage == 4 and state.stage_4_complete:
        return "value_confirmation"
    elif state.stage == 4:
        return "value_tradeoffs"
    elif state.stage == 5 and state.stage_5_complete:
        return "action_planning"
    elif state.stage == 5:
        return "value_confirmation"
    elif state.stage == 6 and state.stage_6_complete:
        return "plan_refinement"
    elif state.stage == 6:
        return "action_planning"
    elif state.stage == 7 and state.stage_7_complete:
        return "summary_generation"
    elif state.stage == 7:
        return "plan_refinement"

    # Default to introduction for new conversations
    return "introduction"


def route_from_introduction(state: MetaAgentState) -> Literal["rapport_building", "__end__"]:
    """Route from introduction stage."""
    # Wait for user to respond to introduction
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        return "rapport_building"
    return "__end__"


def route_from_rapport_building(state: MetaAgentState) -> Literal["intent_clarification", "__end__"]:
    """Route from rapport building stage."""
    # Wait for user response, then move to intent clarification
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to rapport building
    if len(user_messages) > len(ai_messages):
        return "intent_clarification"
    return "__end__"


def route_from_intent_clarification(state: MetaAgentState) -> Literal["value_discovery", "__end__"]:
    """Route from intent clarification stage."""
    # Wait for user response, then move to value discovery
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to clarification
    if len(user_messages) > len(ai_messages):
        return "value_discovery"
    return "__end__"


def route_from_value_discovery(state: MetaAgentState) -> Literal["value_discovery", "value_tradeoffs", "__end__"]:
    """Route from value discovery stage."""
    # Wait for user response
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    if len(user_messages) <= len(ai_messages):
        return "__end__"

    # Stay in value discovery until target depth is reached
    if state.value_depth < state.target_value_depth:
        return "value_discovery"

    # Move to trade-offs when depth is reached
    return "value_tradeoffs"


def route_from_value_tradeoffs(state: MetaAgentState) -> Literal["value_confirmation", "__end__"]:
    """Route from value trade-offs stage."""
    # Wait for user response, then move to confirmation
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to trade-off question
    if len(user_messages) > len(ai_messages):
        return "value_confirmation"
    return "__end__"


def route_from_value_confirmation(state: MetaAgentState) -> Literal["action_planning", "__end__"]:
    """Route from value confirmation stage."""
    # Wait for user response, then move to action planning
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to confirmation question
    if len(user_messages) > len(ai_messages):
        return "action_planning"
    return "__end__"


def route_from_action_planning(state: MetaAgentState) -> Literal["plan_refinement", "__end__"]:
    """Route from action planning stage."""
    # Wait for user response, then move to plan refinement
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to action plan
    if len(user_messages) > len(ai_messages):
        return "plan_refinement"
    return "__end__"


def route_from_plan_refinement(state: MetaAgentState) -> Literal["summary_generation", "__end__"]:
    """Route from plan refinement stage."""
    # Wait for user response, then move to summary
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has responded to refinement question
    if len(user_messages) > len(ai_messages):
        return "summary_generation"
    return "__end__"


def route_from_summary_generation(state: MetaAgentState) -> Literal["feedback_collection", "__end__"]:
    """Route from summary generation stage."""
    # Wait for user feedback on summary
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]

    # If user has provided feedback on summary
    if len(user_messages) > len(ai_messages):
        return "feedback_collection"
    return "__end__"


# opik.configure(use_local=True)
opik_tracer = OpikTracer(
    project_name="vd-meta-agent-v2",
)

"""Create the meta-prompting agent graph."""
# Define the graph
workflow = StateGraph(MetaAgentState)

# Add routing entry node
workflow.add_node("router", lambda state: state)

# Add nodes for each stage (with descriptive names)
workflow.add_node("introduction", introduction)
workflow.add_node("rapport_building", rapport_building)
workflow.add_node("intent_clarification", intent_clarification)
workflow.add_node("value_discovery", value_discovery)
workflow.add_node("value_tradeoffs", value_tradeoffs)
workflow.add_node("value_confirmation", value_confirmation)
workflow.add_node("action_planning", action_planning)
workflow.add_node("plan_refinement", plan_refinement)
workflow.add_node("summary_generation", summary_generation)
workflow.add_node("feedback_collection", feedback_collection)

# Set entry point to router
workflow.set_entry_point("router")

# Router determines which stage to enter
workflow.add_conditional_edges("router", route_entry)

# Add individual routing from each stage (semi-linear progression)
workflow.add_conditional_edges("introduction", route_from_introduction)
workflow.add_conditional_edges("rapport_building", route_from_rapport_building)
workflow.add_conditional_edges("intent_clarification", route_from_intent_clarification)
workflow.add_conditional_edges("value_discovery", route_from_value_discovery)
workflow.add_conditional_edges("value_tradeoffs", route_from_value_tradeoffs)
workflow.add_conditional_edges("value_confirmation", route_from_value_confirmation)
workflow.add_conditional_edges("action_planning", route_from_action_planning)
workflow.add_conditional_edges("plan_refinement", route_from_plan_refinement)
workflow.add_conditional_edges("summary_generation", route_from_summary_generation)

# Feedback collection leads to end
workflow.add_edge("feedback_collection", END)

# Compile the graph with opik tracing
graph = workflow.compile().with_config({"callbacks": [opik_tracer]})
