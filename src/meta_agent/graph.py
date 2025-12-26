"""Main graph implementation for meta-prompting agent."""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from meta_agent.state import MetaAgentState, InputState, ActionSuggestion
from meta_agent.prompts import (
    SYSTEM_PROMPT,
    STAGE_1_META_PROMPT,
    STAGE_2_META_PROMPT,
    STAGE_3_META_PROMPT,
    STAGE_4_META_PROMPT,
    STAGE_5_META_PROMPT,
    STAGE_6_META_PROMPT,
    STAGE_7_META_PROMPT,
)
from meta_agent.knowledge import (
    extract_knowledge,
    update_knowledge_graph,
    identify_conflicting_values,
    format_values_summary,
    format_goals_summary,
    create_final_summary,
)
from meta_agent.opik_logger import log_final_feedback


def get_llm() -> ChatOpenAI:
    """Initialize the language model."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# === STAGE NODES ===

def stage_1_rapport(state: MetaAgentState) -> dict:
    """Stage 1: Build rapport and gather context."""
    llm = get_llm()
    
    # Generate question using meta-prompt
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": STAGE_1_META_PROMPT}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    return {
        "messages": [AIMessage(content=question)],
        "last_question": question,
        "stage": 1,
    }


def stage_2_reflect(state: MetaAgentState) -> dict:
    """Stage 2: Reflect back and clarify understanding."""
    llm = get_llm()
    
    # Get last user message
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="I didn't receive your response. Could you share what brought you here?")]}
    
    last_user_msg = user_messages[-1].content
    
    # Extract knowledge from user response
    extracted = extract_knowledge(str(last_user_msg), llm)
    updates = update_knowledge_graph(state, extracted)
    
    # Update intent context
    if extracted.get("goals_mentioned"):
        updates["user_profile"].intent_context.goal_statement = extracted["goals_mentioned"][0]
    if extracted.get("key_phrases"):
        updates["user_profile"].intent_context.desired_outcome = " ".join(extracted["key_phrases"][:2])
    
    # Generate reflection and clarification
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": STAGE_2_META_PROMPT}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 2
    
    return updates


def stage_3_laddering(state: MetaAgentState) -> dict:
    """Stage 3: Explore deeper values through 'why' questions."""
    llm = get_llm()
    
    # Get current goal
    goals = list(state.user_profile.goals.values())
    current_goal = goals[0].statement if goals else state.user_profile.intent_context.goal_statement
    
    if not current_goal:
        current_goal = "what you shared earlier"
    
    # Generate laddering question
    prompt = STAGE_3_META_PROMPT.format(
        goal=current_goal,
        value_depth=state.value_depth,
        target_depth=state.target_value_depth
    )
    
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    # Extract and update from previous user response
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_msg = user_messages[-1].content
        extracted = extract_knowledge(str(last_user_msg), llm)
        updates = update_knowledge_graph(state, extracted)
        
        # Check for confirmation in stage 2
        if state.stage == 2 and any(word in str(last_user_msg).lower() for word in ["yes", "correct", "right", "exactly", "accurate"]):
            updates["intent_confirmed"] = True
            # Mark first goal as confirmed
            if state.user_profile.goals:
                first_goal = list(state.user_profile.goals.values())[0]
                first_goal.confirmed = True
    else:
        updates = {}
    
    # Increment value depth
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 3
    updates["value_depth"] = state.value_depth + 1
    
    return updates


def stage_4_tradeoffs(state: MetaAgentState) -> dict:
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
    conflict = identify_conflicting_values(state)
    
    if not conflict:
        # Skip to next stage if no conflicts found
        updates["stage"] = 5
        updates["stage_4_complete"] = True
        return updates
    
    value1, value2 = conflict
    values_text = f"{value1} and {value2}"
    
    # Generate trade-off question
    prompt = STAGE_4_META_PROMPT.format(values=values_text)
    
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 4
    
    return updates


def stage_5_reflection(state: MetaAgentState) -> dict:
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
    values_summary = format_values_summary(state)
    
    # Generate reflection question
    prompt = STAGE_5_META_PROMPT.format(values_summary=values_summary)
    
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 5
    
    return updates


def stage_6_planning(state: MetaAgentState) -> dict:
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
    goals_text = format_goals_summary(state)
    values_text = format_values_summary(state)
    
    # Generate plan with meta-prompt
    prompt = STAGE_6_META_PROMPT.format(
        goals=goals_text,
        values=values_text
    )
    
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    plan = response.content if isinstance(response.content, str) else str(response.content)
    
    # Parse action suggestions (simplified)
    # In production, use structured output or better parsing
    top_values = sorted(
        state.user_profile.values.items(),
        key=lambda x: x[1].weight,
        reverse=True
    )[:3]
    
    action = ActionSuggestion(
        description="Action plan generated",
        linked_values=[v[0] for v in top_values],
        fit_score=0.8
    )
    state.user_profile.suggested_actions.append(action)
    
    updates["messages"] = [AIMessage(content=plan)]
    updates["last_question"] = plan
    updates["stage"] = 6
    updates["plan_generated"] = True
    updates["user_profile"] = state.user_profile
    
    return updates


def stage_7_adaptation(state: MetaAgentState) -> dict:
    """Stage 7: Refine plan based on feedback."""
    llm = get_llm()
    
    # Extract knowledge from feedback
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"stage": 7}
    
    last_user_msg = user_messages[-1].content
    extracted = extract_knowledge(str(last_user_msg), llm)
    updates = update_knowledge_graph(state, extracted)
    
    # Store feedback in action suggestions
    if state.user_profile.suggested_actions:
        state.user_profile.suggested_actions[-1].user_feedback = str(last_user_msg)
    
    # Generate refinement question
    prompt = STAGE_7_META_PROMPT.format(feedback=str(last_user_msg))
    
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    
    question = response.content if isinstance(response.content, str) else str(response.content)
    
    updates["messages"] = [AIMessage(content=question)]
    updates["last_question"] = question
    updates["stage"] = 7
    updates["stage_7_complete"] = True
    updates["user_profile"] = state.user_profile
    
    return updates


def generate_summary(state: MetaAgentState) -> dict:
    """Generate final summary of values and goals."""
    summary = create_final_summary(state)
    
    return {
        "messages": [AIMessage(content=summary)],
        "final_summary": summary,
        "stage": 8,  # Final stage
    }


def process_feedback(state: MetaAgentState) -> dict:
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
                    "rationale": g.rationale
                }
                for gid, g in state.user_profile.goals.items()
            },
            "values": {
                vname: {
                    "name": v.name,
                    "weight": v.weight,
                    "confirmed": v.confirmed,
                    "rationale": v.rationale
                }
                for vname, v in state.user_profile.values.items()
            }
        }
        
        # Log to Opik
        log_final_feedback(
            user_profile=user_profile_dict,
            final_summary=state.final_summary,
            final_feedback=feedback,
            conversation_id=None  # Could extract from config/thread_id
        )
        
        return {
            "final_feedback": feedback,
            "messages": [AIMessage(content="Thank you for your feedback! Your insights have been recorded.")],
        }
    
    return {}


# === ROUTING LOGIC ===

def route_stage(state: MetaAgentState) -> Literal["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6", "stage_7", "summary", "feedback", "__end__"]:
    """Route to appropriate stage based on state conditions."""
    
    # Check if we're collecting final feedback
    if state.stage == 8 and not state.final_feedback:
        # Wait for user feedback on summary
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]
        
        # If user has responded to summary
        if len(user_messages) > len(ai_messages):
            return "feedback"
        return "__end__"
    
    if state.stage == 8 and state.final_feedback:
        return "__end__"
    
    # Initial stage
    if state.stage == 1 and not state.stage_1_complete:
        # Check if user has responded
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        if user_messages:
            return "stage_2"
        return "__end__"
    
    # Stage routing logic based on completion criteria
    if state.trust_level < 0.5 or state.stage == 1:
        return "stage_1"
    
    if not state.intent_confirmed and state.stage <= 2:
        return "stage_2"
    
    if state.value_depth < state.target_value_depth and state.stage <= 3:
        return "stage_3"
    
    if len(state.user_profile.values) >= 2 and not state.stage_4_complete and state.stage <= 4:
        conflict = identify_conflicting_values(state)
        if conflict:
            return "stage_4"
    
    if not state.values_confirmed and state.stage <= 5:
        return "stage_5"
    
    if not state.plan_generated and state.stage <= 6:
        return "stage_6"
    
    if not state.stage_7_complete and state.stage == 7:
        return "stage_7"
    
    # All stages complete, generate summary
    return "summary"


# === BUILD GRAPH ===

def create_graph():
    """Create the meta-prompting agent graph."""
    
    # Define the graph
    workflow = StateGraph(MetaAgentState, input=InputState)
    
    # Add nodes for each stage
    workflow.add_node("stage_1", stage_1_rapport)
    workflow.add_node("stage_2", stage_2_reflect)
    workflow.add_node("stage_3", stage_3_laddering)
    workflow.add_node("stage_4", stage_4_tradeoffs)
    workflow.add_node("stage_5", stage_5_reflection)
    workflow.add_node("stage_6", stage_6_planning)
    workflow.add_node("stage_7", stage_7_adaptation)
    workflow.add_node("summary", generate_summary)
    workflow.add_node("feedback", process_feedback)
    
    # Set entry point
    workflow.set_entry_point("stage_1")
    
    # Add conditional routing from each stage
    workflow.add_conditional_edges("stage_1", route_stage)
    workflow.add_conditional_edges("stage_2", route_stage)
    workflow.add_conditional_edges("stage_3", route_stage)
    workflow.add_conditional_edges("stage_4", route_stage)
    workflow.add_conditional_edges("stage_5", route_stage)
    workflow.add_conditional_edges("stage_6", route_stage)
    workflow.add_conditional_edges("stage_7", route_stage)
    workflow.add_conditional_edges("summary", route_stage)
    
    # Feedback leads to end
    workflow.add_edge("feedback", END)
    
    # Compile with memory
    graph = workflow.compile()
    
    return graph


# Create the graph instance
graph = create_graph()
