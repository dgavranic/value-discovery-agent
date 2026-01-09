"""Main graph implementation with interrupt-based flow for natural conversation."""

from langgraph.graph import StateGraph, END

from meta_agent_v4.state import MetaAgentState
from meta_agent_v4.nodes import (
    preprocessor_node,
    introduction_node,
    rapport_building_node,
    value_discovery_node,
    action_planning_node,
    summary_feedback_node,
)
from meta_agent_v4.router import (
    route_from_preprocessor,
    route_after_introduction,
    validate_rapport_building,
    validate_value_discovery,
    validate_action_planning,
    validate_summary_feedback,
)
from meta_agent_v4.opik_logger import log_final_feedback, log_error


def _finalize_session(state: MetaAgentState) -> dict:
    """Finalize the session and log to Opik.

    Args:
        state: Final MetaAgentState

    Returns:
        Empty dict (no state changes)
    """
    try:
        log_final_feedback(state)
    except Exception as e:
        print(f"⚠️  Error during finalization: {e}")
        log_error(str(e), state, "finalization")

    return {}


def create_graph():
    """Create and configure the value discovery agent graph with interrupts.

    Flow:
    - Preprocessor extracts knowledge from every user message
    - Each stage node generates responses
    - Validation routers decide if stage is complete (loop or advance)
    - Interrupts before each stage allow natural conversation flow

    Returns:
        Compiled StateGraph with interrupts enabled
    """
    workflow = StateGraph(MetaAgentState)

    # Add nodes
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("introduction", introduction_node)
    workflow.add_node("rapport_building", rapport_building_node)
    workflow.add_node("value_discovery", value_discovery_node)
    workflow.add_node("action_planning", action_planning_node)
    workflow.add_node("summary_feedback", summary_feedback_node)
    workflow.add_node("finalize", _finalize_session)

    # Set entry point
    workflow.set_entry_point("preprocessor")

    # Preprocessor routes to appropriate stage
    workflow.add_conditional_edges(
        "preprocessor",
        route_from_preprocessor,
        {
            "introduction": "introduction",
            "rapport_building": "rapport_building",
            "value_discovery": "value_discovery",
            "action_planning": "action_planning",
            "summary_feedback": "summary_feedback",
            "__end__": END
        }
    )

    # === STAGE 0: INTRODUCTION ===
    # Introduction sends message and advances to rapport building
    workflow.add_conditional_edges(
        "introduction",
        route_after_introduction,
        {
            "rapport_building": "rapport_building",
            "__end__": END
        }
    )

    # === STAGE 1: RAPPORT BUILDING ===
    # After generating response, validate and decide to loop or advance
    workflow.add_conditional_edges(
        "rapport_building",
        validate_rapport_building,
        {
            "rapport_building": "rapport_building",  # Loop: stay in stage
            "value_discovery": "value_discovery"      # Advance: move to next
        }
    )

    # === STAGE 2: VALUE DISCOVERY ===
    # After generating response, validate and decide to loop or advance
    workflow.add_conditional_edges(
        "value_discovery",
        validate_value_discovery,
        {
            "value_discovery": "value_discovery",     # Loop: stay in stage
            "action_planning": "action_planning"      # Advance: move to next
        }
    )

    # === STAGE 3: ACTION PLANNING ===
    # After generating response, validate and decide to loop or advance
    workflow.add_conditional_edges(
        "action_planning",
        validate_action_planning,
        {
            "action_planning": "action_planning",     # Loop: stay in stage
            "summary_feedback": "summary_feedback"    # Advance: move to next
        }
    )

    # === STAGE 4: SUMMARY & FEEDBACK ===
    # After generating response, validate and decide to loop or end
    workflow.add_conditional_edges(
        "summary_feedback",
        validate_summary_feedback,
        {
            "summary_feedback": "summary_feedback",   # Loop: collect feedback
            "__end__": "finalize"                     # End: session complete
        }
    )

    # === FINALIZATION ===
    workflow.add_edge("finalize", END)

    # Compile with interrupts before each stage
    # This allows user to control pace and creates natural conversation flow
    return workflow.compile(
        interrupt_before=[
            "rapport_building",
            "value_discovery",
            "action_planning",
            "summary_feedback"
        ]
    )


# Create the graph instance
graph = create_graph()

