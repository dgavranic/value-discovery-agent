"""Main graph implementation with interrupt-based flow for natural conversation."""

from langgraph.graph import StateGraph, END
from opik.integrations.langchain import OpikTracer

from meta_agent_v4.state import MetaAgentState
from meta_agent_v4.nodes import (
    preprocessor_node,
    introduction_node,
    rapport_building_node,
    value_discovery_node,
    action_planning_node,
    summary_feedback_node,
    reply_node,
    validate_rapport_node,
    validate_value_discovery_node,
    validate_action_planning_node,
    validate_summary_node,
)
from meta_agent_v4.router import (
    route_from_preprocessor,
    route_from_reply,
    route_after_validate_summary,
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
    - Each stage node generates responses and routes to REPLY node
    - REPLY node uses interrupt_before to wait for user input
    - After user responds, REPLY routes to appropriate validator
    - Validator decides if stage is complete (loop or advance)

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
    workflow.add_node("reply", reply_node)
    workflow.add_node("validate_rapport", validate_rapport_node)
    workflow.add_node("validate_value_discovery", validate_value_discovery_node)
    workflow.add_node("validate_action_planning", validate_action_planning_node)
    workflow.add_node("validate_summary", validate_summary_node)
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
    workflow.add_edge(
        "introduction",
        "__end__"
    )

    # === STAGE 1: RAPPORT BUILDING ===
    # After generating response, go to REPLY node
    workflow.add_edge("rapport_building", "reply")

    # === STAGE 2: VALUE DISCOVERY ===
    # After generating response, go to REPLY node
    workflow.add_edge("value_discovery", "reply")

    # === STAGE 3: ACTION PLANNING ===
    # After generating response, go to REPLY node
    workflow.add_edge("action_planning", "reply")

    # === STAGE 4: SUMMARY & FEEDBACK ===
    # After generating response, go to REPLY node
    workflow.add_edge("summary_feedback", "reply")

    # === REPLY NODE ===
    # REPLY routes to appropriate validator based on current stage
    workflow.add_conditional_edges(
        "reply",
        route_from_reply,
        {
            "validate_rapport": "validate_rapport",
            "validate_value_discovery": "validate_value_discovery",
            "validate_action_planning": "validate_action_planning",
            "validate_summary": "validate_summary",
            "__end__": END
        }
    )

    # === VALIDATOR NODES ===
    # Each validator decides to loop back to stage or advance
    workflow.add_edge(
        "validate_rapport",
        "preprocessor"
    )

    workflow.add_edge(
        "validate_value_discovery",
        "preprocessor"
    )

    workflow.add_edge(
        "validate_action_planning",
        "preprocessor"
    )

    workflow.add_conditional_edges(
        "validate_summary",
        route_after_validate_summary,
        {
            "summary_feedback": "summary_feedback",
            "finalize": "finalize"
        }
    )

    # === FINALIZATION ===
    workflow.add_edge("finalize", END)

    # Compile the graph with interrupt_before on the REPLY node
    return workflow.compile()


opik_tracer = OpikTracer(
    project_name="vd-meta-agent-v4",
)

# Create the graph instance
graph = create_graph().with_config({"callbacks": [opik_tracer]})
