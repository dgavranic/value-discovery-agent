"""Main graph implementation for meta agent v3."""

from langgraph.graph import StateGraph, END
from opik.integrations.langchain import OpikTracer

from meta_agent_v3.state import MetaAgentState, InputState
from meta_agent_v3.nodes import (
    introduction_node,
    rapport_building_node,
    assess_rapport_completion,
    value_discovery_node,
    assess_value_discovery_completion,
    value_ranking_node,
    assess_value_ranking_completion,
    action_planning_node,
    assess_action_planning_completion,
    summary_feedback_node,
)
from meta_agent_v3.router import (
    route_after_introduction,
    route_after_rapport,
    route_after_value_discovery,
    route_after_value_ranking,
    route_after_action_planning,
    route_after_summary,
)
from meta_agent_v3.opik_logger import log_final_feedback, log_error


# Initialize Opik tracer for monitoring
opik_tracer = OpikTracer(
    project_name="vd-meta-agent-v3",
)


# === GRAPH CONSTRUCTION ===

def create_graph():
    """Create and configure the value discovery agent graph.

    Returns:
        Compiled StateGraph
    """
    # Initialize workflow
    workflow = StateGraph(MetaAgentState)

    # Add nodes for each stage
    workflow.add_node("introduction", introduction_node)
    workflow.add_node("rapport_building", rapport_building_node)
    workflow.add_node("assess_rapport", assess_rapport_completion)
    workflow.add_node("value_discovery", value_discovery_node)
    workflow.add_node("assess_value_discovery", assess_value_discovery_completion)
    workflow.add_node("value_ranking", value_ranking_node)
    workflow.add_node("assess_value_ranking", assess_value_ranking_completion)
    workflow.add_node("action_planning", action_planning_node)
    workflow.add_node("assess_action_planning", assess_action_planning_completion)
    workflow.add_node("summary_feedback", summary_feedback_node)
    workflow.add_node("finalize", _finalize_session)

    # Set entry point
    workflow.set_entry_point("introduction")

    # === STAGE 0: INTRODUCTION ===
    workflow.add_conditional_edges(
        "introduction",
        route_after_introduction,
        {
            "rapport_building": "rapport_building",
            END: END,
        }
    )

    # === STAGE 1: RAPPORT BUILDING ===
    # After rapport building, assess completion
    workflow.add_edge("rapport_building", "assess_rapport")

    # Route based on assessment
    workflow.add_conditional_edges(
        "assess_rapport",
        route_after_rapport,
        {
            "rapport_building": "rapport_building",
            "value_discovery": "value_discovery"
        }
    )

    # === STAGE 2: VALUE DISCOVERY ===
    workflow.add_edge("value_discovery", "assess_value_discovery")

    workflow.add_conditional_edges(
        "assess_value_discovery",
        route_after_value_discovery,
        {
            "value_discovery": "value_discovery",
            "value_ranking": "value_ranking"
        }
    )

    # === STAGE 3: VALUE RANKING ===
    workflow.add_edge("value_ranking", "assess_value_ranking")

    workflow.add_conditional_edges(
        "assess_value_ranking",
        route_after_value_ranking,
        {
            "value_ranking": "value_ranking",
            "action_planning": "action_planning"
        }
    )

    # === STAGE 4: ACTION PLANNING ===
    workflow.add_edge("action_planning", "assess_action_planning")

    workflow.add_conditional_edges(
        "assess_action_planning",
        route_after_action_planning,
        {
            "action_planning": "action_planning",
            "summary_feedback": "summary_feedback"
        }
    )

    # === STAGE 5: SUMMARY & FEEDBACK ===
    workflow.add_conditional_edges(
        "summary_feedback",
        route_after_summary,
        {
            "summary_feedback": "summary_feedback",
            "__end__": "finalize"
        }
    )

    # === FINALIZATION ===
    workflow.add_edge("finalize", END)

    # Compile and return
    return workflow.compile()


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


# Create the graph instance
graph = create_graph()
