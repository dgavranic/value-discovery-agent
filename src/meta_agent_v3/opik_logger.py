"""Opik integration for logging feedback and tracking conversations."""

from datetime import datetime, timezone

try:
    import opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    opik = None  # Define for type checking


def log_final_feedback(state) -> None:
    """Log the final session data to Opik for A/B testing and analysis.

    Args:
        state: MetaAgentState with complete session data
    """
    if not OPIK_AVAILABLE:
        print("⚠️  Opik not available. Session data not logged.")
        return

    try:
        # Initialize Opik client
        client = opik.Opik()

        # Prepare session metrics
        total_tokens = sum(m.total_tokens for m in state.stage_metrics)
        session_duration = _calculate_session_duration(state)

        # Stage-level metrics
        stage_breakdown = []
        for metrics in state.stage_metrics:
            stage_breakdown.append({
                "stage_name": metrics.stage_name,
                "turn_count": metrics.turn_count,
                "total_tokens": metrics.total_tokens,
                "duration_seconds": _calculate_duration(metrics.start_time, metrics.end_time)
            })

        # Extract discovered values
        values = [
            {
                "name": v.name,
                "weight": v.weight,
                "confirmed": v.confirmed,
                "rationale_count": len(v.rationale)
            }
            for v in state.user_profile.values.values()
        ]
        sorted_values = sorted(values, key=lambda x: x["weight"], reverse=True)

        # Extract goals
        goals = [
            {
                "id": g.id,
                "statement": g.statement,
                "confirmed": g.confirmed,
                "obstacles_count": len(g.obstacles),
                "linked_values": g.values
            }
            for g in state.user_profile.goals.values()
        ]

        # Extract actions
        actions = [
            {
                "description": a.description,
                "linked_values": a.linked_values,
                "fit_score": a.fit_score,
                "user_feedback": a.user_feedback
            }
            for a in state.user_profile.suggested_actions
        ]

        # Metadata for A/B testing
        metadata = {
            "model_name": state.model_name,
            "temperature": state.temperature,
            "total_turns": state.total_turns,
            "total_tokens": total_tokens,
            "session_duration_seconds": session_duration,
            "num_goals": len(goals),
            "num_values": len(values),
            "num_actions": len(actions),
            "intent_confirmed": state.intent_confirmed,
            "values_confirmed": state.values_confirmed,
            "plan_generated": state.plan_generated,
            "stages_completed": len(state.stage_metrics),
            "session_start": state.session_start,
        }

        # Create trace
        trace = client.trace(
            name="Value Discovery Session v3",
            input={
                "session_type": "value_discovery_v3",
                "model": state.model_name,
                "temperature": state.temperature
            },
            output={
                "final_summary": state.final_summary,
                "final_feedback": state.final_feedback,
                "discovered_values": sorted_values[:7],  # Top 7 values
                "discovered_goals": goals,
                "suggested_actions": actions
            },
            metadata=metadata,
            tags=["value-discovery", "meta-agent-v3", "production"]
        )

        # Log stage-level spans
        for stage_data in stage_breakdown:
            client.span(
                trace_id=trace.id,
                name=f"Stage: {stage_data['stage_name']}",
                input={"stage": stage_data["stage_name"]},
                output={
                    "turn_count": stage_data["turn_count"],
                    "tokens_used": stage_data["total_tokens"]
                },
                metadata={
                    "duration_seconds": stage_data["duration_seconds"],
                    "stage_name": stage_data["stage_name"]
                },
                tags=["stage-metrics"]
            )

        print(f"✅ Session logged to Opik (trace_id: {trace.id})")
        print(f"   Total turns: {state.total_turns}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Session duration: {session_duration:.1f}s")
        print(f"   Values discovered: {len(values)}")
        print(f"   Goals identified: {len(goals)}")
        print(f"   Actions suggested: {len(actions)}")

    except Exception as e:
        print(f"⚠️  Error logging to Opik: {e}")


def _calculate_session_duration(state) -> float:
    """Calculate total session duration in seconds.

    Args:
        state: MetaAgentState

    Returns:
        Duration in seconds
    """
    if not state.stage_metrics:
        return 0.0

    try:
        start_time = datetime.fromisoformat(state.session_start)

        # Find the latest end time from stage metrics
        latest_end = None
        for metrics in state.stage_metrics:
            if metrics.end_time:
                end_time = datetime.fromisoformat(metrics.end_time)
                if latest_end is None or end_time > latest_end:
                    latest_end = end_time

        if latest_end:
            duration = (latest_end - start_time).total_seconds()
            return duration

        # If no end times, use current time
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        return duration

    except Exception:
        return 0.0


def _calculate_duration(start_time_str: str, end_time_str: str) -> float:
    """Calculate duration between two ISO timestamp strings.

    Args:
        start_time_str: Start time in ISO format
        end_time_str: End time in ISO format

    Returns:
        Duration in seconds
    """
    if not end_time_str:
        return 0.0

    try:
        start = datetime.fromisoformat(start_time_str)
        end = datetime.fromisoformat(end_time_str)
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def log_error(error_message: str, state, stage: str) -> None:
    """Log an error that occurred during the session.

    Args:
        error_message: Description of the error
        state: Current MetaAgentState
        stage: Stage where error occurred
    """
    if not OPIK_AVAILABLE:
        print(f"⚠️  Error in {stage}: {error_message}")
        return

    try:
        client = opik.Opik()

        trace = client.trace(
            name=f"Error in Value Discovery - {stage}",
            input={
                "stage": stage,
                "model": state.model_name if state else "unknown"
            },
            output={
                "error": error_message
            },
            metadata={
                "total_turns": state.total_turns if state else 0,
                "current_stage": stage
            },
            tags=["error", "value-discovery-v3"]
        )

        print(f"❌ Error logged to Opik (trace_id: {trace.id})")

    except Exception as e:
        print(f"⚠️  Error logging error to Opik: {e}")

