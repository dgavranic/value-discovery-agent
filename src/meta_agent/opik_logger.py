"""Opik integration for logging feedback and tracking conversations."""

import os
from typing import Optional

try:
    import opik
    from opik import track
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    # Create dummy decorator if Opik not available
    def track(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def log_final_feedback(
    user_profile: dict,
    final_summary: str,
    final_feedback: str,
    conversation_id: Optional[str] = None
) -> None:
    """Log the final feedback and discovered values/goals to Opik.
    
    Args:
        user_profile: The user's knowledge profile as a dictionary
        final_summary: The generated summary shown to the user
        final_feedback: The user's feedback on the summary
        conversation_id: Optional conversation/session ID
    """
    if not OPIK_AVAILABLE:
        print("⚠️  Opik not available. Feedback not logged.")
        return
    
    try:
        # Initialize Opik client
        client = opik.Opik()
        
        # Create a dataset for value discovery sessions if it doesn't exist
        dataset_name = "value-discovery-feedback"
        
        # Prepare metadata
        metadata = {
            "conversation_id": conversation_id or "unknown",
            "num_goals": len(user_profile.get("goals", {})),
            "num_values": len(user_profile.get("values", {})),
            "values_confirmed": any(
                v.get("confirmed", False) 
                for v in user_profile.get("values", {}).values()
            ),
        }
        
        # Extract top values
        values = user_profile.get("values", {})
        sorted_values = sorted(
            values.items(),
            key=lambda x: x[1].get("weight", 0),
            reverse=True
        )
        top_values = [
            {
                "name": name,
                "weight": val.get("weight", 0),
                "confirmed": val.get("confirmed", False)
            }
            for name, val in sorted_values[:5]
        ]
        
        # Extract goals
        goals = [
            {
                "id": goal.get("id"),
                "statement": goal.get("statement"),
                "confirmed": goal.get("confirmed", False)
            }
            for goal in user_profile.get("goals", {}).values()
        ]
        
        # Log as a trace
        trace = client.trace(
            name="Value Discovery Session",
            input={
                "session_type": "value_discovery",
                "conversation_id": conversation_id
            },
            output={
                "summary": final_summary,
                "feedback": final_feedback,
                "discovered_values": top_values,
                "discovered_goals": goals
            },
            metadata=metadata,
            tags=["value-discovery", "meta-prompting", "feedback"]
        )
        
        print(f"✅ Feedback logged to Opik (trace_id: {trace.id})")
        
        # Also log to dataset for analysis
        try:
            dataset = client.get_or_create_dataset(name=dataset_name)
            dataset.insert([{
                "input": {"conversation_id": conversation_id},
                "output": {
                    "summary": final_summary,
                    "feedback": final_feedback,
                    "values": top_values,
                    "goals": goals
                },
                "metadata": metadata
            }])
            print(f"✅ Added to dataset: {dataset_name}")
        except Exception as e:
            print(f"⚠️  Could not add to dataset: {e}")
            
    except Exception as e:
        print(f"❌ Error logging to Opik: {e}")
        print("Feedback data:")
        print(f"Summary: {final_summary[:100]}...")
        print(f"Feedback: {final_feedback}")


@track(name="Stage Completion")
def log_stage_completion(
    stage_number: int,
    stage_name: str,
    question_generated: str,
    user_response: str,
    extracted_knowledge: dict,
    conversation_id: Optional[str] = None
) -> None:
    """Log individual stage completions for debugging and analysis.
    
    Args:
        stage_number: The stage number (1-7)
        stage_name: Name of the stage
        question_generated: The question the AI generated
        user_response: The user's response
        extracted_knowledge: Extracted structured knowledge
        conversation_id: Optional conversation/session ID
    """
    if not OPIK_AVAILABLE:
        return
    
    try:
        client = opik.Opik()
        
        client.trace(
            name=f"Stage {stage_number}: {stage_name}",
            input={
                "stage": stage_number,
                "question": question_generated,
                "conversation_id": conversation_id
            },
            output={
                "user_response": user_response,
                "extracted": extracted_knowledge
            },
            metadata={
                "stage_name": stage_name,
                "conversation_id": conversation_id
            },
            tags=["stage-completion", f"stage-{stage_number}"]
        )
    except Exception as e:
        print(f"⚠️  Could not log stage completion: {e}")


def log_value_discovery(
    values: dict,
    goals: dict,
    conversation_id: Optional[str] = None
) -> None:
    """Log discovered values and goals throughout the conversation.
    
    Args:
        values: Dictionary of discovered values
        goals: Dictionary of discovered goals
        conversation_id: Optional conversation/session ID
    """
    if not OPIK_AVAILABLE:
        return
    
    try:
        client = opik.Opik()
        
        # Format for logging
        value_data = [
            {
                "name": name,
                "weight": val.get("weight", 0),
                "confirmed": val.get("confirmed", False),
                "rationale_count": len(val.get("rationale", []))
            }
            for name, val in values.items()
        ]
        
        goal_data = [
            {
                "id": goal.get("id"),
                "statement": goal.get("statement"),
                "confirmed": goal.get("confirmed", False),
                "linked_values": goal.get("values", [])
            }
            for goal in goals.values()
        ]
        
        client.trace(
            name="Knowledge Graph Update",
            input={
                "conversation_id": conversation_id,
                "update_type": "knowledge_graph"
            },
            output={
                "values": value_data,
                "goals": goal_data
            },
            metadata={
                "num_values": len(values),
                "num_goals": len(goals),
                "conversation_id": conversation_id
            },
            tags=["knowledge-graph", "values", "goals"]
        )
    except Exception as e:
        print(f"⚠️  Could not log value discovery: {e}")
