# Meta Agent V4 - Interrupt-Based Value Discovery Agent

## Overview

Meta Agent V4 is a simplified, interrupt-based implementation of the value discovery conversational agent. It uses LangGraph's interrupt mechanism to create a natural, user-controlled conversation flow.

## Key Improvements Over V3

### 1. **Simplified Architecture**
- **5 stages** instead of 6 (merged Value Ranking into Value Discovery)
- **No separate assessment nodes** - validation integrated into routers
- **Cleaner flow** - each stage node handles generation, routers handle validation

### 2. **Interrupt-Based Flow**
- Uses `interrupt_before` for each stage
- User controls conversation pace naturally
- More flexible progression (can loop within stages)
- Feels more conversational, less robotic

### 3. **DRY & SOLID Principles**
- Single responsibility: each node does one thing
- No code duplication between assessment and generation
- Clean separation of concerns (knowledge, nodes, routing, prompts)
- Easy to test and maintain

### 4. **LLM-Driven Validation**
- Every stage advancement decision made by LLM
- Evaluates completion based on actual conversation quality
- Fallback logic ensures finite completion time
- Forced advancement after maximum turns to prevent infinite loops

## Architecture

```
User Message
    ↓
Preprocessor (extracts knowledge)
    ↓
Route to Stage
    ↓
Stage Node (generates response)
    ↓
Validation Router (LLM assesses completion)
    ↓
Loop (same stage) OR Advance (next stage)
```

### Stages

1. **Introduction** (Stage 0)
   - Welcome message, explain process
   - Move to Rapport Building

2. **Rapport Building** (Stage 1)
   - Extract specific problem/goal details
   - Build trust and psychological safety
   - Gather context: obstacles, emotions, specifics
   - **Exit criteria**: Clear goal + rich context + 5-8 turns

3. **Value Discovery** (Stage 2)
   - Explore WHY goals matter (3-4 levels deep)
   - Use Clean Language & Motivational Interviewing
   - Help prioritize and rank values naturally
   - **Exit criteria**: 3-5 distinct values + rationale + 6-10 turns

4. **Action Planning** (Stage 3)
   - Generate value-aligned action plans
   - Use A/B testing: present options, gather preferences
   - Refine based on feedback
   - **Exit criteria**: 3-5 concrete actions + user buy-in + 4-7 turns

5. **Summary & Feedback** (Stage 4)
   - Present comprehensive summary
   - Celebrate discoveries
   - Collect feedback for improvement
   - **Exit criteria**: Feedback received + 1-2 turns

## Files

- **`state.py`** - State definitions (simplified from v3)
- **`knowledge.py`** - Knowledge extraction & management
- **`prompts.py`** - All prompts including validation prompts
- **`nodes.py`** - Stage node implementations
- **`router.py`** - Routing logic with LLM-based validation
- **`graph.py`** - Graph construction with interrupts
- **`utils.py`** - Helper functions
- **`opik_logger.py`** - Logging for A/B testing

## Key Features

### 1. Knowledge Extraction
Every user message is processed to extract:
- Goals mentioned
- Values (what matters to them)
- Emotional tone
- Obstacles
- Context details
- Engagement level

### 2. Dynamic Validation
Each stage has LLM-based validation that assesses:
- Have completion criteria been met?
- Is the user ready to advance?
- What's still missing?
- Should we force advancement? (prevent infinite loops)

### 3. Interrupt-Based Flow
Interrupts happen BEFORE each stage:
- Allows user to respond at their own pace
- Natural conversation flow
- No forced progression
- User can elaborate as much as they want

### 4. Metrics Tracking
For A/B testing and analysis:
- Tokens used per stage
- Turn count per stage
- Duration per stage
- Total session metrics
- Model configuration (for testing different models)

## Usage

```python
from meta_agent_v4 import graph

# Initialize with first message
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(
    {"messages": [("user", "I want to figure out my career direction")]},
    config
)

# Continue conversation
result = graph.invoke(
    {"messages": [("user", "I'm feeling stuck in my current role")]},
    config
)
```

## Validation Logic

Each stage has:
1. **LLM Assessment** - Primary validation using structured prompt
2. **Fallback Logic** - If LLM fails, use rule-based fallback
3. **Force Advancement** - After max turns, must advance

Example validation flow:
```python
def validate_stage(state):
    # Try LLM assessment
    assessment = llm.invoke(validation_prompt)
    
    if assessment.should_advance:
        finalize_stage_metrics(state, stage_name)
        return "next_stage"
    
    # Check force conditions
    if turn_count >= MAX_TURNS:
        finalize_stage_metrics(state, stage_name)
        return "next_stage"
    
    # Loop
    return "current_stage"
```

## Differences from V3

| Feature | V3 | V4 |
|---------|----|----|
| Stages | 6 (separate value ranking) | 5 (ranking merged) |
| Validation | Separate assessment nodes | Integrated in routers |
| Flow | Linear with conditional edges | Interrupt-based loops |
| Progression | Automatic after assessment | User-controlled via interrupts |
| Complexity | Higher (more nodes) | Lower (cleaner flow) |
| Code Reuse | Some duplication | DRY principles |

## Testing Different Models

Configure via state:
```python
state = MetaAgentState(
    model_name="gpt-4o",  # or "gpt-4o-mini", "claude-3-5-sonnet"
    temperature=0.7
)
```

Metrics logged to Opik for comparison:
- Which model completes faster?
- Which model discovers more values?
- Which model gets better user feedback?
- Which model uses fewer tokens?

## Future Enhancements

1. **Human-in-the-loop checkpoints** - Let user review before advancing
2. **Dynamic question generation** - More adaptive to user's style
3. **Value conflict resolution** - Explicit handling of conflicting values
4. **Multi-language support** - i18n for prompts
5. **Visualization** - Show value map as conversation progresses

