# Meta-Prompting Value Discovery Agent

A sophisticated conversational agent that uses **meta-prompting** to discover user values, goals, and motivations through a structured 7-stage dialogue.

## 🎯 Overview

Instead of hard-coded questions, this agent uses **instructional templates** that tell the AI how to generate contextually appropriate questions based on the conversation state. It builds a knowledge graph of user values and goals through progressive exploration.

## 🏗️ Architecture

### State Management
- **UserProfile**: Stores goals, values, and conversation history
- **MetaAgentState**: Tracks conversation stage, metrics, and knowledge graph
- **Memory**: LangGraph MemorySaver maintains state across turns

### 7-Stage Conversation Flow

```
Stage 1: Rapport & Context
    ↓
Stage 2: Reflect & Clarify
    ↓
Stage 3: Laddering (Why exploration)
    ↓
Stage 4: Trade-off Probing
    ↓
Stage 5: Value Reflection
    ↓
Stage 6: Plan Generation
    ↓
Stage 7: Reflection & Adaptation
    ↓
Final Summary & Feedback Collection
```

### Knowledge Graph

The agent builds a structured representation:

```python
{
  "goals": {
    "g1": {
      "statement": "Start a small business",
      "confirmed": True,
      "values": ["autonomy", "financial_security"],
      "rationale": ["I want control over my time"]
    }
  },
  "values": {
    "autonomy": {
      "weight": 0.85,
      "confirmed": True,
      "rationale": ["freedom to choose", "self-direction"]
    }
  }
}
```

## 🚀 Usage

### With LangGraph CLI

```bash
# Start the server
langgraph dev

# The meta_agent will be available at:
# http://localhost:2024
```

### Programmatic Usage

```python
from meta_agent import graph, InputState

# Initialize state
state = InputState(messages=[])

# Run conversation
config = {"configurable": {"thread_id": "user-123"}}

# First turn
result = graph.invoke(state, config=config)
print(result["messages"][-1].content)

# User response
state = InputState(messages=[{"role": "user", "content": "I want to..."}])
result = graph.invoke(state, config=config)
```

## 🧠 Key Features

### 1. **Meta-Prompts**
Each stage has an instructional template that tells the AI *how* to generate questions:

```python
STAGE_3_META_PROMPT = """
Examine the current confirmed goal: {goal}
Generate a question that explores *why* that goal matters...
"""
```

### 2. **Knowledge Extraction**
After each user response:
- Extract goals, values, emotional tone
- Update knowledge graph weights
- Identify value conflicts
- Track conversation depth

### 3. **Adaptive Routing**
The graph automatically routes based on:
- Trust level
- Intent confirmation
- Value exploration depth
- Completion criteria

### 4. **Final Summary**
At the end, generates:
- Discovered core values (ranked by confidence)
- Primary goals with value connections
- Suggested next steps
- Request for user feedback

### 5. **Opik Integration**
Logs to Opik for analysis:
- Final feedback and summary
- Discovered values/goals
- Conversation metadata
- Stage completions (optional)

## 📊 Stage Details

### Stage 1: Rapport & Context
**Goal**: Build trust and understand what brought user here  
**Output**: One warm, open question  
**Knowledge**: Initial goal statement, desired outcome, emotional tone

### Stage 2: Reflect & Clarify
**Goal**: Confirm understanding before going deeper  
**Output**: Paraphrased understanding + confirmation question  
**Knowledge**: Validated intent, corrections

### Stage 3: Laddering
**Goal**: Explore deeper "why" behind goals  
**Output**: Progressive "why" questions (depth = 3 by default)  
**Knowledge**: Underlying values, motivations, value hierarchy

### Stage 4: Trade-off Probing
**Goal**: Reveal value priorities through conflict  
**Output**: Scenario-based trade-off question  
**Knowledge**: Value weights, relative priorities

### Stage 5: Value Reflection
**Goal**: Confirm discovered values  
**Output**: Summary + invitation to refine  
**Knowledge**: Confirmed values, user's preferred phrasing

### Stage 6: Plan Generation
**Goal**: Create value-aligned action steps  
**Output**: 2-3 suggestions explicitly linked to values  
**Knowledge**: Action suggestions with value alignment

### Stage 7: Reflection & Adaptation
**Goal**: Refine plan based on feedback  
**Output**: Question about obstacles/fit/concerns  
**Knowledge**: Obstacles, emotional fit, refinements

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-key-here
OPIK_API_KEY=your-opik-key  # Optional
OPIK_WORKSPACE=your-workspace  # Optional
```

### State Parameters
- `target_value_depth`: How many "why" iterations (default: 3)
- `trust_level`: Initial trust level (default: 0.5)
- `stage`: Current stage (1-7)

## 🎨 Design Principles

1. **Autonomy Support**: Guide, don't prescribe
2. **Psychological Safety**: Non-judgmental, validating tone
3. **Reflective Listening**: Paraphrase and confirm understanding
4. **Progressive Disclosure**: Build from surface goals to deep values
5. **Bidirectional Learning**: User can correct AI's understanding

## 📁 File Structure

```
src/meta_agent/
├── __init__.py          # Package exports
├── state.py             # State dataclasses
├── prompts.py           # Meta-prompt templates
├── knowledge.py         # Extraction & graph updates
├── graph.py             # LangGraph implementation
└── opik_logger.py       # Opik integration
```

## 🧪 Testing

```bash
# Unit tests (coming soon)
pytest tests/meta_agent/

# Interactive testing
langgraph dev
# Then use the LangGraph Studio or curl
```

## 📝 Example Conversation

```
AI: What brought you here today, and what outcome would feel meaningful to you?

User: I'm thinking about starting a side business, maybe something online.

AI: It sounds like you're exploring starting an online side business. 
    Is that accurate, or would you describe it differently?

User: Yes, exactly. I want more control over my income.

AI: Why is having control over your income important to you?

User: I guess I want more freedom and security at the same time.

AI: I'm noticing two values: freedom and security. If you had to choose 
    in your current situation, which matters more - the independence to 
    make your own choices, or the stability of reliable income?

[... conversation continues through all stages ...]

AI: 🎯 **Discovery Summary**

Based on our conversation, here's what I understand about what matters most to you:

**Core Values:**
• **Autonomy** (confidence: 85%)
  ↳ _I want control over my time and decisions_
• **Financial security** (confidence: 75%)
  ↳ _Need steady income for family_

[... continues with goals and action plan ...]

**Please provide feedback:**
• Does this accurately capture what matters to you?
• What would you change or refine?
```

## 🔮 Future Enhancements

- [ ] Vector embeddings for semantic value clustering
- [ ] Neo4j integration for richer knowledge graphs
- [ ] Multi-session memory across conversations
- [ ] Value conflict resolution strategies
- [ ] Integration with action tracking systems
- [ ] Personalization based on historical patterns

## 📚 References

Based on psychological research in:
- Self-Determination Theory (Deci & Ryan)
- Motivational Interviewing (Miller & Rollnick)
- Value-based decision making
- Reflective listening techniques

---

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) 🦜🕸️
