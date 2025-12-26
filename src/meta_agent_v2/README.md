# Meta-Prompting Value Discovery Agent v2

An improved intent-aware conversational agent that uses meta-prompting to dynamically generate questions and build a knowledge graph of user values and goals.

## Key Improvements

### 1. **Context-Aware Question Generation**
- Every stage now includes full conversation history in LLM prompts
- Current state (goals, values, emotional tone) is explicitly provided to the model
- Meta-prompts instruct the AI how to generate the next question based on context

### 2. **Knowledge Graph Building**
- Structured extraction of goals, values, obstacles, and emotional tone
- Incremental knowledge updates with each user response
- Value weights that increase when mentioned multiple times
- Goal-value linking with rationale

### 3. **Multi-Stage Discovery Flow**

```
Stage 0: Introduction → Explains the journey
Stage 1: Rapport Building → Gather initial context
Stage 2: Intent Clarification → Reflect back and confirm understanding
Stage 3: Value Discovery (Laddering) → Explore "why" recursively
Stage 4: Value Trade-offs → Identify priorities through comparison
Stage 5: Value Confirmation → Reflect and validate discoveries
Stage 6: Action Planning → Generate value-aligned suggestions
Stage 7: Plan Refinement → Iterate based on feedback
Stage 8: Summary Generation → Present final discoveries
Stage 9: Feedback Collection → Log user feedback to Opik
```

## How It Works

### Meta-Prompt Architecture

Instead of hardcoding questions, each stage uses **instructional templates** that tell the LLM *how to think* about what to ask next. For example:

**Stage 3 (Laddering):**
```
Current state:
- Confirmed goal: {goal}
- Values discovered so far: {values_summary}
- Current depth: {depth}

Meta-prompt: Generate a question that explores *why* that goal matters.
If user already gave a reason, go one level deeper.
Keep question open and emotionally safe.
```

### Knowledge Extraction Pipeline

Each user response goes through:

1. **Extract** → Parse goals, values, emotions, obstacles using LLM
2. **Update** → Merge into knowledge graph with confidence scores
3. **Evaluate** → Check completeness (depth, confirmations)
4. **Generate** → Use meta-prompt + state to create next question
5. **Repeat** → Continue until convergence

### State Management

The agent maintains:
- **Goals**: Statement, confirmation status, linked values, obstacles
- **Values**: Name, weight (0-1), rationale, confirmation status  
- **Intent Context**: Goal statement, desired outcome, emotional tone
- **Actions**: Suggestions with value alignment scores
- **Conversation History**: Full message sequence

## Usage

```python
from meta_agent_v2.graph import graph
from langchain_core.messages import HumanMessage

# Start a conversation
config = {"configurable": {"thread_id": "user-123"}}

# Initial message
response = graph.invoke({
    "messages": [HumanMessage(content="I want to explore my career direction")]
}, config)

# Continue conversation
response = graph.invoke({
    "messages": [HumanMessage(content="I feel stuck in my current role")]
}, config)

# Access final summary and feedback
print(response["final_summary"])
```

## Meta-Prompt Design Principles

Each meta-prompt follows these principles:

1. **Goal Definition** - What this stage must achieve
2. **Context Provision** - Current state of knowledge graph
3. **Instruction Format** - How to generate the next question
4. **Tone Guidance** - Empathic, curious, non-judgmental
5. **Output Specification** - Exact format expected

## Knowledge Representation

```yaml
user_profile:
  goals:
    g1:
      statement: "Start a small online business"
      confirmed: true
      values: [autonomy, stability]
      rationale: ["control over time", "steady income"]
      
  values:
    autonomy:
      weight: 0.85
      confirmed: true
      rationale: ["I want to control my schedule"]
      
    stability:
      weight: 0.65
      confirmed: true
      rationale: ["need predictable income"]
      
  suggested_actions:
    - description: "Research niche markets"
      linked_values: [autonomy, stability]
      fit_score: 0.8
      user_feedback: "This feels right, but..."
```

## Feedback Loop

At the end (Stage 8), the agent:
1. Generates comprehensive summary of values & goals
2. Asks for user feedback
3. Logs everything to Opik for analysis:
   - Discovered values with weights
   - Confirmed goals
   - User's feedback on accuracy
   - Full conversation trace

This creates a learning dataset for improving the meta-prompts over time.

## Key Files

- `graph.py` - Main LangGraph implementation with stage nodes and routing
- `prompts.py` - Meta-prompt templates for each stage
- `knowledge.py` - Knowledge extraction and graph update logic
- `state.py` - Data structures (Goal, Value, UserProfile, MetaAgentState)
- `opik_logger.py` - Opik integration for feedback logging

## Future Enhancements

- [ ] Semantic clustering of values using embeddings
- [ ] Value conflict detection using vector similarity
- [ ] Personalized meta-prompt adaptation based on user profile
- [ ] Structured output parsing for action suggestions
- [ ] Multi-goal exploration with graph-based planning
- [ ] Real-time value weight updates using sentiment analysis

