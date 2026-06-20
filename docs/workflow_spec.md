# ADK Exploration-Validation Workflow Specification

## 1. Architectural Overview

Currently, the pipeline uses a single `Agent` per domain (Researcher, Engineer) that is burdened with two conflicting goals:
1. Navigating search tools to gather context.
2. Generating a strictly validated JSON `Report` schema.

This causes the LLM to frequently skip tool usage to eagerly fulfill the JSON constraint. 

To solve this, we will transition to using ADK's native `SequentialAgent` to decouple the gathering phase from the reporting phase. This avoids the complexity of graph-based conditional workflows (YAGNI) while ensuring robust behavior.

The new architecture splits the workload into two strictly sequential steps per track:
**Phase 1 (Exploration)** -> **Phase 2 (Reporting)**

## 2. Sequence Nodes & Components

For each track (Academic and Practitioner), we define a `SequentialAgent` containing two sub-agents:

### Node A: `Explorer` (Agent)
- **Role**: Explores the search space using tools.
- **Constraints**: NO strict JSON output schema. It outputs raw markdown notes or pushes structured citations to the session state.
- **Prompt Requirement**: "You must use your tools to find at least 3 distinct sources before returning your final summary. Keep searching until you do." (Trusting the LLM via `max_steps`).
- **Tools**: `[search_arxiv]` (or `[search_github]`).

### Node B: `Reporter` (Agent)
- **Role**: Synthesizes the gathered information into the final structured format.
- **Constraints**: `output_schema = Report`.
- **Tools**: NONE. This prevents the agent from getting distracted by tools while writing the JSON.
- **Prompt**: "Given the following gathered research context from the Explorer, synthesize a final report. You MUST output your final response as a valid JSON object matching the schema."

## 3. ADK Sequence Definition

We will define the pipeline by composing agents using `SequentialAgent` and `ParallelAgent`:

```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent

# 1. Define the components (Academic track example)
academic_explorer = Agent(
    name="academic_explorer",
    tools=[search_arxiv],
    instruction="Explore academic sources. Find at least 3 distinct sources. Output markdown notes.",
    # No output_schema -> allows tool usage
)

academic_reporter = Agent(
    name="academic_reporter",
    output_schema=Report,
    instruction="Synthesize the explorer's notes into the final JSON Report.",
    output_key="report_1"
)

# 2. Sequence them
academic_sequence = SequentialAgent(
    name="academic_sequence",
    sub_agents=[academic_explorer, academic_reporter]
)

# Practitioner track sequence defined similarly...
# practitioner_sequence = SequentialAgent(...)

# 3. Fan-out
fanout = ParallelAgent(
    name="fanout", 
    sub_agents=[academic_sequence, practitioner_sequence]
)
```

## 4. Orchestration
The top-level pipeline runner will execute the `fanout` `ParallelAgent`. The `SequentialAgent` inside will handle the execution order cleanly, yielding the `Report` objects from the terminal `Reporter` agents to satisfy the downstream `Judge` agent exactly as the current architecture does, but without the tool-skipping bugs.
