---
name: get-keyword
description: Generates a YAML topic configuration file for the literature review orchestrator based on a human-provided research context.
---

# get-keyword

This skill assists the user in breaking down a broad research question or context into a structured YAML configuration file. This YAML file is the required input for the 2-wave orchestrator pipeline.

## Instructions for the Agent

When the user invokes this skill (e.g., by asking to "generate the topic config", "get keywords for my research", or "make the yaml file"), follow these steps:

### 1. Analyze the Context
Read the user's provided research context, goals, and any specific constraints. 

### 2. Decompose into Topics
Break the overarching research question down into logically distinct topics. 
- You should aim for a mix of foundational/theoretical topics (Wave 1 candidates) and synthesis/applied topics (Wave 2 candidates).
- Ensure the topics are mutually exclusive but collectively exhaustive for the research scope.

### 3. Construct the YAML
Generate a YAML block with a root `topics` list. For each topic, you **must** provide the following fields:

- `slug`: A short, lowercase, hyphenated string used as the unique identifier (e.g., `graph-topology`).
- `description`: A 1-2 sentence explanation of what the topic explores.
- `search_keywords`: A list of 2-4 highly specific search queries. These will be directly used by the explorer agents, so include academic terminology, acronyms, and specific methodologies.
- `rationale`: A brief explanation of why this topic was chosen, its importance to the overall goal, and which other topics it might depend on.

### Example YAML Format
```yaml
topics:
  - slug: truth-maintenance-belief-revision
    description: >
      Truth maintenance and belief revision systems, specifically looking at how logical constraints handle evolving facts.
    search_keywords:
      - truth maintenance system incremental belief revision
      - JTMS
      - ATMS
    rationale: >
      Foundational for understanding constraint verdicts. Groups C and F depend on this vocabulary.
```

### 4. Output the Result
Present the generated YAML to the user in a code block. Offer to save it directly to a `topics.yaml` file in the workspace using your file editing tools if the user desires.
