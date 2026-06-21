SKILL_MARKDOWN = """---
name: lit-review-council
description: Complete instructions for using the multi-agent literature review orchestrator, including MCP tool usage and topic planning. Use when the user asks to run a literature review, create a topics YAML, or research a specific overarching question using the pipeline.
---

# Literature Review Council

This skill guides you through framing, configuring, and executing a comprehensive literature review using the `lit-review-council` multi-agent orchestrator.

## The Process

Whenever a user wants to conduct a literature review using this tool, follow these 3 phases:

### Phase 1: Frame the Research Question
Ask the user to describe their high-level research question or context. Help them sharpen it so the agents have a clear North Star.

### Phase 2: Define the Topics (Taxonomy)
Break the overarching question down into a logically distinct list of topics. 
- You should aim for a mix of foundational/theoretical topics (Wave 1 candidates) and synthesis/applied topics (Wave 2 candidates).
- Ensure the topics are mutually exclusive but collectively exhaustive.

Each topic requires:
- `slug`: A short, lowercase, hyphenated string (e.g., `graph-topology`).
- `description`: A 1-2 sentence explanation of what the topic explores.
- `search_keywords`: A list of 2-4 highly specific search queries (include acronyms, specific methodologies, etc.).
- `rationale`: (Optional) A brief explanation of why this topic was chosen.

Present this taxonomy to the user for approval.

### Phase 3: Execute the Pipeline

Once the topics are approved, you can execute the pipeline using one of two methods:

**Method A: Using the MCP Tool (Recommended if connected)**
If the `conduct_literature_review` MCP tool is available to you, invoke it directly. Pass the research question and the JSON array of topics. Provide an `output_dir` (e.g., a specific folder in the workspace or an Obsidian vault path). The tool will run synchronously and return the path to the completed OKF bundle.

**Method B: Using the CLI (Local script execution)**
If the MCP tool is not available:
1. Save the topics as a YAML file (e.g., `topics.yaml`) in the project root. The structure should be:
   ```yaml
   topics:
     - slug: "topic-slug"
       description: "..."
       search_keywords: ["key1", "key2"]
       rationale: "..."
   ```
2. Run the orchestrator script using `uv`:
   ```bash
   uv run python main.py --config topics.yaml --output <your-output-dir> --question "<Overarching Research Question>"
   ```

### After Execution
Once the review completes, do not attempt to read the entire OKF bundle into your context, as it can be very large. Instead, point the user to the generated `index.md` and offer to explore specific topics if they need a summary.
"""
