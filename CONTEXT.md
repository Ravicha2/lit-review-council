# Domain Context

## Pipeline Stages

- **Stage 1 (Research Fan-out)**: Agents perform independent research. Strict **Source Isolation** is enforced at the code level by dynamically appending domain filters (e.g., `site:arxiv.org` for Researcher, `site:github.com` for Engineer) to the search tool, ensuring no overlap in raw source material.
- **Stage 2 (Independent Evaluation)**: A dedicated "Judge" agent (which did not author any reports) evaluates and ranks the anonymized reports. This eliminates self-preference bias. The Judge determines the winning report based on accuracy and insight without knowing which role authored which report.
- **Stage 3 (Synthesis & Persistence)**: A final agent writes the briefed markdown. A strict **URL Validation** step ensures every cited URL exists in the upstream reports, using **URL Normalization** (stripping scheme, `www`, trailing slashes, and sorting query params) to prevent false-positive failures from minor LLM formatting differences.

## Infrastructure

- **Kaggle MCP Requirement**: Satisfied using the official `@modelcontextprotocol/server-filesystem` executed via `npx` for the file-write tool, avoiding the complexity of building a custom Python MCP server while still meeting rubric constraints.

## Architecture Patterns

- **GatheredSources**: A lightweight Pydantic schema used to pass data from the `Gatherer` node to the `Validator` node in the ADK workflow. It separates unstructured reasoning (`notes: str`) from strictly parsed data (`urls: list[str]`), allowing the Python validator to cleanly count `len(node_input.urls)` without brittle regex parsing.
- **Validator Loop Directive**: When the `Validator` node routes back to the `Gatherer` (e.g., due to insufficient sources), it returns a specific string directive in its `Event.output`. This acts as the `node_input` for the `Gatherer`'s next turn, explicitly prompting it to use different search keywords and preventing infinite loops over identical search results.
- **Top-Level Workflow Fan-Out**: The entire pipeline orchestrates concurrency by using a parent `Workflow` with a fan-out edge `('START', (research_workflow, engineer_workflow))` and a `JoinNode` to fan-in, rather than relying on the deprecated `ParallelAgent` class or manual `asyncio.gather` scripts.
- **Validator State Aggregation**: The Python `Validator` node acts as the state accumulator. Across multiple loops, it shallow-merges the `GatheredSources` (`node_input`) into `ctx.state` arrays. Once conditions are met, it yields the combined arrays as its final `Event.output` for the `Reporter` node.
