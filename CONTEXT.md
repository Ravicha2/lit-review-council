# Domain Context

## Pipeline Stages

- **Stage 1 (Research Fan-out)**: Agents perform independent research. Strict **Source Isolation** is enforced at the code level by dynamically appending domain filters (e.g., `site:arxiv.org` for Researcher, `site:github.com` for Engineer) to the search tool, ensuring no overlap in raw source material.
- **Stage 2 (Independent Evaluation)**: A dedicated "Judge" agent (which did not author any reports) evaluates and ranks the anonymized reports. This eliminates self-preference bias. The Judge determines the winning report based on accuracy and insight without knowing which role authored which report.
- **Stage 3 (Synthesis & Persistence)**: A final agent writes the briefed markdown. A strict **URL Validation** step ensures every cited URL exists in the upstream reports, using **URL Normalization** (stripping scheme, `www`, trailing slashes, and sorting query params) to prevent false-positive failures from minor LLM formatting differences.

## Infrastructure

- **Kaggle MCP Requirement**: Satisfied using the official `@modelcontextprotocol/server-filesystem` executed via `npx` for the file-write tool, avoiding the complexity of building a custom Python MCP server while still meeting rubric constraints.

