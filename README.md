# Multi-Agent Literature Review Pipeline

This project is a multi-agent literature review pipeline built with the [Google ADK](https://github.com/google/google-adk) (Agent Development Kit). It coordinates multiple specialized agents to research a topic from different perspectives, independently evaluates their work, and synthesizes a well-grounded final report.

This project was built as a capstone submission for the Kaggle "AI Agents: Intensive Vibe Coding Capstone".

## Motivation

When tasked with researching complex technical topics, LLMs can suffer from a lack of diverse grounding and a strong bias toward their own generated outputs (self-preference bias). 

This pipeline solves these issues through a multi-agent architecture:
1. **Source Isolation**: Instead of a single generic search, the pipeline splits work between a **Researcher** (strictly citing `arxiv.org`) and an **Engineer** (strictly citing `github.com` and official documentation).
2. **Independent Evaluation**: To eliminate self-preference bias during peer review, the generated reports are anonymized and handed off to an independent **Judge** agent. The Judge blindly evaluates the reports for accuracy and insight.
3. **Anti-Hallucination Guardrails**: A final **Synthesis** agent writes the finalized markdown report. To ensure security and factual grounding, the pipeline employs a strict validation loop: the output is parsed, and if any cited URL was not present in the original agent reports, the run is rejected and the Synthesis agent is forced to retry.

## Architecture

The system consists of 4 agents connected via ADK workflows:

- **Stage 1 (Parallel Fan-out)**: The `Researcher` and `Engineer` agents conduct domain-isolated research using DuckDuckGo search.
- **Stage 2 (Evaluation)**: Reports are shuffled and anonymized. The `Judge` evaluates them and selects a winner.
- **Stage 3 (Synthesis)**: The `Synthesis` agent writes the final brief.
- **Persistence**: File writing is handled via a lightweight MCP client wrapping `@modelcontextprotocol/server-filesystem` via `npx`, satisfying Kaggle MCP rubric requirements.

## Setup Instructions

### Prerequisites
- [uv](https://github.com/astral-sh/uv) installed for Python dependency management.
- Node.js and `npx` installed (required for the MCP filesystem server).
- A valid [Gemini API Key](https://aistudio.google.com/).

### Installation

1. Clone or download the repository.
2. Initialize the environment variables:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

Run the pipeline from the command line using `uv`. Pass your desired research topic as an argument:

```bash
uv run python src/pipeline.py "graph topology for knowledge base constraint objects"
```

### Output
The pipeline will execute the stages asynchronously and print its progress to the console. Once completed, it will append the synthesized research brief to `litreview_log.md` in the root directory.

If the Synthesis agent hallucinates a URL, the pipeline will detect it and automatically retry up to 2 times before failing loudly.
