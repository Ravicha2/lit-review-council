# Multi-Agent Literature Review Pipeline

A multi-agent literature review pipeline built with the [Google ADK](https://github.com/google/google-adk) (Agent Development Kit). It coordinates specialized agents to research a topic from academic and practitioner perspectives, evaluates their work through a peer review ensemble, and synthesizes a well-grounded final report.

Built as a capstone submission for the Kaggle "AI Agents: Intensive Vibe Coding Capstone".

## Motivation

LLMs researching complex topics suffer from two problems: lack of diverse grounding and self-preference bias (favoring their own outputs).

This pipeline addresses both:

1. **Source Isolation**: Two independent tracks, each with its own explorer (search) and reporter (write) agent. The academic track searches ArXiv, OpenAlex, and scholarly publishers (ACM, IEEE, Springer). The practitioner track searches GitHub and engineering docs/blogs.
2. **Peer Review Ensemble**: Three reviewers (Researcher, Engineer, Architect) evaluate anonymized reports. Borda-count voting aggregates rankings so no single reviewer dominates.
3. **Anti-Hallucination Guardrails**: The Synthesis agent's output is parsed and validated. Dangling citations like `(Author, Year)` or `[1]` are rejected. Every URL in the final report must exist in the original source references, or the run is retried (up to 2 times). A blog-tier ratio check warns when over 50% of sources are blog/forum tier.

## Architecture

```
Stage 1 (Parallel Fan-out)
├── Academic Track (SequentialAgent)
│   ├── academic_explorer  → searches ArXiv, OpenAlex, Tavily (scholarly domains)
│   └── academic_reporter  → writes Researcher report with structured references
└── Practitioner Track (SequentialAgent)
    ├── practitioner_explorer → searches GitHub, Tavily (engineering domains)
    └── practitioner_reporter → writes Engineer report with structured references

Stage 2 (Peer Review Ensemble)
├── researcher_reviewer  → ranks anonymized reports (Researcher perspective)
├── engineer_reviewer    → ranks anonymized reports (Engineer perspective)
└── architect_reviewer   → ranks anonymized reports (Architect perspective)
    → Borda-count tally → winning report selected

Stage 3 (Synthesis)
└── synthesis agent → condensed final brief with YAML frontmatter
    → citation validation loop (rejects hallucinated/dangling URLs, retries up to 2x)
    → blog-tier ratio warning if >50% sources are blog_or_forum

Persistence → MCP filesystem server writes final report to litreview_log.md
```

## Search Providers

| Provider | Domains | Used By |
|----------|---------|---------|
| ArXiv API | arxiv.org | Academic explorer |
| OpenAlex API | openalex.org | Academic explorer |
| Tavily (scholarly) | acm.org, ieee.org, springer.com, sciencedirect.com, nature.com, science.org, wiley.com | Academic explorer |
| GitHub API | github.com | Practitioner explorer |
| Tavily (engineering) | github.com, docs.microsoft.com, aws.amazon.com, cloud.google.com, medium.com, dev.to | Practitioner explorer |

All providers use tenacity retry with exponential backoff for 429/5xx errors.

## Source Tiers

Every reference is classified into one of four tiers:

- **peer_reviewed**: ArXiv preprints, ACM/IEEE papers, conference proceedings
- **established_project**: GitHub repos with meaningful adoption (stars, active maintenance)
- **vendor_doc**: Official documentation from a company/project
- **blog_or_forum**: Medium, personal blogs, Stack Overflow, Reddit

The synthesis step warns when more than half of cited sources are blog_or_forum tier.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) for Python dependency management
- Node.js and `npx` (required for the MCP filesystem server)
- An [OpenRouter](https://openrouter.ai/) API key (models route through OpenRouter)

### Installation

1. Clone the repository.
2. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env`:
   ```env
   OPENROUTER_API_KEY=your_key
   ENG_MODEL=openrouter/qwen/qwen3.5-flash-02-23
   RESEARCH_MODEL=openrouter/google/gemma-4-26b-a4b-it
   JUDGE_MODEL=openrouter/deepseek/deepseek-v4-flash
   GITHUB_TOKEN=            # optional, raises rate limits
   TAVILY_API_KEY=          # optional, enables Tavily search
   OPENALEX_API_KEY=         # optional, raises rate limits
   MAX_SOURCES=10
   ```

## Usage

```bash
uv run python src/pipeline.py "graph topology for knowledge base constraint objects"
```

### Output

The pipeline runs all stages and prints progress to the console. On completion it appends the synthesized research brief to `litreview_log.md` via the MCP filesystem server.

If the Synthesis agent hallucinates a URL or uses dangling citation syntax, the pipeline detects it and retries (up to 2 attempts before failing with an error).