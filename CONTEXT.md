# Domain Context

## Glossary

- **Wave 1**: Topics that run independently, with no prior context. Determined by the Planner agent.
- **Wave 2**: Topics that benefit from Wave 1 context. Each Wave 2 topic receives distilled summaries from its Wave 1 dependencies.
- **Planner**: An LLM agent that reads topic configs and outputs a `PipelinePlan` (wave assignments + dependency mapping).
- **Distiller**: An LLM agent that condenses a full synthesis report into `DistilledContext` (key terms, conclusion, top URLs). Used to pass context from Wave 1 to Wave 2.
- **OKF (Open Knowledge Format)**: Output format for the pipeline. Each topic is a concept file with YAML frontmatter in a directory bundle, with cross-links between related topics.
- **Prior Context**: Distilled summary from Wave 1 topics, injected into Wave 2 explorer prompts. Explorers use it to refine search queries but must not cite it directly.

## Multi-Topic Orchestration

For arbitrary topic lists, the pipeline runs in two waves:

1. **Planner** reads the topic config YAML and produces a `PipelinePlan`.
2. **Wave 1** runs all independent topics in parallel (each is a full `run_pipeline` call).
3. **Distiller** produces a `DistilledContext` per Wave 1 topic.
4. **Wave 2** runs dependent topics in parallel, each receiving prior context from its Wave 1 dependencies.
5. **OKF writer** outputs all results as a concept bundle directory.

If a Wave 1 topic fails, its distillation is set to empty string. Wave 2 topics that depended on it still run, just without that context. Max 3 distillations per Wave 2 topic to control token budget.

## Single-Topic Pipeline Stages

- **Stage 1 (Research Fan-out)**: Agents perform independent research. Strict **Source Isolation** is enforced at the code level by dynamically appending domain filters to the search tool.
- **Stage 2 (Peer Review Ensemble)**: Three reviewers evaluate anonymized reports. Borda-count voting aggregates rankings.
- **Stage 3 (Synthesis & Persistence)**: A final agent writes the briefed markdown. **URL Validation** ensures every cited URL exists in upstream reports.

## Architecture Patterns

- **Validator Loop Directive**: When the `Validator` node routes back to the `Gatherer`, it returns a string directive prompting different search keywords.
- **Top-Level Workflow Fan-Out**: Concurrency via `ParallelAgent` for fan-out and `SequentialAgent` within each track.
- **Validator State Aggregation**: The `Validator` node shallow-merges `GatheredSources` across loops until conditions are met.
- **OKF Output Bundle**: Each topic produces a concept file with YAML frontmatter (`type`, `title`, `description`, `tags`, `timestamp`, `resource`). The `resource` field points to `./slug.md#references`. An `index.md` links all topics.
