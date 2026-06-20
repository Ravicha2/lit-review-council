# Multi-agent lit-review pipeline — project spec

**Stack:** Google ADK (`google-adk` 2.x), Gemini models, Google Search grounding tool.
**Output:** a research brief — recommended approach + alternatives considered, both backed by real searched references — appended to a running markdown log.
**Not in scope:** knowledge-graph/vector-store ingestion, traversal queries, entity extraction. (Deferred — see "Explicitly out of scope.")

---

## 1. Architecture

```
SequentialAgent("LitReviewPipeline")
├── ParallelAgent("research_fanout")
│     ├── LlmAgent("researcher") + search tool → state["report_1"]
│     └── LlmAgent("engineer")   + search tool → state["report_2"]
├── LlmAgent("judge")                    # independent evaluator
└── LlmAgent("synthesis_and_persist")    + file-write tool
```

Two fixed roles for v1, split by **source type**, not by reasoning style:
- **Researcher** — biased toward academic/literature evidence (papers, arXiv/
  ACM/IEEE-style sources, citation-heavy search queries like "X et al.").
- **Engineer** — biased toward practitioner/production evidence (official
  docs, GitHub repos, engineering blog posts, benchmarks, postmortems).

This is a deliberate change from an earlier "Thinker vs Engineer" (style
split) draft. A source-type split produces two genuinely different kinds
of references rather than two opinions over the same evidence pool — which
directly strengthens the "recommended vs. alternatives" output, since the
alternative is more likely to reflect a real different angle, not a
rephrasing of the same sources.

N is a config constant, not hardcoded in agent count, so a 3rd role (e.g.
"Practitioner" or "Skeptic") can be added later without restructuring the
pipeline — deferred unless the real topic clearly needs it.

### Why each stage is what it is
- **Stage 1** is plain `ParallelAgent` — no custom logic needed, ADK handles
  concurrent execution and shared state natively.
- **Stage 2** evaluates the reports using an independent Judge agent. Before calling the Judge, a custom pipeline wrapper or callback randomly maps reports to anonymized labels (e.g., A, B) so the Judge evaluates them blindly.
- **Stage 3** is plain `LlmAgent` with a file-write tool. No custom logic
  needed beyond a strict prompt constraint (see §5).

---

## 2. State schema

All keys live in `ctx.session.state` (or ADK's accepted single-leading-underscore
/ `app:`/`user:`/`temp:` prefix convention — confirm current convention against
the installed ADK version before coding, this drifts between releases).

| Key | Written by | Shape | Notes |
|---|---|---|---|
| `topic` | pipeline input | string | raw user-submitted topic |
| `topic_slug` | stage 1 entry (or pre-pipeline hook) | string | lowercase-hyphenated, deterministic from `topic` |
| `report_1` ... `report_N` | each fan-out agent | `{title: str, body: str, references: [{title, url}]}` | structured object, not raw text — see §3 |
| `anon_map_judge` | stage 2, internal only | `{A: "report_2", B: "report_1"}` | map for the judge, never exposed to any LLM prompt |
| `rankings_judge` | stage 2 | ordered list of anon labels, e.g. `["B", "A"]` | raw LLM output before de-anonymization |
| `reviews` | stage 2, final write | `{report_id: {score: float, rationale: str}}` | de-anonymized |
| `winning_report_id` | stage 2 or stage 3 | string | lowest (or highest, see §4) aggregate score |
| `final_brief_path` | stage 3 | string | path to the appended md file |

**Rule:** every key name above is final before any agent prompt is written.
If a key gets renamed mid-build, grep the whole prompt set — `SequentialAgent`
failures from stale key references are silent (empty string, not an error).

---

## 3. Report object contract (stage 1 output)

Each research agent must emit a single structured object to its `output_key`,
not free text. Recommend enforcing this via ADK's structured output / schema
support on `LlmAgent` (check current ADK docs for the exact mechanism —
`output_schema` or equivalent) rather than asking the model to format JSON
in prose and parsing it after the fact.

```
{
  "title": str,            # short, e.g. "Property-graph with typed edge constraints"
  "body": str,              # 3-6 paragraph argument, plain prose
  "references": [
    {"title": str, "url": str}   # MUST come from actual search tool results
  ]
}
```

**Hard constraint:** `references[].url` must be a URL the agent actually
retrieved via the search tool in this run. No training-knowledge citations.
This needs to be enforced two ways, not one:
1. Prompt-level instruction (necessary but not sufficient — models drift).
2. A cheap post-hoc validator in stage 3 or a pipeline callback: every URL
   in the final brief must appear in at least one `report_N.references`
   list already in state. Reject/flag at synthesis time if not. This is the
   single highest-value guardrail in the whole project — citations that
   don't resolve will sink a judged demo faster than anything else.

---

## 4. Stage 2 detail — independent evaluation

This stage evaluates the reports using a single, independent Judge agent to avoid self-preference bias. It performs three steps:

1. **Anonymize reports.** Generate a random shuffle of `{report_1...report_N} → {A, B...}`. Store the map in `anon_map_judge`.

2. **Run the Judge evaluation.** The Judge's prompt receives only the anonymized `{title, body}` of all reports (drop `references` from what the reviewer sees to prevent deanonymization via citation style). Ask the Judge to rank the reports by accuracy and insight, outputting the ordered list of labels and a brief rationale for its first choice. Store the raw output in `rankings_judge`.

3. **De-anonymize.** Map the Judge's ranking back to the real report IDs using `anon_map_judge`. Store the final results in `reviews[report_id]`. Since there is only one Judge, the top-ranked report immediately becomes the `winning_report_id`. No Borda count or complex tie-breakers are needed for N=2.

---

## 5. Stage 3 detail — synthesis and persistence

Single `LlmAgent` with a file-write tool (custom `FunctionTool` wrapping
append-to-file, or shell out — coding agent's call on mechanism).

**Inputs available in context:** `reviews`, all `report_N` objects,
`winning_report_id`, `topic`, `topic_slug`.

**Hard prompt constraints** (enforce these explicitly, not just imply them):
- May only cite URLs already present in some `report_N.references`. No new
  citations invented at synthesis time.
- Must include a "why this was chosen over the alternatives" paragraph
  derived from `reviews` scores/rationale — not a generic restatement.
- Output must match the markdown format in §6 exactly enough to be
  programmatically validated (frontmatter parses as YAML, at minimum).

**Post-write validation step** (can live in stage 3's tool or as a separate
check immediately after): parse the YAML frontmatter, confirm every URL in
the body exists in the upstream `report_N.references` data, confirm
`recommended_approach.score` and `alternatives_considered[].score` are
internally consistent with `reviews`. Fail loud (raise, don't silently
write a malformed entry) — this file is append-only and meant to be
reread later; a malformed entry is harder to fix after three more runs
have appended below it.

---

## 6. Output file format

File: `litreview_log.md`, append-only, one entry per run.

```markdown
## 2026-06-20T14:32:00Z — [Topic, human-readable]

---
topic: "Original topic string as submitted"
topic_slug: graph-topology-kb-constraint-objects
run_id: a1b2c3d4
date: 2026-06-20T14:32:00Z
agents:
  - {role: researcher, model: gemini-2.5-pro}
  - {role: engineer, model: gemini-2.5-pro}
recommended_approach:
  agent_role: engineer
  score: 1.0
  aggregation_method: borda_count
alternatives_considered:
  - {agent_role: researcher, approach: "Ontology-first schema design", score: 0.0}
---

### Recommended approach

[2-4 sentence summary: what it is, why it fits the topic.]

**Why this was chosen over the alternatives:** [2-3 sentences derived from
the peer-review step, naming the actual reasoning, not a generic line.]

**Supporting references:**
1. [Paper title](actual URL from search) — one-line relevance note
2. [Paper title](actual URL from search) — one-line relevance note

### Alternatives considered

**Ontology-first schema design** (Researcher, score 0.0)
1-2 sentence summary of this approach and why it ranked lower.
References: [Paper title](URL)

---
```

With N=2, Borda count reduces to "winner gets 1 point per reviewer who
ranked it first, loser gets 0" — i.e. simple majority vote across the 2
reviewers. Worth noting in code comments so the formula isn't mistaken for
a bug when N=2 collapses to something this simple.

`run_id`: short hash (e.g. first 8 chars of a uuid4), exists so a specific
run is unambiguously referenceable even if `topic_slug` repeats across runs.

---

## 7. Build order (matches the 14-day, uneven-attention plan)

1. **State schema + report object contract** (§2, §3) — no code yet, just
   lock the keys and shapes. Cheapest mistake to fix is one made on paper.
2. **Stage 1** (research fan-out) — build and test standalone. Confirm both
   reports land in state with correct shape, no key collisions, and that
   every reference has a real URL.
3. **Stage 3** (synthesis) — build against **hand-written fake `reviews`
   and `report_N` state**, not live stage 2 output yet. This is deliberately
   out of pipeline order: stage 3 is "boring" (no novel logic) and is also
   your demo's visible output, so get it producing correctly-formatted
   files early, decoupled from stage 2's risk.
4. **Stage 2** (anonymized review) — the one genuinely risky piece. Build
   last, against real stage 1 output. Test the anonymization in isolation
   first (assert no reviewer's prompt ever contains its own real report ID
   or any other agent's name) before trusting the rankings it produces.
5. **Wire `SequentialAgent` end-to-end**, run on 2-3 real demo topics chosen
   in advance. Lock the demo topics early so the same ones get used for
   writeup screenshots, not whatever happens to be on screen when something
   finally works.
6. **Writeup + demo polish** — per your stated priority, this gets real
   time, not leftover time. The anonymization step is the project's one
   genuinely novel idea; the demo should make it *visible* (e.g. show the
   anonymized rankings before de-anonymizing) rather than leaving it as an
   invisible implementation detail a judge has to take on faith from the
   README.

**N=2 first, N=3 only if time permits:** build and fully validate the
2-agent (Researcher/Engineer) pipeline end-to-end — steps 1-6 above —
before considering a 3rd agent. Aggregation behaves qualitatively
differently at N=2 vs N≥3 — at N=2, Borda count collapses to a simple
majority vote (§6 note), so a 3rd agent isn't just "add another
`ParallelAgent` branch," it's also the first real test of whether the
anonymization and tie-break logic (§4) generalizes. Treat it as a bounded,
optional addition late in the build, using the same locked demo topic from
step 5 — not a separate topic for the Kaggle video.

**Priority if day 14 arrives and something's still rough:** the working
pipeline for your supervisor's review comes first — that deadline has real
consequences. The Kaggle submission (§9) is explicitly lower-stakes scope
riding on top of the same build; cut from §9 before cutting anything in
§1-§7 if time runs short.

---

## 8. Kaggle capstone submission (secondary, do not over-invest)

This pipeline also doubles as a submission for the Kaggle "AI Agents:
Intensive Vibe Coding Capstone" (deadline July 6, 2026). Treat everything
in this section as low-effort bolt-on, not core scope — per explicit
direction, "just do it without being too serious."

**Rubric requires demonstrating ≥3 of:** Agent/Multi-agent system (ADK),
MCP Server, Antigravity, Security features, Deployability, Agent skills
(CLI). The pipeline already covers one for free:

1. **Multi-agent system (ADK)** — covered by the core architecture, no
   extra work.
2. **MCP Server** — wrap one existing tool (the search tool or the
   file-write tool in stage 3) as an MCP server instead of a plain
   `FunctionTool`. This is a thin protocol wrapper around logic you're
   already building, not new functionality.
3. **Security features** — the citation-validation guardrail in §3/§5
   ("every URL in the final brief must already exist in some upstream
   `report_N.references`, reject the write if not") already exists for
   correctness reasons. Frame it explicitly as a grounding/anti-hallucination
   security feature in code comments and the writeup. No new code required,
   just relabeling.

**Submission requirements** (Kaggle Writeup ≤2500 words, video ≤5 min on
YouTube, public project link or GitHub repo with setup instructions):
- Video must cover, in order, within 5 minutes: problem statement → why
  agents → architecture (the diagram from this project) → demo → build
  notes. Script this before recording; don't ad-lib a 5-minute video.
- README.md (if submitting via GitHub) needs: problem, solution,
  architecture diagram, setup instructions — per the Documentation
  criterion (20 of 100 points), this is graded, not just nice-to-have.
- Demo topic for the video should be the real lit-review topic, not a toy
  example — it's simultaneously your supervisor deliverable and your
  Kaggle demo, no need to build two.

---

## 9. Explicitly out of scope (for this submission)

- Deduplication across multiple runs on related/overlapping topics.
- N > 2 fixed roles, or fully dynamic/configurable N (deferred, not hard-blocked).
- Full mesh (N×(N-1)) cross-review — superseded by the anonymized
  every-agent-ranks-everything design (N review calls total).

These were considered during design and deliberately deferred — listing
them in the writeup as "future work" is more credible than pretending they
were never considered.

---

## 10. Open decisions still owned by the coding agent

- Exact ADK mechanism for structured `LlmAgent` output (schema/output_schema
  API — confirm against whatever ADK version is actually installed; this
  has moved between releases).
- Exact state-key prefix convention (`temp:`, `app:`, none) — confirm
  against installed ADK version, don't assume from older docs.
- File-write tool implementation (custom `FunctionTool` vs. shell-out).
- All prompt wording for every agent.