import json
from typing import Optional, Any

EXPLORER_INSTRUCTION_TEMPLATE = """You are a {role} exploring an open technical
question using {source_type} sources.

1. Use your search tools to find sources that directly address the core
   question — not just tangentially related material. Find at least {MIN_SOURCES} such
   sources, up to a maximum of {MAX_SOURCES} search calls. If after {MAX_SOURCES} calls you still
   don't have {MIN_SOURCES} strong sources, stop and report what you found plus what's
   missing — don't keep searching indefinitely.
2. Output a markdown summary of your findings and the exact URLs you found,
   noting for each source how directly it addresses the question."""

REPORTER_INSTRUCTION_TEMPLATE = """You are a {role} synthesizing a final report
from the research context gathered below.

Research context from explorer:
{explorer_findings}

OUTPUT REQUIREMENTS — not optional:
1. Every substantive claim must cite a specific source from the explorer's
   findings. Do not write more than 2-3 sentences in a row without a citation.
   Do not introduce claims or sources the explorer did not find.
2. For claims about what a specific tool does or doesn't do, cite something
   more specific than a bare repo link (a file, README section, or doc
   passage) OR explicitly mark the claim as inferred, not confirmed.
3. If there's an identifiable fork between alternatives, present a comparison
   table with concrete rows, not vague adjectives.
4. Include a short section on when the non-preferred alternative is actually
   the right choice.
5. End with a position-stated verdict and confidence.
6. IF THE EXPLORER'S FINDINGS ARE THIN OR TANGENTIAL: say so explicitly in
   the report rather than writing a confident recommendation anyway. A report
   that honestly states "evidence here is limited, treat this as a starting
   point not a conclusion" is more useful than a polished report built on
   weak grounding.
7. For any claim that materially affects your recommendation (not minor context),
   do not rely solely on a blog_or_forum source. If your strongest evidence for a
   load-bearing claim is blog_or_forum tier, either find a stronger source or
   explicitly flag the claim as weakly supported in the body text.

CRITICAL: You MUST output your final response as a valid JSON object matching exactly this schema:
   {{
     "title": "A descriptive title for your report",
     "body": "The markdown formatted text of your report containing your analysis",
     "references": [
       {{
         "title": "Title of source",
         "url": "URL of source",
         "source_tier": "peer_reviewed | established_project | vendor_doc | blog_or_forum",
         "usage": "cited | rejected | unevaluated"
       }}
     ]
   }}
   If the `references` list is empty, all citations will be stripped from the final report.
   
   Definitions for source_tier:
   - peer_reviewed: arXiv preprints, ACM/IEEE papers, conference proceedings
   - established_project: GitHub repos with meaningful adoption signal (stars, active maintenance, used in production by named orgs)
   - vendor_doc: official documentation from a company/project
   - blog_or_forum: Medium, personal blogs, Stack Overflow, Reddit, forum posts

   Definitions for usage:
   - cited: You used this source to support a claim and explicitly included its URL in the body text.
   - rejected: You evaluated this source but found it irrelevant or weak, so you did not cite it in the body.
   - unevaluated: You found this source but did not evaluate it.

JSON FORMATTING RULE: You MUST properly escape all JSON string values. For the 'body' field, any newlines MUST be written as `\\n` instead of actual newline characters. DO NOT include any raw control characters in strings."""

ENSEMBLE_REVIEW_INSTRUCTION = """You are participating in a Peer Review Ensemble.
You will read two anonymized research reports on the same topic.
Your job is to evaluate both reports based on their accuracy, insight, and how well they balance academic rigor with real-world engineering feasibility.

Report A:
{report_a_text}

Report B:
{report_b_text}

Read both reports directly. State which report is stronger, citing specific claims or evidence from the reports themselves.

CRITICAL: You MUST output your evaluation as a valid JSON object matching exactly this schema:
{{
  "ranking": ["A", "B"] or ["B", "A"],
  "rationale": "Provide a 1-2 sentence rationale citing specific evidence."
}}

JSON FORMATTING RULE: Do NOT wrap the JSON in markdown blocks (e.g., no ```json). Output ONLY the raw JSON object. You MUST properly escape all JSON string values."""

def build_ensemble_instruction(role: str, report_a_text: str, report_b_text: str) -> str:
    """Returns the pure formatted instruction string for an ensemble reviewer role."""
    base = ENSEMBLE_REVIEW_INSTRUCTION.format(report_a_text=report_a_text, report_b_text=report_b_text)
    if role == "Researcher":
        return f"You are a Research Scientist.\n\n{base}"
    elif role == "Engineer":
        return f"You are a Software Engineer.\n\n{base}"
    else:
        return f"You are a Senior Systems Architect and Principal Investigator.\n\n{base}"

def build_synthesis_prompt(
    topic: str,
    topic_slug: str,
    winning_report_id: str,
    rationale: str,
    report_1: Any,
    report_2: Any,
    validation_error: Optional[str] = None
) -> str:
    """Returns the pure formatted prompt string for the synthesis step."""
    winning_role = "researcher" if winning_report_id == "report_1" else "engineer"
    
    def dump(rep):
        if hasattr(rep, "model_dump_json"): return rep.model_dump_json()
        if isinstance(rep, str): return rep
        return json.dumps(rep)
    
    prompt = f"""Synthesize the final, highly-condensed litreview_log.md entry for the topic: {topic}.
Topic slug: {topic_slug}
Winning approach was from the {winning_role} agent, with rationale: {rationale}.

Report 1 (Researcher):
{dump(report_1)}

Report 2 (Engineer):
{dump(report_2)}

Strict Constraints:
1. OPTIMIZE FOR HUMAN REVIEW: Focus entirely on brevity. Extract only the most critical insights, eliminating fluff and redundant context.
2. SCANNABILITY: Use a clear "TL;DR" section, bullet points, and bold text for key terms.
3. ACTIONABLE TAKEAWAYS: Clearly highlight the final decisions and next steps based on the winning approach.
4. CONDENSED ALTERNATIVES: Briefly summarize the rationale and alternatives considered, focusing only on the essential differences that drove the final decision.
5. GROUNDING AND CITATION FORMAT: DO NOT use dangling citations like (Author, Year) or [1]. Every citation MUST be an inline Markdown link: [Source Title](URL). Ensure every cited URL already exists in the provided reports' references.
6. FRONTMATTER: The `markdown` field of your JSON output must start with YAML frontmatter exactly like this:
---
title: {topic}
slug: {topic_slug}
---
7. REFERENCES: You must return the full list of references you cited in your markdown in the `references` JSON array. Make sure to preserve the `source_tier` of each reference from the original reports.

CRITICAL: You MUST output your final response as a valid JSON object containing exactly two keys: "markdown" (the full report string with frontmatter) and "references" (the JSON array of citations). Do not output raw markdown outside of the JSON object.
"""
    if validation_error:
        prompt += f"\n\nERROR TO FIX IN THIS RETRY:\n{validation_error}"
    
    return prompt
