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

def build_explorer_instruction(role: str, source_type: str, min_sources: int, max_sources: int, prior_context: Optional[str] = None) -> str:
    prompt = EXPLORER_INSTRUCTION_TEMPLATE.format(
        role=role,
        source_type=source_type,
        MIN_SOURCES=min_sources,
        MAX_SOURCES=max_sources
    )
    if prior_context:
        prompt += f"\n\nPrior research context:\n{prior_context}\n\nCRITICAL RULE: Use this prior context to refine your search queries and vocabulary, but DO NOT cite these prior findings directly in your output. You must find your own primary sources."
    return prompt

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

CRITICAL CITATION FORMAT: When you cite a source in the body text, you MUST include its exact URL as an inline Markdown link. 
Example: `This is a claim [Source Title](https://example.com/url)`.
DO NOT use footnotes like `[1]` or `(Author, Year)` without the URL. If the URL is missing from the body text, the system will reject your output.

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
        return f"You are a technical editor evaluating claim-to-evidence grounding, not source-type credibility.\n\n{base}"

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

AUDIENCE: This document is read by a student or research lead who is about to go DO the
literature review themselves. They are not a supervisor skimming for a verdict. They need
a trail they can act on: what to read first, why, what's still open, and where the evidence
actually came from. Optimizing for "sounds polished" over "is navigable" fails this reader.

CORE PRINCIPLE: Every sentence must do one of two jobs —
  (a) answer the core question (what should I do, what should I read, what's unresolved), or
  (b) let the reader verify or navigate the trail (which source says what, how strong is it,
      where do reports disagree).
Cut anything whose only job is to argue persuasively for the winning approach. Do NOT cut
the comparison table, the alternatives section, the full reference list, or any disagreement
between the two reports — those are navigation and verification aids, not argument padding,
even though they take space.
 
Strict Constraints:
 
1. TL;DR — 2-4 sentences: winning approach, one-line why, confidence level (high/medium/low
   based on how much the two reports agreed and how strong the underlying source_tier mix was).
 
2. RECOMMENDED APPROACH — state it plainly. Then a short "why" grounded in citations, but do
   not re-argue the case at length — one tight paragraph, not a persuasive essay. The reader
   should be able to act on this without reading further if they trust the trail.
 
3. COMPARISON TABLE — REQUIRED, carried over near-verbatim from whichever report(s) had one.
   If both reports have tables, merge into one. Do not compress rows into prose. This is the
   single fastest artifact for a reader doing their own evaluation — preserve it intact.
 
4. WHEN THE ALTERNATIVE IS RIGHT — REQUIRED, one short section. State plainly the conditions
   under which the non-winning approach would actually be the better choice. This is direct
   answer to a real question ("does this apply to my case"), not hedging — keep it concrete
   and specific, not generic caveats.
 
5. WHERE THE REPORTS AGREED / DISAGREED — REQUIRED if the two reports reached different
   confidence levels or cited conflicting evidence on any sub-claim. State the disagreement
   plainly and ground each side in its citation. If they fully agreed, state that in one line
   instead of omitting the section.
 
6. EVIDENCE TRAIL — REQUIRED, split into two parts:
   a. "Foundation sources" — every reference actually cited above, grouped by source_tier
      (peer_reviewed first, then established_project, vendor_doc, blog_or_forum last). For
      blog_or_forum sources used as support, flag inline: "(lower-trust source — verify
      independently before relying on this)".
   b. "Further reading, not yet evaluated" — every reference present in either report's
      references list but NOT cited in this synthesis (usage == "unevaluated" or "rejected").
      Include these even though they weren't used in the argument — they are leads for the
      reader's own search, not padding. If a source was explicitly "rejected" rather than
      simply unevaluated, say one line why, so the reader doesn't waste time re-discovering
      the same dead end.
 
7. OPEN QUESTIONS — REQUIRED if anything in either report was hedged, marked "(inferred, not
   confirmed)", or left unresolved. List these as concrete next things to check, not vague
   caveats. This section is often the most valuable part of the document for a student — do
   not shrink it for the sake of brevity.
 
8. GROUNDING AND CITATION FORMAT: Do NOT use dangling citations like (Author, Year) or [1].
   Every citation MUST be an inline Markdown link: [Source Title](URL). Every cited URL must
   already exist in the provided reports' references — never invent a citation or URL.
 
9. FRONTMATTER: The `markdown` field of your JSON output must start with YAML frontmatter
   exactly like this:
---
title: {topic}
slug: {topic_slug}
---
 
10. REFERENCES: Return the FULL list of references from both reports in the `references` JSON
    array — not just the ones cited in the markdown body. Preserve each reference's original
    `source_tier`. Set `usage` to "cited" if it appears in the synthesis markdown, otherwise
    carry over its original `usage` value ("rejected" or "unevaluated") from whichever report
    it came from.
 
LENGTH: No target length. Cut prose that re-argues a settled point. Do not cut structure,
tables, the full reference list, or open questions for the sake of brevity. A longer document
that is fully navigable beats a short one that hides its trail.
 
CRITICAL: You MUST output your final response as a valid JSON object containing exactly two
keys: "markdown" (the full report string with frontmatter) and "references" (the JSON array
of citations, including both cited and uncited/rejected ones per constraint 10). Do not output
raw markdown outside of the JSON object.
"""
    if validation_error:
        prompt += f"\n\nCRITICAL ERROR TO FIX IN THIS RETRY:\n{validation_error}\n"
        prompt += "Look closely at your previous 'markdown' text. You MUST find any occurrence of `[1]`, `[2]`, or `(Author, Year)` and REPLACE IT entirely with the proper inline markdown link `[Source Title](URL)`. Do not just add the URL at the end; you must remove the dangling citation format completely."
 
    return prompt