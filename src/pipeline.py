import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import random
from dotenv import load_dotenv
load_dotenv()
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from ddgs import DDGS

from google.genai.types import Content, Part
from google.adk.agents import Agent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
import os

# Instantiate LiteLLM. Note: Requires OPENROUTER_API_KEY set in .env or environment
# e.g. openrouter/deepseek/deepseek-v4-flash or openrouter/kimi/kimi-2.6
eng_model = LiteLlm(model=os.getenv("ENG_MODEL") or "openrouter/deepseek/deepseek-v4-flash")
research_model = LiteLlm(model=os.getenv("RESEARCH_MODEL") or "openrouter/deepseek/deepseek-v4-flash")
judge_model = LiteLlm(model=os.getenv("JUDGE_MODEL") or "openrouter/deepseek/deepseek-v4-flash")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.schema import Reference, Report, JudgeRanking, SynthesisResult, PeerReviewResult

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

import os
from src.providers import ArxivProvider, GithubProvider, create_adk_tool

search_arxiv = create_adk_tool(
    ArxivProvider(), 
    name="search_arxiv", 
    description="Search arXiv for academic papers matching the query using the official API."
)

search_github = create_adk_tool(
    GithubProvider(), 
    name="search_github", 
    description="Search GitHub for repositories matching the query and fetch their READMEs."
)

async def write_to_filesystem_mcp(content: str, path: str):
    """Use MCP filesystem server to write the final markdown log."""
    abs_path = os.path.abspath(path)
    allowed_dir = os.path.dirname(abs_path)
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", allowed_dir]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                result = await session.call_tool("read_file", {"path": abs_path})
                if getattr(result, "isError", False):
                    existing = ""
                else:
                    existing = result.content[0].text if result.content else ""
                    if "ENOENT" in existing and "no such file" in existing:
                        existing = ""
            except Exception:
                existing = ""
            
            new_content = existing + "\n\n" + content if existing else content
            await session.call_tool("write_file", {"path": abs_path, "content": new_content})

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

async def before_judge(*, callback_context, **kwargs):
    reports = {
        "report_1": callback_context.session.state.get("report_1"),
        "report_2": callback_context.session.state.get("report_2"),
    }
    
    labels = ["A", "B"]
    random.shuffle(labels)
    anon_map = {labels[0]: "report_1", labels[1]: "report_2"}
    callback_context.session.state["anon_map_judge"] = anon_map
    
    def dump(rep):
        if rep:
            return rep.model_dump_json() if hasattr(rep, "model_dump_json") else json.dumps(rep)
        return ""
        
    callback_context.session.state["anon_report_a"] = dump(reports[anon_map["A"]])
    callback_context.session.state["anon_report_b"] = dump(reports[anon_map["B"]])
    
    rev_res = callback_context.session.state.get("review_researcher")
    rev_eng = callback_context.session.state.get("review_engineer")
    
    def get_rev_info(rev):
        if not rev:
            return "Unknown", "No rationale"
        defer = rev.defer if hasattr(rev, "defer") else rev.get("defer", False)
        rationale = rev.rationale if hasattr(rev, "rationale") else rev.get("rationale", "")
        verdict = "Defer to this report" if defer else "Do not defer to this report"
        return verdict, rationale
        
    res_verdict, res_rationale = get_rev_info(rev_res)
    eng_verdict, eng_rationale = get_rev_info(rev_eng)

    if anon_map["A"] == "report_1":
        callback_context.session.state["review_a_verdict"] = eng_verdict
        callback_context.session.state["review_a_reason"] = eng_rationale
        callback_context.session.state["review_b_verdict"] = res_verdict
        callback_context.session.state["review_b_reason"] = res_rationale
    else:
        callback_context.session.state["review_a_verdict"] = res_verdict
        callback_context.session.state["review_a_reason"] = res_rationale
        callback_context.session.state["review_b_verdict"] = eng_verdict
        callback_context.session.state["review_b_reason"] = eng_rationale


async def after_judge(*, callback_context, **kwargs):
    rankings = callback_context.session.state.get("rankings_judge")
    if not rankings:
        return
    
    ranking_list = rankings.ranking if hasattr(rankings, "ranking") else rankings.get("ranking", [])
    rationale = rankings.rationale if hasattr(rankings, "rationale") else rankings.get("rationale", "")
    
    if not ranking_list:
        ranking_list = ["A", "B"]
        
    anon_map = callback_context.session.state.get("anon_map_judge")
    top_label = ranking_list[0]
    winning_report_id = anon_map.get(top_label)
    
    callback_context.session.state["winning_report_id"] = winning_report_id
    callback_context.session.state["judge_rationale"] = rationale

def get_synthesis_prompt(ctx) -> str:
    topic = ctx.session.state.get("topic")
    topic_slug = ctx.session.state.get("topic_slug")
    winning_report_id = ctx.session.state.get("winning_report_id")
    winning_role = "researcher" if winning_report_id == "report_1" else "engineer"
    rationale = ctx.session.state.get("judge_rationale") or ctx.session.state.get("peer_review_rationale")
    
    rep1 = ctx.session.state.get("report_1")
    rep2 = ctx.session.state.get("report_2")
    
    def dump(rep):
        if hasattr(rep, "model_dump_json"): return rep.model_dump_json()
        if isinstance(rep, str): return rep
        return json.dumps(rep)
    
    prompt = f"""Synthesize the final litreview_log.md entry for the topic: {topic}.
Topic slug: {topic_slug}
Winning approach was from the {winning_role} agent, with rationale: {rationale}.

Report 1 (Researcher):
{dump(rep1)}

Report 2 (Engineer):
{dump(rep2)}

Strict Constraints:
1. Ensure every cited URL in your markdown body already exists in the provided reports' references.
2. The output must have YAML frontmatter exactly like this:
---
title: {topic}
slug: {topic_slug}
---
3. Include the rationale and alternatives considered.
"""
    val_err = ctx.session.state.get("validation_error")
    if val_err:
        prompt += f"\n\nERROR TO FIX IN THIS RETRY:\n{val_err}"
    
    return prompt

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

REVIEW_INSTRUCTION = """You are reviewing a research report written by a colleague
on the same topic. You did not write this report. Evaluate it on its own merits.

Report to review:
{other_report_text}

Score it 0-10 on two dimensions:
- accuracy: are the claims well-supported by the cited sources?
- insight: does it surface a genuinely useful, non-obvious recommendation?

Then state: would you defer to this report's recommended approach over your own,
yes or no, and why in one sentence.

Do not summarize the report back. Output only the evaluation."""

def get_review_instruction(ctx, other_report_key: str) -> str:
    other_report = ctx.session.state.get(other_report_key)
    report_text = other_report.model_dump_json() if hasattr(other_report, "model_dump_json") else json.dumps(other_report) if other_report else ""
    return REVIEW_INSTRUCTION.format(other_report_text=report_text)

researcher = Agent(
    name="researcher",
    model=research_model,
    instruction="""You are a Researcher investigating an open technical
question using academic sources.

OUTPUT REQUIREMENTS — these are not optional:

1. Every substantive claim must cite a specific source you actually retrieved
   via the search tool. Do not write more than 2-3 sentences in a row without
   a citation. If you cannot find a source for a claim, do not make the claim.
   
   CRITICAL: You MUST include every cited source in the `references` array of 
   the JSON output. If the `references` list is empty, all citations will be 
   stripped from the final report.

2. If the question has identifiable alternatives or a fork (approach A vs B),
   present them as a comparison table with concrete rows — capability,
   property graphs, etc, whatever dimensions the evidence actually supports.
   Pull specific facts into table cells, not vague adjectives like "more
   flexible." Prefer "O(neighbors) traversal vs full node scan" over
   "more efficient."

3. For your non-preferred alternative, include a short section: "When
   [alternative] is actually the right choice" — name the specific
   conditions under which someone should pick the option you're not
   recommending. A recommendation with no honest exceptions is a weaker
   recommendation.

4. End with a single-paragraph verdict: which approach you recommend, stated
   as a position, and your confidence. State what evidence would change
   your mind.

5. Do not summarize the literature neutrally. Your job is to synthesize
   evidence into a defensible recommendation, not to report what exists.

If your search tool returns too few sources to support a claim with this
density, say so explicitly rather than padding with unsupported reasoning —
note it as a gap in available evidence, not a confident claim.""",
    tools=[search_arxiv],
    output_schema=Report,
    output_key="report_1"
)

engineer = Agent(
    name="engineer",
    model=eng_model,
    instruction="""You are an Engineer investigating an open technical
question using practitioner/production sources (real codebases, docs,
production patterns) rather than academic papers.

OUTPUT REQUIREMENTS — these are not optional:

1. Every substantive claim must cite a specific source you actually retrieved
   via the search tool. Do not write more than 2-3 sentences in a row without
   a citation. If you cannot find a source for a claim, do not make the claim.

   CRITICAL: You MUST include every cited source in the `references` array of 
   the JSON output. If the `references` list is empty, all citations will be 
   stripped from the final report.

2. If the question has identifiable alternatives or a fork (approach A vs B),
   present them as a comparison table with concrete rows — capability,
   property graphs, etc, whatever dimensions the evidence actually supports.
   Pull specific facts into table cells, not vague adjectives like "more
   flexible." Prefer "O(neighbors) traversal vs full node scan" over
   "more efficient."

3. For your non-preferred alternative, include a short section: "When
   [alternative] is actually the right choice" — name the specific
   conditions under which someone should pick the option you're not
   recommending. A recommendation with no honest exceptions is a weaker
   recommendation.

4. End with a single-paragraph verdict: which approach you recommend, stated
   as a position, and your confidence. State what evidence would change
   your mind.

5. Do not summarize the literature neutrally. Your job is to synthesize
   evidence into a defensible recommendation, not to report what exists.

Favor concrete evidence from real systems over abstract argument: cite
specific projects, specific schema/API choices they made, and specific
numbers (performance, scale, error rates) where the source provides them.
A claim like "the knowing project uses 38 edge types with weighted RWR
traversal" is the standard to hit — not "edge typing is common practice.\"""",
    tools=[search_github],
    output_schema=Report,
    output_key="report_2"
)

researcher_reviewer = Agent(
    name="researcher_reviewer",
    model=research_model,
    instruction=lambda ctx: get_review_instruction(ctx, "report_2"),
    output_schema=PeerReviewResult,
    output_key="review_researcher"
)

engineer_reviewer = Agent(
    name="engineer_reviewer",
    model=eng_model,
    instruction=lambda ctx: get_review_instruction(ctx, "report_1"),
    output_schema=PeerReviewResult,
    output_key="review_engineer"
)

judge = Agent(
    name="judge",
    model=judge_model,
    instruction=lambda ctx: f"""You are a tiebreaker. Two reviewers evaluated
two reports and reached conflicting verdicts on which is stronger. Your job is
to resolve the disagreement, not re-review from scratch.

Report A:
{ctx.session.state['anon_report_a']}

Report B:
{ctx.session.state['anon_report_b']}

Reviewer of Report A's verdict: {ctx.session.state.get('review_a_verdict')}
Reviewer of Report A's reasoning: {ctx.session.state.get('review_a_reason')}

Reviewer of Report B's verdict: {ctx.session.state.get('review_b_verdict')}
Reviewer of Report B's reasoning: {ctx.session.state.get('review_b_reason')}

Read both reports directly — do not just defer to one reviewer's reasoning
without checking it against the actual report content. State which report
is stronger, citing specific claims or evidence from the reports themselves
(not just "reviewer 1 was right"). If you find both reviewers were
reasoning from weak or comparable evidence, say so explicitly rather than
picking arbitrarily — a forced low-confidence pick is still useful, but
flag the low confidence.

Output: which report wins, your confidence (high/medium/low), and a
1-2 sentence reason grounded in the report content.""",
    before_agent_callback=before_judge,
    after_agent_callback=after_judge,
    output_schema=JudgeRanking,
    output_key="rankings_judge"
)

synthesis = Agent(
    name="synthesis",
    model=judge_model,
    instruction=get_synthesis_prompt,
    output_schema=SynthesisResult,
    output_key="synthesis_result"
)

fanout = ParallelAgent(name="fanout", sub_agents=[researcher, engineer])
review_fanout = ParallelAgent(name="review_fanout", sub_agents=[researcher_reviewer, engineer_reviewer])

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

async def run_pipeline(topic: str):
    session_service = InMemorySessionService()
    session_id = "sess1"
    user_id = "user1"
    
    await session_service.create_session(app_name="app", user_id=user_id, session_id=session_id)
    
    def get_state(runner: Runner):
        # We can extract the state from the session_service
        # For simplicity we'll just run individual agents if Runner doesn't share state well natively.
        pass

    # Wait, Runner wraps a single node. We can create a SequentialAgent for Judge and Synthesis, 
    # but the ParallelAgent executes the fanout.
    # To keep state consistent across calls, we need a single workflow or we manually pass the session ID.
    
    print(f"Starting pipeline for topic: {topic}")
    
    # Let's create a root agent that orchestrates this, or just run them via Runner.
    # If we run them sequentially on the same session, they share state!
    
    # 1. Fanout
    runner = Runner(agent=fanout, session_service=session_service, app_name="app")
    print("Running Stage 1: Research Fan-out...")
    
    init_state = {
        "topic": topic,
        "topic_slug": topic.lower().replace(" ", "-"),
        "validation_error": None
    }
    
    async for event in runner.run_async(
        user_id=user_id, 
        session_id=session_id, 
        state_delta=init_state,
        new_message=Content(parts=[Part(text=f"Research topic: {topic}")])
    ):
        pass

    # 2. Peer Review
    runner_review = Runner(agent=review_fanout, session_service=session_service, app_name="app")
    print("Running Stage 2: Peer Review...")
    
    async for event in runner_review.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(parts=[Part(text="Please review the other report.")])
    ):
        pass

    session_obj = await session_service.get_session(app_name="app", user_id=user_id, session_id=session_id)
    review_researcher = session_obj.state.get("review_researcher")
    review_engineer = session_obj.state.get("review_engineer")
    
    r_defer = review_researcher.defer if hasattr(review_researcher, "defer") else review_researcher.get("defer", False)
    e_defer = review_engineer.defer if hasattr(review_engineer, "defer") else review_engineer.get("defer", False)

    # 3. Judge (if needed)
    delta_state = {}
    needs_judge = False
    if r_defer and not e_defer:
        # Researcher defers to Engineer, Engineer favors its own
        delta_state["winning_report_id"] = "report_2"
        delta_state["peer_review_rationale"] = "Net agreement to use engineer's report."
    elif not r_defer and e_defer:
        # Engineer defers to Researcher, Researcher favors its own
        delta_state["winning_report_id"] = "report_1"
        delta_state["peer_review_rationale"] = "Net agreement to use researcher's report."
    else:
        # Genuine disagreement or tie
        needs_judge = True
        
    if needs_judge:
        print("Running Stage 3: Independent Evaluation (Tie-breaker)...")
        runner_judge = Runner(agent=judge, session_service=session_service, app_name="app")
        async for event in runner_judge.run_async(
            user_id=user_id, 
            session_id=session_id,
            state_delta=delta_state,
            new_message=Content(parts=[Part(text="Please evaluate the reports.")])
        ):
            pass
        
        session_obj = await session_service.get_session(app_name="app", user_id=user_id, session_id=session_id)
        winning_report_id = session_obj.state.get("winning_report_id")
        print(f"Winning report ID from Judge: {winning_report_id}")
    else:
        winning_report_id = delta_state.get("winning_report_id")
        print(f"Winning report ID from Peer Review: {winning_report_id}")
    
    # 4. Synthesis with retry loop
    print("Running Stage 4: Synthesis & Persistence...")
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"Synthesis validation failed. Retrying (attempt {attempt}/{max_retries})...")
            # We inject the error into state
            session_obj.state.update({"validation_error": "Previous attempt included hallucinated URLs. Only use URLs present in the original reports."})
            
        runner_synth = Runner(agent=synthesis, session_service=session_service, app_name="app")
        async for event in runner_synth.run_async(
            user_id=user_id, 
            session_id=session_id,
            state_delta=delta_state,
            new_message=Content(parts=[Part(text="Please synthesize the final report.")])
        ):
            pass
            
        session_obj = await session_service.get_session(app_name="app", user_id=user_id, session_id=session_id)
        result = session_obj.state.get("synthesis_result")
        
        # Clear delta_state after first attempt so we only merge it once (or we can keep it, it's fine)
        # But we must ensure validation error clears if we retry
        delta_state = {}
        if not result:
            print("No synthesis result produced.")
            continue
            
        # Parse result
        urls_cited = result.urls_cited if hasattr(result, "urls_cited") else result.get("urls_cited", [])
        markdown = result.markdown if hasattr(result, "markdown") else result.get("markdown", "")
        
        # Validation
        valid_urls = set()
        for rep_key in ["report_1", "report_2"]:
            rep = session_obj.state.get(rep_key)
            if rep:
                refs = rep.references if hasattr(rep, "references") else rep.get("references", [])
                for ref in refs:
                    url = ref.url if hasattr(ref, "url") else ref.get("url", "")
                    valid_urls.add(url.strip().lower())
        
        hallucinated = []
        for url in urls_cited:
            if url.strip().lower() not in valid_urls:
                hallucinated.append(url)
        
        if not hallucinated:
            log_path = "litreview_log.md"
            print("Synthesis successful. Writing to log via MCP...")
            await write_to_filesystem_mcp(markdown, log_path)
            print("Done!")
            break
        elif attempt == max_retries:
            raise ValueError(f"Failed to synthesize valid URLs after {max_retries} retries. Hallucinated: {hallucinated}")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "graph topology for knowledge base constraint objects"
    asyncio.run(run_pipeline(topic))
