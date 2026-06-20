import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import random
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from google.genai.types import Content, Part
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm

eng_model = LiteLlm(model=os.getenv("ENG_MODEL") or "openrouter/deepseek/deepseek-v4-flash")
research_model = LiteLlm(model=os.getenv("RESEARCH_MODEL") or "openrouter/deepseek/deepseek-v4-flash")
judge_model = LiteLlm(model=os.getenv("JUDGE_MODEL") or "openrouter/deepseek/deepseek-v4-flash")

MAX_SOURCES = int(os.getenv("MAX_SOURCES", "10"))
MIN_SOURCES = min(MAX_SOURCES, 5)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.schema import Reference, Report, JudgeRanking, SynthesisResult, PeerReviewResult

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

from src.providers import ArxivProvider, GithubProvider, TavilyProvider, create_adk_tool

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

search_tavily = create_adk_tool(
    TavilyProvider(),
    name="search_tavily",
    description="Search the web for real-world implementation details, blogs, and documentation."
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

def get_synthesis_prompt(ctx) -> str:
    topic = ctx.session.state.get("topic")
    topic_slug = ctx.session.state.get("topic_slug")
    winning_report_id = ctx.session.state.get("winning_report_id")
    winning_role = "researcher" if winning_report_id == "report_1" else "engineer"
    rationale = ctx.session.state.get("peer_review_rationale")
    
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

EXPLORER_INSTRUCTION_TEMPLATE = f"""You are a {{role}} exploring an open technical
question using {{source_type}} sources.

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

CRITICAL: You MUST output your final response as a valid JSON object matching exactly this schema:
   {{
     "title": "A descriptive title for your report",
     "body": "The markdown formatted text of your report containing your analysis",
     "references": [
       {{"title": "Title of source", "url": "URL of source"}}
     ]
   }}
   If the `references` list is empty, all citations will be stripped from the final report.

JSON FORMATTING RULE: You MUST properly escape all JSON string values. For the 'body' field, any newlines MUST be written as `\\n` instead of actual newline characters. DO NOT include any raw control characters in strings."""

academic_explorer = Agent(
    name="academic_explorer",
    model=research_model,
    instruction=EXPLORER_INSTRUCTION_TEMPLATE.format(
        role="Researcher", 
        source_type="academic"
    ),
    tools=[search_arxiv]
)

academic_reporter = Agent(
    name="academic_reporter",
    model=research_model,
    instruction=REPORTER_INSTRUCTION_TEMPLATE.format(
        role="Researcher", 
        explorer_findings="[Context provided in the user message]"
    ),
    output_schema=Report,
    output_key="report_1"
)

academic_sequence = SequentialAgent(name="academic_sequence", sub_agents=[academic_explorer, academic_reporter])

practitioner_explorer = Agent(
    name="practitioner_explorer",
    model=eng_model,
    instruction=EXPLORER_INSTRUCTION_TEMPLATE.format(
        role="Engineer", 
        source_type="practitioner/production"
    ),
    tools=[search_github, search_tavily]
)

practitioner_reporter = Agent(
    name="practitioner_reporter",
    model=eng_model,
    instruction=REPORTER_INSTRUCTION_TEMPLATE.format(
        role="Engineer", 
        explorer_findings="[Context provided in the user message]"
    ),
    output_schema=Report,
    output_key="report_2"
)

practitioner_sequence = SequentialAgent(name="practitioner_sequence", sub_agents=[practitioner_explorer, practitioner_reporter])

ENSEMBLE_REVIEW_INSTRUCTION = """You are participating in a Peer Review Ensemble.
You will read two anonymized research reports on the same topic.
Your job is to evaluate both reports based on their accuracy, insight, and how well they balance academic rigor with real-world engineering feasibility.

Report A:
{report_a_text}

Report B:
{report_b_text}

Read both reports directly. State which report is stronger, citing specific claims or evidence from the reports themselves.
Output your evaluation as a ranking list (e.g. ["A", "B"] if A is better, or ["B", "A"] if B is better) and provide a 1-2 sentence rationale."""

def get_ensemble_instruction(ctx, role: str) -> str:
    rep_a = ctx.session.state.get('anon_report_a', '{}')
    rep_b = ctx.session.state.get('anon_report_b', '{}')
    base = ENSEMBLE_REVIEW_INSTRUCTION.format(report_a_text=rep_a, report_b_text=rep_b)
    if role == "Researcher":
        return f"You are a Research Scientist.\n\n{base}"
    elif role == "Engineer":
        return f"You are a Software Engineer.\n\n{base}"
    else:
        return f"You are a Senior Systems Architect and Principal Investigator.\n\n{base}"

researcher_reviewer = Agent(
    name="researcher_reviewer",
    model=research_model,
    instruction=lambda ctx: get_ensemble_instruction(ctx, "Researcher"),
    output_schema=JudgeRanking,
    output_key="rankings_researcher"
)

engineer_reviewer = Agent(
    name="engineer_reviewer",
    model=eng_model,
    instruction=lambda ctx: get_ensemble_instruction(ctx, "Engineer"),
    output_schema=JudgeRanking,
    output_key="rankings_engineer"
)

architect_reviewer = Agent(
    name="architect_reviewer",
    model=judge_model,
    instruction=lambda ctx: get_ensemble_instruction(ctx, "Architect"),
    output_schema=JudgeRanking,
    output_key="rankings_architect"
)

fanout = ParallelAgent(name="fanout", sub_agents=[academic_sequence, practitioner_sequence])
review_fanout = ParallelAgent(name="review_fanout", sub_agents=[researcher_reviewer, engineer_reviewer, architect_reviewer])

synthesis = Agent(
    name="synthesis",
    model=judge_model,
    instruction=get_synthesis_prompt,
    output_schema=SynthesisResult,
    output_key="synthesis_result"
)

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

async def run_pipeline(topic: str):
    session_service = InMemorySessionService()
    session_id = "sess1"
    user_id = "user1"
    
    await session_service.create_session(app_name="app", user_id=user_id, session_id=session_id)
    
    print(f"Starting pipeline for topic: {topic}")
    
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

    session_obj = await session_service.get_session(app_name="app", user_id=user_id, session_id=session_id)
    print("=== RESEARCHER REPORT ===")
    print(session_obj.state.get("report_1"))
    print("=== ENGINEER REPORT ===")
    print(session_obj.state.get("report_2"))

    # Anonymize before reviews
    reports = {
        "report_1": session_obj.state.get("report_1"),
        "report_2": session_obj.state.get("report_2"),
    }
    labels = ["A", "B"]
    random.shuffle(labels)
    anon_map = {labels[0]: "report_1", labels[1]: "report_2"}
    
    def dump(rep):
        if rep:
            return rep.model_dump_json() if hasattr(rep, "model_dump_json") else json.dumps(rep)
        return ""
        
    session_obj.state["anon_map"] = anon_map
    session_obj.state["anon_report_a"] = dump(reports[anon_map["A"]])
    session_obj.state["anon_report_b"] = dump(reports[anon_map["B"]])
    # Update state in DB just in case (though InMemorySessionService persists changes)
    
    # 2. Peer Review Ensemble
    runner_review = Runner(agent=review_fanout, session_service=session_service, app_name="app")
    print("Running Stage 2: Peer Review Ensemble...")
    
    async for event in runner_review.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(parts=[Part(text="Please evaluate and rank the anonymized reports.")])
    ):
        pass

    session_obj = await session_service.get_session(app_name="app", user_id=user_id, session_id=session_id)
    
    # 3. Tally rankings
    rank_res = session_obj.state.get("rankings_researcher")
    rank_eng = session_obj.state.get("rankings_engineer")
    rank_arc = session_obj.state.get("rankings_architect")
    
    scores = {"A": 0, "B": 0}
    reasons = []
    
    for r_obj, reviewer_name in [(rank_res, "Researcher"), (rank_eng, "Engineer"), (rank_arc, "Architect")]:
        if r_obj:
            lst = r_obj.ranking if hasattr(r_obj, "ranking") else r_obj.get("ranking", [])
            rat = r_obj.rationale if hasattr(r_obj, "rationale") else r_obj.get("rationale", "")
            if lst and len(lst) >= 1:
                # 1st place gets 2 points, 2nd gets 1 point
                if lst[0] in scores: scores[lst[0]] += 2
                if len(lst) > 1 and lst[1] in scores: scores[lst[1]] += 1
            reasons.append(f"{reviewer_name} rationale: {rat}")

    top_label = "A" if scores["A"] >= scores["B"] else "B"
    winning_report_id = anon_map.get(top_label, "report_1")
    
    delta_state = {
        "winning_report_id": winning_report_id,
        "peer_review_rationale": " | ".join(reasons)
    }
    
    print(f"Winning report ID from Ensemble: {winning_report_id} (Scores: {scores})")
    
    # 4. Synthesis with retry loop
    print("Running Stage 3: Synthesis & Persistence...")
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"Synthesis validation failed. Retrying (attempt {attempt}/{max_retries})...")
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
        
        delta_state = {}
        if not result:
            print("No synthesis result produced.")
            continue
            
        urls_cited = result.urls_cited if hasattr(result, "urls_cited") else result.get("urls_cited", [])
        markdown = result.markdown if hasattr(result, "markdown") else result.get("markdown", "")
        
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
