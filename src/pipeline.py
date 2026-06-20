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
from src.scoring import generate_anonymization_map, tally_ensemble_rankings, extract_hallucinated_urls
from src.prompts import EXPLORER_INSTRUCTION_TEMPLATE, REPORTER_INSTRUCTION_TEMPLATE, build_ensemble_instruction, build_synthesis_prompt

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

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

academic_explorer = Agent(
    name="academic_explorer",
    model=research_model,
    instruction=EXPLORER_INSTRUCTION_TEMPLATE.format(
        role="Researcher", 
        source_type="academic",
        MIN_SOURCES=MIN_SOURCES,
        MAX_SOURCES=MAX_SOURCES
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
        source_type="practitioner/production",
        MIN_SOURCES=MIN_SOURCES,
        MAX_SOURCES=MAX_SOURCES
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

researcher_reviewer = Agent(
    name="researcher_reviewer",
    model=research_model,
    instruction=lambda ctx: build_ensemble_instruction(
        "Researcher", 
        ctx.session.state.get('anon_report_a', '{}'), 
        ctx.session.state.get('anon_report_b', '{}')
    ),
    output_schema=JudgeRanking,
    output_key="rankings_researcher"
)

engineer_reviewer = Agent(
    name="engineer_reviewer",
    model=eng_model,
    instruction=lambda ctx: build_ensemble_instruction(
        "Engineer", 
        ctx.session.state.get('anon_report_a', '{}'), 
        ctx.session.state.get('anon_report_b', '{}')
    ),
    output_schema=JudgeRanking,
    output_key="rankings_engineer"
)

architect_reviewer = Agent(
    name="architect_reviewer",
    model=judge_model,
    instruction=lambda ctx: build_ensemble_instruction(
        "Architect", 
        ctx.session.state.get('anon_report_a', '{}'), 
        ctx.session.state.get('anon_report_b', '{}')
    ),
    output_schema=JudgeRanking,
    output_key="rankings_architect"
)

fanout = ParallelAgent(name="fanout", sub_agents=[academic_sequence, practitioner_sequence])
review_fanout = ParallelAgent(name="review_fanout", sub_agents=[researcher_reviewer, engineer_reviewer, architect_reviewer])

synthesis = Agent(
    name="synthesis",
    model=judge_model,
    instruction=lambda ctx: build_synthesis_prompt(
        topic=ctx.session.state.get("topic"),
        topic_slug=ctx.session.state.get("topic_slug"),
        winning_report_id=ctx.session.state.get("winning_report_id"),
        rationale=ctx.session.state.get("peer_review_rationale"),
        report_1=ctx.session.state.get("report_1"),
        report_2=ctx.session.state.get("report_2"),
        validation_error=ctx.session.state.get("validation_error")
    ),
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
    anon_map = generate_anonymization_map(["report_1", "report_2"])
    
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
    
    top_label, reasons = tally_ensemble_rankings([rank_res, rank_eng, rank_arc])
    winning_report_id = anon_map.get(top_label, "report_1")
    
    delta_state = {
        "winning_report_id": winning_report_id,
        "peer_review_rationale": " | ".join(reasons)
    }
    
    print(f"Winning report ID from Ensemble: {winning_report_id} (Winner: {top_label})")
    
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
        
        source_reports = [session_obj.state.get("report_1"), session_obj.state.get("report_2")]
        hallucinated = extract_hallucinated_urls(urls_cited, source_reports)
        
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
