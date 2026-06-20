import os
import json
import random
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from duckduckgo_search import DDGS

from google.genai.types import Content, Part
from google.adk.agents import Agent, ParallelAgent
from google.adk import Runner
from google.adk.agents.context import Context
from google.adk.sessions import InMemorySessionService

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

class Reference(BaseModel):
    title: str = Field(description="Title of the paper or resource")
    url: str = Field(description="Actual URL retrieved from the search tool")

class Report(BaseModel):
    title: str = Field(description="Short title for the report")
    body: str = Field(description="3-6 paragraph argument in plain prose")
    references: list[Reference] = Field(description="List of supporting references")

class JudgeRanking(BaseModel):
    ranking: list[str] = Field(description="Ordered list of anonymized report labels (e.g. ['A', 'B'] or ['B', 'A']), from best to worst")
    rationale: str = Field(description="Brief explanation for the first choice")

class SynthesisResult(BaseModel):
    markdown: str = Field(description="The complete synthesized markdown string including YAML frontmatter")
    urls_cited: list[str] = Field(description="List of all URLs actually cited in the markdown body")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_arxiv(query: str) -> list[dict]:
    """Search arXiv for academic papers matching the query. Always use this to ground your research."""
    ddgs = DDGS()
    try:
        results = ddgs.text(f"site:arxiv.org {query}", max_results=5)
        return results if results else []
    except Exception as e:
        return [{"error": str(e)}]

def search_github(query: str) -> list[dict]:
    """Search GitHub and official docs for engineering/practitioner evidence."""
    ddgs = DDGS()
    try:
        results = ddgs.text(f"site:github.com OR site:docs.* {query}", max_results=5)
        return results if results else []
    except Exception as e:
        return [{"error": str(e)}]

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
                existing = result.content[0].text if result.content else ""
            except Exception:
                existing = ""
            
            new_content = existing + "\n\n" + content if existing else content
            await session.call_tool("write_file", {"path": abs_path, "content": new_content})

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

async def before_judge(context: Context):
    reports = {
        "researcher": context.session.state.get("report_1"),
        "engineer": context.session.state.get("report_2"),
    }
    
    roles = ["researcher", "engineer"]
    labels = ["A", "B"]
    random.shuffle(labels)
    anon_map = {labels[0]: roles[0], labels[1]: roles[1]}
    context.session.state["anon_map_judge"] = anon_map
    
    anon_reports = []
    for lbl, role in anon_map.items():
        rep = reports[role]
        if rep:
            # We must be careful how we dump this depending on what ADK returns
            # If it's a dict, use json.dumps, if Pydantic, use model_dump_json()
            if hasattr(rep, "model_dump_json"):
                rep_json = rep.model_dump_json()
            else:
                rep_json = json.dumps(rep)
            anon_reports.append(f"--- Report {lbl} ---\n{rep_json}\n")
    
    context.session.state["anonymized_reports_text"] = "\n".join(anon_reports)

async def after_judge(context: Context):
    rankings = context.session.state.get("rankings_judge")
    if not rankings:
        return
    
    # rankings could be dict or JudgeRanking instance
    ranking_list = rankings.ranking if hasattr(rankings, "ranking") else rankings.get("ranking", [])
    rationale = rankings.rationale if hasattr(rankings, "rationale") else rankings.get("rationale", "")
    
    if not ranking_list:
        ranking_list = ["A", "B"]
        
    anon_map = context.session.state.get("anon_map_judge")
    top_label = ranking_list[0]
    winning_role = anon_map.get(top_label)
    
    context.session.state["winning_role"] = winning_role
    context.session.state["winning_report"] = context.session.state.get("report_1" if winning_role == "researcher" else "report_2")
    context.session.state["judge_rationale"] = rationale

def get_synthesis_prompt(ctx: Context) -> str:
    topic = ctx.session.state.get("topic")
    topic_slug = ctx.session.state.get("topic_slug")
    winning_role = ctx.session.state.get("winning_role")
    rationale = ctx.session.state.get("judge_rationale")
    
    rep1 = ctx.session.state.get("report_1")
    rep2 = ctx.session.state.get("report_2")
    
    def dump(rep):
        if hasattr(rep, "model_dump_json"): return rep.model_dump_json()
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
2. The output must have YAML frontmatter exactly as specified in the project spec.
3. Include the rationale and alternatives considered.
"""
    val_err = ctx.session.state.get("validation_error")
    if val_err:
        prompt += f"\n\nERROR TO FIX IN THIS RETRY:\n{val_err}"
    
    return prompt

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="You are a Researcher. Write a brief report on the topic using academic sources. Use the search tool to ground your citations.",
    tools=[search_arxiv],
    output_schema=Report,
    output_key="report_1"
)

engineer = Agent(
    name="engineer",
    model="gemini-2.5-flash",
    instruction="You are an Engineer. Write a brief report on the topic using practitioner sources. Use the search tool to ground your citations.",
    tools=[search_github],
    output_schema=Report,
    output_key="report_2"
)

judge = Agent(
    name="judge",
    model="gemini-2.5-flash",
    instruction=lambda ctx: f"You are a Judge. Evaluate the following anonymized reports and rank them by accuracy and insight. Reports:\n{ctx.session.state.get('anonymized_reports_text')}",
    before_agent_callback=before_judge,
    after_agent_callback=after_judge,
    output_schema=JudgeRanking,
    output_key="rankings_judge"
)

synthesis = Agent(
    name="synthesis",
    model="gemini-2.5-flash",
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
    fanout = ParallelAgent(name="fanout", sub_agents=[researcher, engineer])
    runner = Runner(node=fanout, session_service=session_service)
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

    # 2. Judge
    print("Running Stage 2: Independent Evaluation...")
    runner_judge = Runner(node=judge, session_service=session_service)
    async for event in runner_judge.run_async(user_id=user_id, session_id=session_id):
        pass
    
    # Let's get the state after judge
    session_obj = session_service.get_session(user_id=user_id, session_id=session_id)
    winning_role = session_obj.state.get("winning_role")
    print(f"Winning role: {winning_role}")
    
    # 3. Synthesis with retry loop
    print("Running Stage 3: Synthesis & Persistence...")
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"Synthesis validation failed. Retrying (attempt {attempt}/{max_retries})...")
            # We inject the error into state
            session_obj.state.update({"validation_error": "Previous attempt included hallucinated URLs. Only use URLs present in the original reports."})
            
        runner_synth = Runner(node=synthesis, session_service=session_service)
        async for event in runner_synth.run_async(user_id=user_id, session_id=session_id):
            pass
            
        session_obj = session_service.get_session(user_id=user_id, session_id=session_id)
        result = session_obj.state.get("synthesis_result")
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
